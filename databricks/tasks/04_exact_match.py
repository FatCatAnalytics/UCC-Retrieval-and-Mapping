# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Exact match (raw + normalized) → `ucc.gold.exact_mapped`
# MAGIC
# MAGIC Joins cleaned UCC against `master_list_current` on
# MAGIC `(DEBT_STATE, SECP_STATE, SECP_NAME)` raw, then on the normalized
# MAGIC name. Anything still unmatched is left in a separate Delta table
# MAGIC for the next stage to pick up.
# MAGIC
# MAGIC Reuses the original Dagster matching logic via
# MAGIC `databricks.shared.matching` so behavior stays identical.

# COMMAND ----------

# MAGIC %pip install --quiet pandas rapidfuzz dagster pydantic

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
_DAGSTER_PROJECT = os.path.join(_REPO_ROOT, "ucc_mapping_project")
for p in (_REPO_ROOT, _DAGSTER_PROJECT):
    if p not in sys.path:
        sys.path.insert(0, p)

from databricks.shared.config import load_config
from databricks.shared import paths
from databricks.shared.audit import PipelineRunLogger, ensure_business_schemas
from databricks.shared.io import read_csv_safe, overwrite_partition
from databricks.shared import matching as M

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

import pandas as pd
import tempfile

# Materialize master_list_current -> a temp CSV that the existing
# loader can read. We do this so we can reuse make_pipeline_config /
# load_master_mapping without rewriting the master loader.
with PipelineRunLogger(spark, cfg, task_name="04_exact_match") as run:
    cleaned = (
        spark.table(paths.t_ucc_clean(cfg))
        .where(f"source_month = '{cfg.source_month}'")
        .toPandas()
    )
    run.row_count_in = len(cleaned)
    print(f"Loaded {len(cleaned):,} cleaned UCC rows")

    master_pdf = spark.table(paths.t_master_list_current(cfg)).toPandas()
    print(f"Loaded {len(master_pdf):,} master_list_current rows")

    # Persist to a tmp dir so the Dagster asset functions can read them
    workdir = tempfile.mkdtemp(prefix="ucc_exact_")
    ucc_path = os.path.join(workdir, "ucc.csv")
    master_path = os.path.join(workdir, "master_list.csv")
    universe_path = os.path.join(workdir, "tracked_banks.csv")  # unused here, must exist

    # The Dagster master loader expects 'Firm ID' header (not Firm_ID)
    master_for_csv = master_pdf.copy()
    if "Firm_ID" in master_for_csv.columns and "Firm ID" not in master_for_csv.columns:
        master_for_csv = master_for_csv.rename(columns={"Firm_ID": "Firm ID"})

    cleaned.to_csv(ucc_path, index=False)
    master_for_csv.to_csv(master_path, index=False)
    pd.DataFrame(columns=["Bank Name", "Firm ID"]).to_csv(universe_path, index=False)

    pl_cfg = M.make_pipeline_config(
        ucc_file=ucc_path,
        master_file=master_path,
        universe_file=universe_path,
        output_dir=workdir,
        embedding_model=cfg.model_dir,
    )

    ucc_df = M.load_ucc_raw(pl_cfg)
    master_df = M.load_master_mapping(pl_cfg)
    norm_master = M.build_normalized_master(master_df)

    after_raw = M.stage_exact_raw(ucc_df, norm_master)
    after_norm = M.stage_exact_normalized(after_raw, norm_master)

    matched = after_norm[after_norm["SECP_NAME_MAPPED"].notna()].copy()
    unmatched = after_norm[after_norm["SECP_NAME_MAPPED"].isna()].copy()

    matched["source_month"] = cfg.source_month
    matched["pipeline_run_id"] = cfg.pipeline_run_id
    unmatched["source_month"] = cfg.source_month
    unmatched["pipeline_run_id"] = cfg.pipeline_run_id

    n_matched = overwrite_partition(
        spark,
        matched,
        table=paths.t_exact_mapped(cfg),
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.gold_schema}",
    )
    # Stash the unmatched portion in a private silver staging table
    staging_table = cfg.table(cfg.silver_schema, "ucc_unmatched_after_exact")
    n_unmatched = overwrite_partition(
        spark,
        unmatched,
        table=staging_table,
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.silver_schema}",
    )
    run.row_count_out = n_matched + n_unmatched

    print(f"Exact-matched rows: {n_matched:,} -> {paths.t_exact_mapped(cfg)}")
    print(f"Pending rows:       {n_unmatched:,} -> {staging_table}")

# COMMAND ----------

dbutils.notebook.exit(f"matched={n_matched};unmatched={n_unmatched}")
