# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Clean / normalize UCC bronze → `ucc.silver.ucc_clean`
# MAGIC
# MAGIC Standardizes state columns, drops rows with empty `SECP_NAME`,
# MAGIC computes `SECP_NAME_NORMALIZED` using
# MAGIC :func:`ucc_mapping_project.utils.normalize_name` (so Databricks
# MAGIC and Dagster produce identical normalized strings).

# COMMAND ----------

# MAGIC %pip install --quiet pandas rapidfuzz

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Make the original Dagster project importable (we reuse its utils)
_DAGSTER_PROJECT = os.path.join(_REPO_ROOT, "ucc_mapping_project")
if _DAGSTER_PROJECT not in sys.path:
    sys.path.insert(0, _DAGSTER_PROJECT)

from databricks.shared.config import load_config
from databricks.shared import paths
from databricks.shared.audit import PipelineRunLogger, ensure_business_schemas
from databricks.shared.io import overwrite_partition

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

import pandas as pd
from ucc_mapping_project.ucc_mapping_project.utils import normalize_name
from ucc_mapping_project.ucc_mapping_project.assets import (
    _standardize_state_columns,
    RAW_MATCH_COLUMNS,
)

bronze = paths.t_ucc_raw(cfg)
print(f"Reading {bronze} for source_month={cfg.source_month}")

with PipelineRunLogger(spark, cfg, task_name="02_clean_ucc") as run:
    sdf = spark.table(bronze).where(f"source_month = '{cfg.source_month}'")
    pdf = sdf.toPandas()
    run.row_count_in = len(pdf)

    keep = [c for c in RAW_MATCH_COLUMNS if c in pdf.columns]
    pdf = pdf[keep + [c for c in pdf.columns if c not in keep]].copy()
    pdf = _standardize_state_columns(pdf)
    pdf = pdf[pdf["SECP_NAME"].notna() & (pdf["SECP_NAME"].str.strip() != "")].copy()
    pdf = pdf.drop_duplicates(subset=RAW_MATCH_COLUMNS).reset_index(drop=True)

    pdf["SECP_NAME_NORMALIZED"] = pdf["SECP_NAME"].apply(normalize_name)
    pdf["source_month"] = cfg.source_month
    pdf["pipeline_run_id"] = cfg.pipeline_run_id

    written = overwrite_partition(
        spark,
        pdf,
        table=paths.t_ucc_clean(cfg),
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.silver_schema}",
    )
    run.row_count_out = written
    print(f"Wrote {written:,} cleaned rows to {paths.t_ucc_clean(cfg)}")

# COMMAND ----------

dbutils.notebook.exit(str(written))
