# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Fuzzy + embedding + cluster + suggest → `ucc.gold.review_candidates`
# MAGIC
# MAGIC Runs stages 3-7 from the original pipeline (fuzzy, advanced
# MAGIC embedding, clustering, hierarchical merge, suggested mappings),
# MAGIC plus the tracked-banks Firm_ID enrichment, and writes the unioned
# MAGIC pre-review candidate set to Delta.
# MAGIC
# MAGIC The output is the input the business reviewers will see in the
# MAGIC review workbook (task 06).

# COMMAND ----------

# MAGIC %pip install --quiet pandas rapidfuzz sentence-transformers faiss-cpu dagster pydantic

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
from databricks.shared.io import overwrite_partition, read_csv_safe
from databricks.shared import matching as M

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

import tempfile
import pandas as pd

with PipelineRunLogger(spark, cfg, task_name="05_fuzzy_embedding_match") as run:
    staging_table = cfg.table(cfg.silver_schema, "ucc_unmatched_after_exact")
    pending = (
        spark.table(staging_table)
        .where(f"source_month = '{cfg.source_month}'")
        .toPandas()
    )
    run.row_count_in = len(pending)
    print(f"Pending unmatched after exact: {len(pending):,}")

    if pending.empty:
        print("Nothing to do — exact match cleared the queue.")
        # Still write an empty review_candidates partition so downstream is idempotent.
        empty_cols = pd.DataFrame(
            columns=[
                "DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_NORMALIZED",
                "SECP_NAME_MAPPED", "matched_master_raw", "match_method", "match_score",
                "confidence", "ambiguous_master_candidates", "cluster_id", "cluster_label",
                "cluster_size", "suggested_mapping", "suggestion_score",
                "suggestion_confidence", "alternative_suggestions", "Firm_ID",
                "source_month", "pipeline_run_id",
            ]
        )
        empty_cols["source_month"] = cfg.source_month
        overwrite_partition(
            spark, empty_cols, table=paths.t_review_candidates(cfg),
            partition_cols=["source_month"],
            create_if_missing_schema=f"{cfg.catalog}.{cfg.gold_schema}",
        )
        run.row_count_out = 0
        dbutils.notebook.exit("0")

    # Load full master + tracked banks
    master_pdf = spark.table(paths.t_master_list_current(cfg)).toPandas()
    if "Firm_ID" in master_pdf.columns and "Firm ID" not in master_pdf.columns:
        master_pdf = master_pdf.rename(columns={"Firm_ID": "Firm ID"})
    tracked_pdf = read_csv_safe(paths.bootstrap_tracked_banks_path(cfg))

    workdir = tempfile.mkdtemp(prefix="ucc_fuzzy_")
    ucc_path = os.path.join(workdir, "ucc.csv")
    master_path = os.path.join(workdir, "master_list.csv")
    universe_path = os.path.join(workdir, "tracked_banks.csv")

    # Pending only — matching stages restart from a "fresh" raw view.
    # Drop helper / pipeline columns the loader doesn't expect.
    write_cols = ["DEBT_STATE", "SECP_STATE", "SECP_NAME"]
    pending[write_cols].to_csv(ucc_path, index=False)
    master_pdf.to_csv(master_path, index=False)
    tracked_pdf.to_csv(universe_path, index=False)

    pl_cfg = M.make_pipeline_config(
        ucc_file=ucc_path,
        master_file=master_path,
        universe_file=universe_path,
        output_dir=workdir,
        embedding_model=cfg.model_dir,
        fuzzy_auto_threshold=cfg.fuzzy_auto_threshold,
        fuzzy_review_threshold=cfg.fuzzy_review_threshold,
        embed_auto_threshold=cfg.embed_auto_threshold,
        embed_high_threshold=cfg.embed_high_threshold,
        embed_medium_threshold=cfg.embed_medium_threshold,
        embed_top_k=cfg.embed_top_k,
        embed_batch_size=cfg.embed_batch_size,
        fuzzy_batch_size=cfg.fuzzy_batch_size,
        fuzzy_workers=cfg.fuzzy_workers,
        cpu_threads=cfg.cpu_threads,
        embedding_workers=cfg.embedding_workers,
        early_stop_threshold=cfg.early_stop_threshold,
        max_candidates_per_query=cfg.max_candidates_per_query,
    )

    ucc_df = M.load_ucc_raw(pl_cfg)
    master_df = M.load_master_mapping(pl_cfg)
    universe_df = M.load_tracked_banks(pl_cfg)
    norm_master = M.build_normalized_master(master_df)
    master_cache = M.build_master_embeddings_cache(pl_cfg, norm_master)

    after_raw = M.stage_exact_raw(ucc_df, norm_master)
    after_norm = M.stage_exact_normalized(after_raw, norm_master)
    after_fuzzy = M.stage_fuzzy(pl_cfg, after_norm, norm_master)
    after_embed = M.stage_embedding(pl_cfg, after_fuzzy, norm_master, master_cache)
    after_cluster = M.stage_clustered(pl_cfg, after_embed, master_cache)
    after_merge = M.stage_merged_clusters(pl_cfg, after_cluster, master_cache)
    after_suggest = M.stage_suggested(pl_cfg, after_merge, norm_master, master_cache)
    enriched = M.stage_universe_enriched(after_suggest, universe_df)

    enriched["source_month"] = cfg.source_month
    enriched["pipeline_run_id"] = cfg.pipeline_run_id

    # Drop attrs and any unhashable extras before Spark conversion
    enriched.attrs.clear()

    n = overwrite_partition(
        spark,
        enriched,
        table=paths.t_review_candidates(cfg),
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.gold_schema}",
    )
    run.row_count_out = n
    print(f"Wrote {n:,} candidate rows to {paths.t_review_candidates(cfg)}")

# COMMAND ----------

dbutils.notebook.exit(str(n))
