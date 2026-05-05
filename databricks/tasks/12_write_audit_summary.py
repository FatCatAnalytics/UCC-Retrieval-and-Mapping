# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Write monthly audit summary → `ucc.gold.monthly_summary`
# MAGIC
# MAGIC One row per `source_month` (overwrite-partitioned). Aggregates the
# MAGIC counts a stakeholder might want at a glance:
# MAGIC totals, auto-mapped, review needed, new entities, master-list
# MAGIC changes applied, Firm_ID coverage.

# COMMAND ----------

# MAGIC %pip install --quiet pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from databricks.shared.config import load_config
from databricks.shared import paths
from databricks.shared.audit import PipelineRunLogger, ensure_business_schemas
from databricks.shared.io import overwrite_partition

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

from datetime import datetime
import pandas as pd

with PipelineRunLogger(spark, cfg, task_name="12_write_audit_summary") as run:
    sm = cfg.source_month

    def safe_count(table, where=None):
        if not spark.catalog.tableExists(table):
            return 0
        df = spark.table(table)
        if where:
            df = df.where(where)
        return int(df.count())

    where_sm = f"source_month = '{sm}'"

    counts = {
        "source_month": sm,
        "pipeline_run_id": cfg.pipeline_run_id,
        "review_run_id": cfg.review_run_id,
        "ucc_raw_rows": safe_count(paths.t_ucc_raw(cfg), where_sm),
        "ucc_clean_rows": safe_count(paths.t_ucc_clean(cfg), where_sm),
        "exact_mapped_rows": safe_count(paths.t_exact_mapped(cfg), where_sm),
        "review_candidate_rows": safe_count(paths.t_review_candidates(cfg), where_sm),
        "review_decision_rows": (
            safe_count(
                paths.t_review_decisions(cfg),
                f"{where_sm} AND review_run_id = '{cfg.review_run_id}'",
            )
            if cfg.review_run_id else 0
        ),
        "review_decision_error_rows": (
            safe_count(
                paths.t_review_decision_errors(cfg),
                f"{where_sm} AND review_run_id = '{cfg.review_run_id}'",
            )
            if cfg.review_run_id else 0
        ),
        "final_mapping_rows": safe_count(paths.t_final_ucc_mapping(cfg), where_sm),
        "master_list_changes_rows": (
            safe_count(
                paths.t_master_list_changes(cfg),
                f"review_run_id = '{cfg.review_run_id}'",
            )
            if cfg.review_run_id else 0
        ),
        "summary_written_at": datetime.utcnow(),
    }

    # Confidence / Firm_ID breakdown (best-effort)
    if spark.catalog.tableExists(paths.t_final_ucc_mapping(cfg)):
        final_df = spark.table(paths.t_final_ucc_mapping(cfg)).where(where_sm).toPandas()
        if not final_df.empty:
            counts["final_with_firm_id"] = int(final_df["Firm_ID"].notna().sum())
            counts["final_pct_with_firm_id"] = round(
                100 * counts["final_with_firm_id"] / max(1, len(final_df)), 2
            )
            for conf in ("AUTO", "REVIEWED", "REVIEW_HIGH", "REVIEW_MEDIUM", "NEW_ENTITY"):
                counts[f"final_conf_{conf}"] = int((final_df["confidence"] == conf).sum())

    summary_pdf = pd.DataFrame([counts])
    n = overwrite_partition(
        spark,
        summary_pdf,
        table=paths.t_monthly_summary(cfg),
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.gold_schema}",
    )
    run.row_count_in = sum(
        v for k, v in counts.items()
        if k.endswith("_rows") and isinstance(v, int)
    )
    run.row_count_out = n

    print("=== Monthly summary ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")

# COMMAND ----------

dbutils.notebook.exit("OK")
