# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Validate review decisions
# MAGIC
# MAGIC Splits the bronze review workbook into:
# MAGIC - `ucc.silver.review_decisions` — clean, joinable decisions
# MAGIC - `ucc.silver.review_decision_errors` — rows that failed validation
# MAGIC
# MAGIC Validation rules (Phase 1, intentionally lenient):
# MAGIC - join keys present (DEBT_STATE, SECP_STATE, SECP_NAME)
# MAGIC - if `decision` column exists → must be one of
# MAGIC   `APPROVE / OVERRIDE / REJECT / NEW` (case-insensitive)
# MAGIC - rows with non-empty `decided_mapping` and a valid decision are
# MAGIC   considered actionable

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
from databricks.shared.validation import validate_review_decisions_df, raise_on_failure

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)
if not cfg.review_run_id:
    raise ValueError("review_run_id widget/job param is required for task 09")

# COMMAND ----------

import pandas as pd

VALID_DECISIONS = {"APPROVE", "OVERRIDE", "REJECT", "NEW"}

with PipelineRunLogger(spark, cfg, task_name="09_validate_review_decisions") as run:
    raw = (
        spark.table(paths.t_review_workbook_raw(cfg))
        .where(
            f"source_month = '{cfg.source_month}' AND review_run_id = '{cfg.review_run_id}'"
        )
        .toPandas()
    )
    run.row_count_in = len(raw)
    print(f"Loaded {len(raw):,} review rows for run {cfg.review_run_id}")

    raise_on_failure(validate_review_decisions_df(raw), "review_workbook")

    df = raw.copy()
    if "Decision" in df.columns and "decision" not in df.columns:
        df = df.rename(columns={"Decision": "decision"})
    if "decision" not in df.columns:
        df["decision"] = None
    df["decision_norm"] = (
        df["decision"].astype("string").str.upper().str.strip().fillna("")
    )

    valid_mask = df["decision_norm"].isin(VALID_DECISIONS)
    decided_mapping_col = next(
        (c for c in ("decided_mapping", "Decided Mapping", "approved_mapping")
         if c in df.columns),
        None,
    )
    if decided_mapping_col and decided_mapping_col != "decided_mapping":
        df = df.rename(columns={decided_mapping_col: "decided_mapping"})
    if "decided_mapping" not in df.columns:
        df["decided_mapping"] = None

    if "decided_firm_id" not in df.columns:
        for alt in ("Decided Firm ID", "decided_firm_id", "Firm_ID_decided"):
            if alt in df.columns:
                df = df.rename(columns={alt: "decided_firm_id"})
                break
        if "decided_firm_id" not in df.columns:
            df["decided_firm_id"] = None

    decisions = df.loc[valid_mask].copy()
    errors = df.loc[~valid_mask].copy()
    errors["error_reason"] = errors["decision_norm"].where(
        errors["decision_norm"].eq(""), "INVALID_DECISION_VALUE"
    ).replace({"": "MISSING_DECISION"})

    decisions["source_month"] = cfg.source_month
    decisions["review_run_id"] = cfg.review_run_id
    decisions["pipeline_run_id"] = cfg.pipeline_run_id
    errors["source_month"] = cfg.source_month
    errors["review_run_id"] = cfg.review_run_id
    errors["pipeline_run_id"] = cfg.pipeline_run_id

    n_dec = overwrite_partition(
        spark, decisions, table=paths.t_review_decisions(cfg),
        partition_cols=["source_month", "review_run_id"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.silver_schema}",
    )
    n_err = overwrite_partition(
        spark, errors, table=paths.t_review_decision_errors(cfg),
        partition_cols=["source_month", "review_run_id"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.silver_schema}",
    )
    run.row_count_out = n_dec
    print(f"Decisions: {n_dec:,} | Errors: {n_err:,}")

# COMMAND ----------

dbutils.notebook.exit(f"decisions={n_dec};errors={n_err}")
