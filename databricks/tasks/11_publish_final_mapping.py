# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — Publish final UCC mapping → `ucc.gold.final_ucc_mapping`
# MAGIC
# MAGIC Combines the exact-mapped rows (from task 04) with the post-review
# MAGIC decisions (from task 09) and any AUTO/REVIEW_HIGH rows that the
# MAGIC reviewer left untouched, producing a single per-`source_month`
# MAGIC partition in `ucc.gold.final_ucc_mapping`.
# MAGIC
# MAGIC Schema (all string except Firm_ID Int64):
# MAGIC `DEBT_STATE, SECP_STATE, SECP_NAME, SECP_NAME_MAPPED,
# MAGIC  Firm_ID, match_method, confidence, source_month,
# MAGIC  review_run_id, pipeline_run_id`

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

import pandas as pd

FINAL_COLS = [
    "DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED",
    "Firm_ID", "match_method", "confidence",
    "source_month", "review_run_id", "pipeline_run_id",
]

with PipelineRunLogger(spark, cfg, task_name="11_publish_final_mapping") as run:
    exact = (
        spark.table(paths.t_exact_mapped(cfg))
        .where(f"source_month = '{cfg.source_month}'")
        .toPandas()
    )
    candidates = (
        spark.table(paths.t_review_candidates(cfg))
        .where(f"source_month = '{cfg.source_month}'")
        .toPandas()
    )

    decisions_df = pd.DataFrame()
    if cfg.review_run_id and spark.catalog.tableExists(paths.t_review_decisions(cfg)):
        decisions_df = (
            spark.table(paths.t_review_decisions(cfg))
            .where(
                f"source_month = '{cfg.source_month}' AND review_run_id = '{cfg.review_run_id}'"
            )
            .toPandas()
        )

    run.row_count_in = len(exact) + len(candidates) + len(decisions_df)
    print(f"exact={len(exact):,} candidates={len(candidates):,} decisions={len(decisions_df):,}")

    JOIN_KEYS = ["DEBT_STATE", "SECP_STATE", "SECP_NAME"]

    # Reviewer decisions take priority where available
    decisions_view = decisions_df.copy()
    if not decisions_view.empty:
        decided_mapping = decisions_view.get("decided_mapping")
        suggested_mapping = decisions_view.get("suggested_mapping") if "suggested_mapping" in decisions_view.columns else None
        existing_mapping = decisions_view.get("SECP_NAME_MAPPED") if "SECP_NAME_MAPPED" in decisions_view.columns else None

        decisions_view["SECP_NAME_MAPPED"] = (
            decided_mapping
            .where(decided_mapping.astype("string").str.strip().fillna("") != "")
            .fillna(suggested_mapping if suggested_mapping is not None else pd.Series([None] * len(decisions_view)))
            .fillna(existing_mapping if existing_mapping is not None else pd.Series([None] * len(decisions_view)))
        )
        decisions_view["match_method"] = "Review " + decisions_view.get("decision_norm", "DECIDED").astype("string")
        decisions_view["confidence"] = "REVIEWED"
        if "Firm_ID" not in decisions_view.columns:
            decisions_view["Firm_ID"] = decisions_view.get("decided_firm_id")
        keep = JOIN_KEYS + ["SECP_NAME_MAPPED", "Firm_ID", "match_method", "confidence"]
        decisions_view = decisions_view[[c for c in keep if c in decisions_view.columns]].drop_duplicates(JOIN_KEYS, keep="last")

    # Concatenate with priority decisions > exact > candidates AUTO
    parts = []
    if not decisions_view.empty:
        parts.append(decisions_view.assign(__priority__=0))
    if not exact.empty:
        parts.append(exact.assign(__priority__=1))
    auto_remainder = candidates[candidates["confidence"] == "AUTO"]
    if not auto_remainder.empty:
        parts.append(auto_remainder.assign(__priority__=2))

    if not parts:
        print("Nothing to publish; writing empty partition")
        empty = pd.DataFrame(columns=FINAL_COLS)
        empty["source_month"] = cfg.source_month
        overwrite_partition(
            spark, empty, table=paths.t_final_ucc_mapping(cfg),
            partition_cols=["source_month"],
            create_if_missing_schema=f"{cfg.catalog}.{cfg.gold_schema}",
        )
        run.row_count_out = 0
        dbutils.notebook.exit("0")

    combined = pd.concat(parts, ignore_index=True, sort=False)

    # Resolve duplicates by lowest priority value (decisions win)
    combined = combined.sort_values("__priority__").drop_duplicates(JOIN_KEYS, keep="first")
    combined = combined.drop(columns=["__priority__"])

    for col in FINAL_COLS:
        if col not in combined.columns:
            combined[col] = None
    combined["source_month"] = cfg.source_month
    combined["review_run_id"] = cfg.review_run_id
    combined["pipeline_run_id"] = cfg.pipeline_run_id

    final = combined[FINAL_COLS].copy()

    n = overwrite_partition(
        spark,
        final,
        table=paths.t_final_ucc_mapping(cfg),
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.gold_schema}",
    )
    run.row_count_out = n
    print(f"Wrote {n:,} rows to {paths.t_final_ucc_mapping(cfg)}")

# COMMAND ----------

dbutils.notebook.exit(str(n))
