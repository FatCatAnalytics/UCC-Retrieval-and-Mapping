# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Export review workbook (.xlsx)
# MAGIC
# MAGIC Reads `ucc.gold.review_candidates` for the current month, filters
# MAGIC to rows the business needs to look at (REVIEW_HIGH / REVIEW_MEDIUM
# MAGIC / NEW_ENTITY with a SUGGEST_*), and writes:
# MAGIC
# MAGIC `/Volumes/ucc/gold/exports/{source_month}/review_out/review_workbook_{source_month}.xlsx`
# MAGIC
# MAGIC The workbook has three sheets:
# MAGIC - **Fuzzy/Embedding Review** — needs accept/override
# MAGIC - **New Entity Suggestions** — clustered new firms with suggested mappings
# MAGIC - **Auto-Mapped (FYI)** — read-only sample for spot-checking

# COMMAND ----------

# MAGIC %pip install --quiet pandas openpyxl

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
from databricks.shared.io import ensure_dir

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

REVIEW_COLS = [
    "DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_NORMALIZED",
    "SECP_NAME_MAPPED", "matched_master_raw", "match_method", "match_score",
    "confidence", "ambiguous_master_candidates",
    "suggested_mapping", "suggestion_score", "suggestion_confidence",
    "alternative_suggestions", "cluster_id", "cluster_label", "cluster_size",
    "Firm_ID",
]
DECISION_COLS = ["decision", "decided_mapping", "decided_firm_id", "reviewer", "notes"]

with PipelineRunLogger(spark, cfg, task_name="06_export_review_workbook") as run:
    pdf = (
        spark.table(paths.t_review_candidates(cfg))
        .where(f"source_month = '{cfg.source_month}'")
        .toPandas()
    )
    run.row_count_in = len(pdf)
    print(f"Total candidates loaded: {len(pdf):,}")

    out = pdf.copy()
    for col in REVIEW_COLS:
        if col not in out.columns:
            out[col] = None

    review_mask = out["confidence"].isin(["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW"])
    new_mask = out["confidence"] == "NEW_ENTITY"
    auto_mask = out["confidence"] == "AUTO"

    review_df = out.loc[review_mask, REVIEW_COLS].copy()
    new_df = out.loc[new_mask, REVIEW_COLS].copy()
    auto_df = out.loc[auto_mask, REVIEW_COLS].head(500).copy()  # sample for FYI

    for df in (review_df, new_df):
        for c in DECISION_COLS:
            df[c] = ""

    review_df = review_df.sort_values(
        ["confidence", "match_score"], ascending=[True, False]
    )
    new_df = new_df.sort_values(
        ["suggestion_confidence", "cluster_size", "cluster_id", "SECP_NAME"],
        ascending=[True, False, True, True],
        na_position="last",
    )

    out_dir = paths.review_export_dir(cfg)
    ensure_dir(out_dir)
    workbook_path = paths.review_export_xlsx(cfg)
    print(f"Writing workbook to {workbook_path}")

    wb = Workbook()
    wb.remove(wb.active)
    for sheet_name, df in [
        ("Fuzzy_Embedding_Review", review_df),
        ("New_Entity_Suggestions", new_df),
        ("Auto_Mapped_FYI", auto_df),
    ]:
        ws = wb.create_sheet(sheet_name)
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)
    wb.save(workbook_path)

    run.row_count_out = len(review_df) + len(new_df)
    print(
        f"Review needed: {len(review_df):,} | "
        f"New entities: {len(new_df):,} | "
        f"Auto sample: {len(auto_df):,}"
    )

# COMMAND ----------

dbutils.notebook.exit(workbook_path)
