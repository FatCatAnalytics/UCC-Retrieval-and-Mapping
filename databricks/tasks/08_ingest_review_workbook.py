# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Ingest completed review workbook → `ucc.bronze.review_workbook_raw`
# MAGIC
# MAGIC Reads the operator-completed Excel file from
# MAGIC `/Volumes/ucc/bronze/landing/{source_month}/review_return/completed_review.xlsx`,
# MAGIC tags it with a fresh `review_run_id`, and writes a verbatim copy
# MAGIC to bronze. Validation and master-list updates happen in tasks 09
# MAGIC and 10 respectively.
# MAGIC
# MAGIC **Job parameters**
# MAGIC - `review_filename` (default `completed_review.xlsx`)
# MAGIC
# MAGIC The new `review_run_id` is exited so a downstream Databricks Jobs
# MAGIC task can inject it as the `review_run_id` widget for tasks 09-12.

# COMMAND ----------

# MAGIC %pip install --quiet pandas openpyxl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from databricks.shared.config import load_config, declare_widgets
from databricks.shared import paths
from databricks.shared.audit import PipelineRunLogger, ensure_audit_tables, ensure_business_schemas
from databricks.shared.io import overwrite_partition

declare_widgets(dbutils)
try:
    dbutils.widgets.text("review_filename", "completed_review.xlsx", "review_filename")
except Exception:
    pass
review_filename = dbutils.widgets.get("review_filename") or "completed_review.xlsx"

cfg = load_config(dbutils)
ensure_audit_tables(spark, cfg)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

from datetime import datetime
import pandas as pd
import hashlib

review_path = paths.review_return_path(cfg, filename=review_filename)
print(f"Reading {review_path}")

with PipelineRunLogger(spark, cfg, task_name="08_ingest_review_workbook") as run:
    sheets = pd.read_excel(review_path, sheet_name=None, engine="openpyxl")
    print(f"Found sheets: {list(sheets.keys())}")

    # Stamp a fresh review_run_id (overrides any inbound widget)
    review_run_id = datetime.utcnow().strftime("review_%Y%m%dT%H%M%SZ")
    cfg.review_run_id = review_run_id

    # Hash of the file for audit
    h = hashlib.sha256()
    try:
        with open(review_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()
        size = os.path.getsize(review_path)
    except Exception as e:
        print(f"Hash skipped: {e}")
        sha, size = None, None

    # Concat all sheets (preserve sheet name) -> single bronze table
    frames = []
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["__sheet__"] = sheet_name
        frames.append(df)
    if not frames:
        raise ValueError("Review workbook has no non-empty sheets")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["review_run_id"] = review_run_id
    combined["source_month"] = cfg.source_month
    combined["pipeline_run_id"] = cfg.pipeline_run_id
    combined["ingested_at"] = datetime.utcnow()

    run.row_count_in = len(combined)
    n = overwrite_partition(
        spark,
        combined,
        table=paths.t_review_workbook_raw(cfg),
        partition_cols=["source_month", "review_run_id"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.bronze_schema}",
    )
    run.row_count_out = n

    # Audit row in review_workbooks
    audit_row = pd.DataFrame([{
        "review_run_id": review_run_id,
        "source_month": cfg.source_month,
        "workbook_path": review_path,
        "sha256": sha,
        "sheet_count": len(sheets),
        "row_count": n,
        "status": "INGESTED",
        "ingested_at": datetime.utcnow(),
    }])
    spark.createDataFrame(audit_row).write.mode("append").saveAsTable(
        paths.t_review_workbooks(cfg)
    )
    print(f"Wrote {n:,} rows to {paths.t_review_workbook_raw(cfg)}; review_run_id={review_run_id}")

# COMMAND ----------

dbutils.notebook.exit(review_run_id)
