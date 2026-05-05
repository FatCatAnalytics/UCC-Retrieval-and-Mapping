# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest raw UCC CSV → `ucc.bronze.ucc_raw`
# MAGIC
# MAGIC Reads `/Volumes/ucc/bronze/landing/{source_month}/source/ucc.csv`,
# MAGIC tags each row with `source_month` and `pipeline_run_id`, then writes
# MAGIC to `ucc.bronze.ucc_raw` partitioned by `source_month` (idempotent
# MAGIC overwrite of that partition).
# MAGIC
# MAGIC Also records a row in `ucc.audit.source_files`.

# COMMAND ----------

# MAGIC %pip install --quiet pandas rapidfuzz

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from databricks.shared.config import load_config
from databricks.shared import paths
from databricks.shared.audit import (
    PipelineRunLogger,
    ensure_business_schemas,
    record_source_file,
    hash_file_locally,
)
from databricks.shared.io import read_csv_safe, overwrite_partition

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)
src = paths.source_ucc_path(cfg)
print(f"Reading {src}")

# COMMAND ----------

with PipelineRunLogger(spark, cfg, task_name="01_ingest_ucc_raw") as run:
    df = read_csv_safe(src)
    run.row_count_in = len(df)

    # Stamp lineage columns
    df["source_month"] = cfg.source_month
    df["pipeline_run_id"] = cfg.pipeline_run_id

    # Idempotent partition overwrite
    written = overwrite_partition(
        spark,
        df,
        table=paths.t_ucc_raw(cfg),
        partition_cols=["source_month"],
        create_if_missing_schema=f"{cfg.catalog}.{cfg.bronze_schema}",
    )
    run.row_count_out = written

    # Try to fingerprint — best-effort; skip if the volume mount won't
    # let us read the file as a local POSIX path.
    sha = None
    size = None
    try:
        sha = hash_file_locally(src)
        size = os.path.getsize(src)
    except Exception as e:
        print(f"Skipping sha/size fingerprint: {e}")

    record_source_file(
        spark,
        cfg,
        file_kind="ucc_raw",
        file_path=src,
        row_count=written,
        size_bytes=size,
        sha256=sha,
    )
    print(f"Wrote {written:,} rows to {paths.t_ucc_raw(cfg)}")

# COMMAND ----------

dbutils.notebook.exit(str(written))
