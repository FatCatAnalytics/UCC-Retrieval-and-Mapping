# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Load master_list_current
# MAGIC
# MAGIC On the **first** run of the month, this task copies the bootstrap
# MAGIC `master_list.csv` into `ucc.silver.master_list_current` and into a
# MAGIC dated snapshot row in `ucc.silver.master_list_history`.
# MAGIC
# MAGIC On subsequent runs, the bootstrap CSV is a fallback only — the
# MAGIC table-of-record is `ucc.silver.master_list_current`, which is
# MAGIC updated by task `10_update_master_list.py` after each review cycle.
# MAGIC
# MAGIC **Job parameters**
# MAGIC - `force_bootstrap` (default `false`): set to `true` to overwrite
# MAGIC   `master_list_current` from the bootstrap CSV.

# COMMAND ----------

# MAGIC %pip install --quiet pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from databricks.shared.config import load_config, declare_widgets
from databricks.shared import paths
from databricks.shared.audit import PipelineRunLogger, ensure_business_schemas
from databricks.shared.io import read_csv_safe, overwrite_delta, append_delta

# Add an extra widget specific to this task
declare_widgets(dbutils)
try:
    dbutils.widgets.dropdown("force_bootstrap", "false", ["false", "true"], "force_bootstrap")
except Exception:
    pass
force_bootstrap = (dbutils.widgets.get("force_bootstrap") or "false").lower() == "true"

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

from datetime import datetime
import pandas as pd

current_table = paths.t_master_list_current(cfg)
history_table = paths.t_master_list_history(cfg)
bootstrap = paths.bootstrap_master_path(cfg)

with PipelineRunLogger(spark, cfg, task_name="03_load_master_current") as run:
    table_exists = spark.catalog.tableExists(current_table)
    if table_exists and not force_bootstrap:
        n = spark.table(current_table).count()
        print(f"{current_table} already exists with {n:,} rows — skipping bootstrap")
        run.row_count_in = n
        run.row_count_out = n
    else:
        print(f"Bootstrapping {current_table} from {bootstrap}")
        df = read_csv_safe(bootstrap)
        # Standardize Firm ID -> Firm_ID (Spark friendly)
        if "Firm ID" in df.columns and "Firm_ID" not in df.columns:
            df = df.rename(columns={"Firm ID": "Firm_ID"})
        df["effective_from"] = datetime.utcnow()
        df["loaded_via"] = "bootstrap"
        df["source_month"] = cfg.source_month

        run.row_count_in = len(df)
        n = overwrite_delta(spark, df, table=current_table)
        run.row_count_out = n
        print(f"Wrote {n:,} rows to {current_table}")

        # Append a history snapshot
        snap = df.copy()
        snap["snapshot_taken_at"] = datetime.utcnow()
        snap["snapshot_source_month"] = cfg.source_month
        append_delta(spark, snap, table=history_table)
        print(f"Appended snapshot to {history_table}")

# COMMAND ----------

dbutils.notebook.exit("OK")
