# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Validate inputs
# MAGIC
# MAGIC Pre-flight check before the monthly run. Confirms that:
# MAGIC - the bootstrap master / tracked-banks files exist on the volume,
# MAGIC - the source `ucc.csv` is present for `source_month`,
# MAGIC - the embedding model directory exists,
# MAGIC - basic schema checks pass.
# MAGIC
# MAGIC Fails fast so downstream tasks don't waste compute.

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
from databricks.shared.io import read_csv_safe
from databricks.shared.validation import (
    validate_required_paths,
    validate_ucc_raw_df,
    validate_master_df,
    validate_tracked_banks_df,
    raise_on_failure,
)

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)

# COMMAND ----------

required = [
    paths.bootstrap_master_path(cfg),
    paths.bootstrap_tracked_banks_path(cfg),
    paths.source_ucc_path(cfg),
    cfg.model_dir,
]

with PipelineRunLogger(spark, cfg, task_name="00_validate_inputs") as run:
    raise_on_failure(validate_required_paths(required), "paths")

    # quick schema / content checks (only first ~5k rows for speed)
    ucc_head = read_csv_safe(paths.source_ucc_path(cfg)).head(5000)
    raise_on_failure(validate_ucc_raw_df(ucc_head), "ucc.csv")

    master_head = read_csv_safe(paths.bootstrap_master_path(cfg)).head(5000)
    raise_on_failure(validate_master_df(master_head), "master_list.csv")

    tracked_head = read_csv_safe(paths.bootstrap_tracked_banks_path(cfg)).head(5000)
    raise_on_failure(validate_tracked_banks_df(tracked_head), "tracked_banks.csv")

    run.row_count_in = len(ucc_head)
    run.row_count_out = len(ucc_head)

print("Inputs OK")

# COMMAND ----------

dbutils.notebook.exit("OK")
