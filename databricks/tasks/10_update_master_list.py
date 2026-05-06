# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Update master_list_current with approved decisions
# MAGIC
# MAGIC Applies the validated review decisions to
# MAGIC `ucc.silver.master_list_current` and appends the change log to
# MAGIC `ucc.audit.master_list_changes`.
# MAGIC
# MAGIC Decision semantics:
# MAGIC - `APPROVE`: insert/update row in master_list_current with the
# MAGIC   suggested mapping
# MAGIC - `OVERRIDE`: replace the mapping with `decided_mapping`
# MAGIC - `NEW`: insert a brand-new firm
# MAGIC - `REJECT`: no master change (logged as REJECT in audit)
# MAGIC
# MAGIC The previous master_list_current row (if any) is also snapshotted
# MAGIC into `ucc.silver.master_list_history` for full lineage.

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
from databricks.shared.audit import (
    PipelineRunLogger, ensure_business_schemas, ensure_audit_tables
)
from databricks.shared.io import append_delta, overwrite_delta

cfg = load_config(dbutils)
ensure_business_schemas(spark, cfg)
ensure_audit_tables(spark, cfg)
if not cfg.review_run_id:
    raise ValueError("review_run_id widget/job param is required for task 10")

# COMMAND ----------

from datetime import datetime
import uuid
import pandas as pd

JOIN_KEYS = ["DEBT_STATE", "SECP_STATE", "SECP_NAME"]

with PipelineRunLogger(spark, cfg, task_name="10_update_master_list") as run:
    decisions_df = (
        spark.table(paths.t_review_decisions(cfg))
        .where(
            f"source_month = '{cfg.source_month}' AND review_run_id = '{cfg.review_run_id}'"
        )
        .toPandas()
    )
    run.row_count_in = len(decisions_df)
    print(f"Decisions to apply: {len(decisions_df):,}")

    if decisions_df.empty:
        print("No decisions to apply; exiting.")
        dbutils.notebook.exit("0")

    master_pdf = spark.table(paths.t_master_list_current(cfg)).toPandas()
    if "Firm_ID" not in master_pdf.columns and "Firm ID" in master_pdf.columns:
        master_pdf = master_pdf.rename(columns={"Firm ID": "Firm_ID"})

    # Build a key for joining
    for df in (master_pdf, decisions_df):
        for c in JOIN_KEYS:
            df[c] = df[c].astype("string").str.upper().str.strip()

    # Construct a (key -> existing master row) lookup
    master_indexed = master_pdf.set_index(JOIN_KEYS, drop=False)

    changes = []
    new_master_rows = []

    for _, row in decisions_df.iterrows():
        decision = (row.get("decision_norm") or "").upper()
        key = tuple(row[c] for c in JOIN_KEYS)
        existing = master_indexed.loc[[key]] if key in master_indexed.index else None
        existing_row = existing.iloc[0].to_dict() if existing is not None and len(existing) else {}

        old_mapping = existing_row.get("SECP_NAME_MAPPED")
        old_firm = existing_row.get("Firm_ID")

        if decision == "APPROVE":
            new_mapping = row.get("decided_mapping") or row.get("suggested_mapping") or row.get("SECP_NAME_MAPPED")
            new_firm = row.get("decided_firm_id") or row.get("Firm_ID") or old_firm
            change_type = "APPROVE"
        elif decision == "OVERRIDE":
            new_mapping = row.get("decided_mapping")
            new_firm = row.get("decided_firm_id") or old_firm
            change_type = "OVERRIDE"
        elif decision == "NEW":
            new_mapping = row.get("decided_mapping") or row.get("suggested_mapping")
            new_firm = row.get("decided_firm_id")
            change_type = "NEW"
        elif decision == "REJECT":
            changes.append({
                "change_id": str(uuid.uuid4()),
                "review_run_id": cfg.review_run_id,
                "source_month": cfg.source_month,
                "change_type": "REJECT",
                "DEBT_STATE": key[0], "SECP_STATE": key[1], "SECP_NAME": key[2],
                "SECP_NAME_MAPPED_old": old_mapping,
                "SECP_NAME_MAPPED_new": None,
                "Firm_ID_old": old_firm,
                "Firm_ID_new": None,
                "decided_by": row.get("reviewer"),
                "decided_at": datetime.utcnow(),
            })
            continue
        else:
            continue

        if not new_mapping:
            # Nothing actionable; log and move on.
            continue

        new_row = dict(existing_row)
        new_row.update({
            "DEBT_STATE": key[0],
            "SECP_STATE": key[1],
            "SECP_NAME": key[2],
            "SECP_NAME_MAPPED": new_mapping,
            "Firm_ID": new_firm,
            "effective_from": datetime.utcnow(),
            "loaded_via": f"review:{change_type}",
            "source_month": cfg.source_month,
        })
        new_master_rows.append(new_row)
        changes.append({
            "change_id": str(uuid.uuid4()),
            "review_run_id": cfg.review_run_id,
            "source_month": cfg.source_month,
            "change_type": change_type,
            "DEBT_STATE": key[0], "SECP_STATE": key[1], "SECP_NAME": key[2],
            "SECP_NAME_MAPPED_old": old_mapping,
            "SECP_NAME_MAPPED_new": new_mapping,
            "Firm_ID_old": old_firm,
            "Firm_ID_new": new_firm,
            "decided_by": row.get("reviewer"),
            "decided_at": datetime.utcnow(),
        })

    print(f"Master row updates: {len(new_master_rows):,} | Changes logged: {len(changes):,}")

    # Snapshot existing master into history before mutating
    if not master_pdf.empty:
        snap = master_pdf.copy()
        snap["snapshot_taken_at"] = datetime.utcnow()
        snap["snapshot_source_month"] = cfg.source_month
        snap["snapshot_review_run_id"] = cfg.review_run_id
        append_delta(spark, snap, table=paths.t_master_list_history(cfg))

    if new_master_rows:
        new_pdf = pd.DataFrame(new_master_rows)
        # Drop existing rows with the same key, then concat the new ones.
        idx_keys = list(zip(*(master_pdf[c] for c in JOIN_KEYS))) if not master_pdf.empty else []
        new_keys = set(zip(*(new_pdf[c] for c in JOIN_KEYS)))
        kept = master_pdf[~pd.Series(idx_keys, index=master_pdf.index).isin(new_keys)] \
            if idx_keys else master_pdf
        merged = pd.concat([kept, new_pdf], ignore_index=True, sort=False)
        overwrite_delta(spark, merged, table=paths.t_master_list_current(cfg))
        print(f"master_list_current now has {len(merged):,} rows")

    if changes:
        change_df = pd.DataFrame(changes)
        append_delta(spark, change_df, table=paths.t_master_list_changes(cfg))
        print(f"Logged {len(changes):,} master_list_changes rows")

    run.row_count_out = len(new_master_rows)

# COMMAND ----------

dbutils.notebook.exit(str(len(new_master_rows)))
