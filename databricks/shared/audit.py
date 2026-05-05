"""Audit logging helpers — pipeline runs, source files, master_list changes.

Audit tables are created on first use so a fresh workspace can run end-to-end
without separate DDL.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from . import paths
from .config import TaskConfig


# ---------------------------------------------------------------------------
# DDL — defensive create-if-not-exists for all audit tables
# ---------------------------------------------------------------------------

_AUDIT_DDL = {
    "pipeline_runs": """
        CREATE TABLE IF NOT EXISTS {table} (
            pipeline_run_id STRING,
            task_name STRING,
            source_month STRING,
            review_run_id STRING,
            status STRING,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            row_count_in BIGINT,
            row_count_out BIGINT,
            error_message STRING,
            params STRING
        ) USING DELTA
    """,
    "source_files": """
        CREATE TABLE IF NOT EXISTS {table} (
            pipeline_run_id STRING,
            source_month STRING,
            file_kind STRING,
            file_path STRING,
            size_bytes BIGINT,
            sha256 STRING,
            row_count BIGINT,
            ingested_at TIMESTAMP
        ) USING DELTA
    """,
    "review_workbooks": """
        CREATE TABLE IF NOT EXISTS {table} (
            review_run_id STRING,
            source_month STRING,
            workbook_path STRING,
            sha256 STRING,
            sheet_count INT,
            row_count BIGINT,
            status STRING,
            ingested_at TIMESTAMP
        ) USING DELTA
    """,
    "master_list_changes": """
        CREATE TABLE IF NOT EXISTS {table} (
            change_id STRING,
            review_run_id STRING,
            source_month STRING,
            change_type STRING,
            DEBT_STATE STRING,
            SECP_STATE STRING,
            SECP_NAME STRING,
            SECP_NAME_MAPPED_old STRING,
            SECP_NAME_MAPPED_new STRING,
            Firm_ID_old BIGINT,
            Firm_ID_new BIGINT,
            decided_by STRING,
            decided_at TIMESTAMP
        ) USING DELTA
    """,
}


def ensure_audit_tables(spark, cfg: TaskConfig) -> None:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.audit_schema}")
    spark.sql(_AUDIT_DDL["pipeline_runs"].format(table=paths.t_pipeline_runs(cfg)))
    spark.sql(_AUDIT_DDL["source_files"].format(table=paths.t_source_files(cfg)))
    spark.sql(_AUDIT_DDL["review_workbooks"].format(table=paths.t_review_workbooks(cfg)))
    spark.sql(
        _AUDIT_DDL["master_list_changes"].format(table=paths.t_master_list_changes(cfg))
    )


def ensure_business_schemas(spark, cfg: TaskConfig) -> None:
    for schema in (cfg.bronze_schema, cfg.silver_schema, cfg.gold_schema):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{schema}")


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------

class PipelineRunLogger:
    """Tiny helper that wraps a single task's audit row.

    Use:

        with PipelineRunLogger(spark, cfg, task_name="01_ingest_ucc_raw") as run:
            run.row_count_in = ...
            ...
            run.row_count_out = ...
    """

    def __init__(self, spark, cfg: TaskConfig, task_name: str) -> None:
        self.spark = spark
        self.cfg = cfg
        self.task_name = task_name
        self.row_count_in: Optional[int] = None
        self.row_count_out: Optional[int] = None
        self._started_at: Optional[datetime] = None
        self._error: Optional[str] = None

    def __enter__(self) -> "PipelineRunLogger":
        ensure_audit_tables(self.spark, self.cfg)
        self._started_at = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self._error = f"{exc_type.__name__}: {exc}"
        self._write("FAILED" if exc is not None else "OK")
        return False  # never swallow exceptions

    def _write(self, status: str) -> None:
        ended_at = datetime.utcnow()
        params_json = json.dumps(self.cfg.to_dict(), default=str)
        row: Dict[str, Any] = {
            "pipeline_run_id": self.cfg.pipeline_run_id,
            "task_name": self.task_name,
            "source_month": self.cfg.source_month,
            "review_run_id": self.cfg.review_run_id,
            "status": status,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "row_count_in": self.row_count_in,
            "row_count_out": self.row_count_out,
            "error_message": self._error,
            "params": params_json,
        }
        df = self.spark.createDataFrame([row])
        df.write.mode("append").saveAsTable(paths.t_pipeline_runs(self.cfg))


# ---------------------------------------------------------------------------
# Source-file fingerprinting
# ---------------------------------------------------------------------------

def record_source_file(
    spark,
    cfg: TaskConfig,
    *,
    file_kind: str,
    file_path: str,
    row_count: int,
    size_bytes: Optional[int] = None,
    sha256: Optional[str] = None,
) -> None:
    ensure_audit_tables(spark, cfg)
    row = {
        "pipeline_run_id": cfg.pipeline_run_id,
        "source_month": cfg.source_month,
        "file_kind": file_kind,
        "file_path": file_path,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "row_count": row_count,
        "ingested_at": datetime.utcnow(),
    }
    spark.createDataFrame([row]).write.mode("append").saveAsTable(
        paths.t_source_files(cfg)
    )


def hash_file_locally(local_path: str) -> str:
    """Stream-hash a file at ``local_path`` (works for /Volumes paths in
    serverless after the file is staged locally; otherwise omit and pass
    ``None`` to ``record_source_file``)."""
    import hashlib

    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
