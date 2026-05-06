"""Databricks-side configuration helpers.

Avoids hard-coding any paths/secrets in task notebooks. All values come
from notebook widgets / job parameters with sensible defaults that match
the recommended UC Volume layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional


# -- defaults ----------------------------------------------------------------

DEFAULT_CATALOG = "ucc"
DEFAULT_BRONZE_SCHEMA = "bronze"
DEFAULT_SILVER_SCHEMA = "silver"
DEFAULT_GOLD_SCHEMA = "gold"
DEFAULT_AUDIT_SCHEMA = "audit"

DEFAULT_LANDING_ROOT = "/Volumes/ucc/bronze/landing"
DEFAULT_GOLD_EXPORT_ROOT = "/Volumes/ucc/gold/exports"
DEFAULT_AUDIT_ARTIFACTS_ROOT = "/Volumes/ucc/audit/artifacts"

DEFAULT_MODEL_DIR = f"{DEFAULT_AUDIT_ARTIFACTS_ROOT}/models/bge-small-en-v1.5"
DEFAULT_HF_MODEL_ID = "BAAI/bge-small-en-v1.5"


# -- dataclass ---------------------------------------------------------------

@dataclass
class TaskConfig:
    """Runtime config assembled from widgets/job params."""

    source_month: str  # YYYY-MM, drives partition and many paths
    catalog: str = DEFAULT_CATALOG
    bronze_schema: str = DEFAULT_BRONZE_SCHEMA
    silver_schema: str = DEFAULT_SILVER_SCHEMA
    gold_schema: str = DEFAULT_GOLD_SCHEMA
    audit_schema: str = DEFAULT_AUDIT_SCHEMA

    landing_root: str = DEFAULT_LANDING_ROOT
    gold_export_root: str = DEFAULT_GOLD_EXPORT_ROOT
    model_dir: str = DEFAULT_MODEL_DIR
    hf_model_id: str = DEFAULT_HF_MODEL_ID

    review_run_id: Optional[str] = None  # set when ingesting completed_review.xlsx

    # tunables (mirror PipelineConfig defaults from the Dagster project)
    fuzzy_auto_threshold: int = 96
    fuzzy_review_threshold: int = 85
    embed_auto_threshold: float = 0.95
    embed_high_threshold: float = 0.90
    embed_medium_threshold: float = 0.85
    embed_top_k: int = 20
    embed_batch_size: int = 256
    fuzzy_batch_size: int = 1024
    fuzzy_workers: int = -1
    cpu_threads: int = -1
    embedding_workers: int = 4
    early_stop_threshold: float = 0.98
    max_candidates_per_query: int = 50

    # generated bookkeeping
    pipeline_run_id: str = field(
        default_factory=lambda: datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    )

    # ----- helpers ---------------------------------------------------------

    def table(self, schema: str, name: str) -> str:
        return f"{self.catalog}.{schema}.{name}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -- widget plumbing ---------------------------------------------------------

def _safe_get_widget(dbutils, name: str, default: str) -> str:
    """Read a widget; if not declared, declare it with the default and return."""
    try:
        return dbutils.widgets.get(name)
    except Exception:
        try:
            dbutils.widgets.text(name, default, name)
        except Exception:
            pass
        return default


WIDGET_DEFS = [
    ("source_month", ""),  # required at runtime
    ("catalog", DEFAULT_CATALOG),
    ("bronze_schema", DEFAULT_BRONZE_SCHEMA),
    ("silver_schema", DEFAULT_SILVER_SCHEMA),
    ("gold_schema", DEFAULT_GOLD_SCHEMA),
    ("audit_schema", DEFAULT_AUDIT_SCHEMA),
    ("landing_root", DEFAULT_LANDING_ROOT),
    ("gold_export_root", DEFAULT_GOLD_EXPORT_ROOT),
    ("model_dir", DEFAULT_MODEL_DIR),
    ("hf_model_id", DEFAULT_HF_MODEL_ID),
    ("review_run_id", ""),
]


def declare_widgets(dbutils) -> None:
    """Idempotently declare all widgets used by the pipeline."""
    for name, default in WIDGET_DEFS:
        try:
            dbutils.widgets.text(name, default, name)
        except Exception:
            pass


def load_config(dbutils) -> TaskConfig:
    """Build a :class:`TaskConfig` from declared widgets."""
    declare_widgets(dbutils)
    raw = {name: _safe_get_widget(dbutils, name, default) for name, default in WIDGET_DEFS}

    source_month = raw["source_month"].strip()
    if not source_month:
        raise ValueError(
            "source_month widget/job param is required (format YYYY-MM, e.g. 2026-04)"
        )

    review_run_id = raw["review_run_id"].strip() or None

    return TaskConfig(
        source_month=source_month,
        catalog=raw["catalog"] or DEFAULT_CATALOG,
        bronze_schema=raw["bronze_schema"] or DEFAULT_BRONZE_SCHEMA,
        silver_schema=raw["silver_schema"] or DEFAULT_SILVER_SCHEMA,
        gold_schema=raw["gold_schema"] or DEFAULT_GOLD_SCHEMA,
        audit_schema=raw["audit_schema"] or DEFAULT_AUDIT_SCHEMA,
        landing_root=raw["landing_root"] or DEFAULT_LANDING_ROOT,
        gold_export_root=raw["gold_export_root"] or DEFAULT_GOLD_EXPORT_ROOT,
        model_dir=raw["model_dir"] or DEFAULT_MODEL_DIR,
        hf_model_id=raw["hf_model_id"] or DEFAULT_HF_MODEL_ID,
        review_run_id=review_run_id,
    )
