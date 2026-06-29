"""I/O helpers for the Databricks port.

Stage outputs are passed between Databricks Job tasks by persisting them to a
Unity Catalog **Volume** as pickle files. Pickle is used (instead of round-
tripping pandas <-> Spark) because several intermediates are not tabular:
``normalized_master_data`` and ``master_embeddings_cache`` are Python dicts
holding numpy arrays / FAISS-friendly structures, and some result DataFrames
carry list-valued columns (``alternative_suggestions``,
``ambiguous_master_candidates``). Pickle keeps every dtype intact and lossless.

Curated **result** DataFrames are *additionally* published as Unity Catalog
Delta tables (best-effort) so downstream consumers get first-class UC tables.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Where intermediate stage outputs live (a UC Volume path on Databricks).
STAGE_DIR = os.environ.get(
    "UCC_STAGE_DIR",
    f"{os.environ.get('UCC_DATA_ROOT', '/Volumes/main/ucc/data')}/_staging",
)

# Unity Catalog target for published result tables.
UC_CATALOG = os.environ.get("UCC_CATALOG", "main")
UC_SCHEMA = os.environ.get("UCC_SCHEMA", "ucc")


def stage_file(name: str) -> str:
    return str(Path(STAGE_DIR) / f"{name}.pkl")


def save_stage(obj: Any, name: str) -> str:
    """Persist a stage output (DataFrame or dict) to the staging volume."""
    Path(STAGE_DIR).mkdir(parents=True, exist_ok=True)
    path = stage_file(name)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_stage(name: str) -> Any:
    """Load a previously persisted stage output."""
    path = stage_file(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage output '{name}' not found at {path}. "
            f"Run its upstream task first (or use run_pipeline for an in-process run)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def get_spark():
    """Return the active SparkSession (on Databricks one already exists)."""
    try:
        from pyspark.sql import SparkSession
    except Exception as exc:  # pragma: no cover - only on non-Databricks envs
        raise RuntimeError("pyspark is unavailable; Delta export requires Databricks") from exc
    return SparkSession.builder.getOrCreate()


def get_dbutils():
    """Best-effort access to dbutils (for secrets / widgets) when on Databricks."""
    try:
        from pyspark.dbutils import DBUtils  # type: ignore

        return DBUtils(get_spark())
    except Exception:
        try:
            import IPython  # type: ignore

            return IPython.get_ipython().user_ns.get("dbutils")
        except Exception:
            return None


def _sanitize_for_spark(pdf: pd.DataFrame) -> pd.DataFrame:
    """Make a pandas DataFrame safe to hand to Spark.

    List/dict-valued cells are serialized to JSON strings; pandas nullable
    Int64 is downcast to float (NaN-safe) so Spark schema inference succeeds.
    """
    import json

    out = pdf.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (list, dict, tuple, set)) else v
            )
        elif str(out[col].dtype) == "Int64":
            out[col] = out[col].astype("float64")
    return out


def write_delta(pdf: pd.DataFrame, table: str, mode: str = "overwrite") -> Optional[str]:
    """Publish a pandas DataFrame as a UC Delta table ``catalog.schema.table``.

    Best-effort: logs and returns None if Spark/UC is unavailable.
    """
    fqn = f"{UC_CATALOG}.{UC_SCHEMA}.{table}"
    try:
        spark = get_spark()
        sdf = spark.createDataFrame(_sanitize_for_spark(pdf))
        (
            sdf.write.format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(fqn)
        )
        return fqn
    except Exception as exc:  # pragma: no cover
        import logging

        logging.getLogger("ucc_pipeline").warning(
            "Delta publish to %s skipped/failed: %s", fqn, exc
        )
        return None
