"""IO helpers — Volume <-> local copies, CSV reads with encoding fallback,
Delta upserts keyed by ``source_month`` / ``review_run_id``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# CSV reads with encoding/delimiter fallbacks
# ---------------------------------------------------------------------------

def detect_delimiter(file_path: str, encoding: str = "latin-1") -> str:
    with open(file_path, "r", encoding=encoding) as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def read_csv_safe(
    file_path: str,
    *,
    usecols: Optional[Sequence[str]] = None,
    dtype: Optional[dict] = None,
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Read a CSV, sniffing delimiter and falling back UTF-8 -> latin-1."""
    encodings = [encoding] if encoding else []
    for enc in encodings + ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(
                file_path,
                sep=detect_delimiter(file_path, encoding=enc),
                encoding=enc,
                usecols=usecols,
                dtype=dtype,
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "ascii", b"", 0, 1, f"Could not decode {file_path} with any of {encodings + ['utf-8','latin-1']}"
    )


# ---------------------------------------------------------------------------
# Volume <-> local file staging
# ---------------------------------------------------------------------------

def stage_volume_file_locally(volume_path: str, local_root: str = "/tmp/ucc") -> str:
    """Copy a /Volumes/... file to /tmp so libraries that don't speak the
    volume API directly (e.g. older sentence-transformers cache loads) can
    read it. Returns the local path."""
    Path(local_root).mkdir(parents=True, exist_ok=True)
    dest = Path(local_root) / Path(volume_path).name
    shutil.copyfile(volume_path, dest)
    return str(dest)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Delta upsert: idempotent overwrite of a partition keyed by source_month
# ---------------------------------------------------------------------------

def overwrite_partition(
    spark,
    pdf: pd.DataFrame,
    *,
    table: str,
    partition_cols: Iterable[str],
    create_if_missing_schema: Optional[str] = None,
) -> int:
    """Idempotently overwrite the partition described by the given values.

    Strategy:
    - If the table exists, ``MERGE`` would be more flexible; for Phase 1 we
      use ``replaceWhere`` style overwrite by writing the whole partition.
    - If the table doesn't exist, create it with ``saveAsTable``.

    Returns the number of rows written.
    """
    if pdf is None or len(pdf) == 0:
        return 0

    sdf = spark.createDataFrame(pdf)

    # Build replaceWhere expression e.g. source_month='2026-04'
    parts = []
    for col in partition_cols:
        if col not in pdf.columns:
            raise KeyError(f"Partition column '{col}' missing from dataframe")
        val = pdf[col].iloc[0]
        if pd.isna(val):
            parts.append(f"{col} IS NULL")
        else:
            v = str(val).replace("'", "''")
            parts.append(f"{col} = '{v}'")
    where_expr = " AND ".join(parts) if parts else None

    writer = sdf.write.format("delta").mode("overwrite")
    if where_expr is not None:
        writer = writer.option("replaceWhere", where_expr)
    if list(partition_cols):
        writer = writer.partitionBy(*partition_cols)

    if spark.catalog.tableExists(table):
        writer.saveAsTable(table)
    else:
        if create_if_missing_schema:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {create_if_missing_schema}")
        # First write — create the table.
        sdf.write.format("delta").mode("overwrite").partitionBy(
            *partition_cols
        ).saveAsTable(table)

    return len(pdf)


def append_delta(spark, pdf: pd.DataFrame, *, table: str) -> int:
    if pdf is None or len(pdf) == 0:
        return 0
    sdf = spark.createDataFrame(pdf)
    sdf.write.format("delta").mode("append").saveAsTable(table)
    return len(pdf)


def overwrite_delta(spark, pdf: pd.DataFrame, *, table: str) -> int:
    if pdf is None or len(pdf) == 0:
        return 0
    sdf = spark.createDataFrame(pdf)
    sdf.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(table)
    return len(pdf)
