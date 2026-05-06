"""Light-weight schema/quality validators for the monthly pipeline.

These checks are intentionally narrow — they only catch the failures
that have actually broken the Dagster runs in the past (missing
columns, empty file, completely invalid encoding, all-NA SECP_NAME).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Sequence

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> "ValidationResult":
        self.ok = False
        self.errors.append(msg)
        return self

    def warn(self, msg: str) -> "ValidationResult":
        self.warnings.append(msg)
        return self


def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def validate_required_paths(paths: Sequence[str]) -> ValidationResult:
    res = ValidationResult(ok=True)
    for p in paths:
        if not _exists(p):
            res.fail(f"Required path missing: {p}")
    return res


def validate_ucc_raw_df(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(ok=True)
    required = {"DEBT_STATE", "SECP_STATE", "SECP_NAME"}
    missing = required - set(df.columns)
    if missing:
        return res.fail(f"UCC raw missing columns: {sorted(missing)}")
    if df.empty:
        return res.fail("UCC raw is empty")
    secp_filled = df["SECP_NAME"].notna() & (df["SECP_NAME"].astype(str).str.strip() != "")
    if not secp_filled.any():
        return res.fail("UCC raw has no rows with non-empty SECP_NAME")
    if (~secp_filled).sum() > 0:
        res.warn(f"{(~secp_filled).sum():,} rows have empty SECP_NAME and will be dropped")
    return res


def validate_master_df(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(ok=True)
    required = {"DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED"}
    missing = required - set(df.columns)
    if missing:
        return res.fail(f"Master list missing columns: {sorted(missing)}")
    if df.empty:
        return res.fail("Master list is empty")
    if "Firm ID" not in df.columns and "Firm_ID" not in df.columns:
        res.warn("Master list has no Firm ID / Firm_ID column; downstream Firm_ID joins will be empty")
    return res


def validate_tracked_banks_df(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(ok=True)
    if df.empty:
        return res.warn("Tracked banks file is empty — Firm_ID enrichment will be skipped")
    expected = {"Bank Name", "Firm ID"}
    if expected - set(df.columns):
        # Allow already-renamed shape too
        renamed = {"SECP_NAME", "Firm_ID"}
        if renamed - set(df.columns):
            return res.fail(
                f"Tracked banks file is missing one of {expected} or {renamed}"
            )
    return res


def validate_review_decisions_df(df: pd.DataFrame) -> ValidationResult:
    """Loose validator — review workbook returns can have one of several
    shapes depending on the operator. We require at least the join key
    and a decision column."""
    res = ValidationResult(ok=True)
    join_keys = {"DEBT_STATE", "SECP_STATE", "SECP_NAME"}
    if join_keys - set(df.columns):
        return res.fail(
            f"Review workbook missing join key columns: {sorted(join_keys - set(df.columns))}"
        )
    decision_cols = {"decision", "Decision", "approved", "Approved"}
    if not decision_cols & set(df.columns):
        res.warn(
            "No decision column found (expected one of "
            f"{sorted(decision_cols)}); rows will be marked PENDING"
        )
    return res


def raise_on_failure(result: ValidationResult, label: str) -> None:
    if not result.ok:
        msg = f"[{label}] validation failed:\n  - " + "\n  - ".join(result.errors)
        raise ValueError(msg)
    if result.warnings:
        for w in result.warnings:
            print(f"[{label}] WARNING: {w}")
