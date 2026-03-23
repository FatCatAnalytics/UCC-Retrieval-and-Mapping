"""
flag_firm_id_matches.py
-----------------------
Adds a `IN_FIRM_ID` flag column to master_mapping.csv.

For every row in master_mapping.csv, the script checks whether
SECP_NAME_MAPPED is present in the firm_id.csv "Banks Tracked" list.

Matching strategy (applied in order):
  1. Exact match (after light normalisation)
  2. Fuzzy match via RapidFuzz token_sort_ratio  ≥ FUZZY_THRESHOLD (default 88)

Output columns added to master_mapping.csv:
  IN_FIRM_ID          – bool  (True / False)
  FIRM_ID_MATCH_TYPE  – "exact" | "fuzzy" | ""
  FIRM_ID_BEST_MATCH  – the matched Banks Tracked name (empty if no match)
  FIRM_ID_SCORE       – fuzzy score 0-100 (100 for exact, 0 if no match)

The enriched file is written to:
  ucc_mappings/master_mapping_flagged.xlsx
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]          # …/ucc_mapping_project
MAPPINGS_DIR = BASE_DIR / "ucc_mappings"

FIRM_ID_PATH    = MAPPINGS_DIR / "firm_id.csv"
MASTER_PATH     = MAPPINGS_DIR / "master_mapping.csv"
OUTPUT_PATH     = MAPPINGS_DIR / "master_mapping_flagged.xlsx"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 92   # token_sort_ratio score (0-100) required for a match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_STRIP_LEGAL = re.compile(
    r"\b(n\.?a\.?|inc\.?|corp\.?|co\.?|llc\.?|ltd\.?|plc\.?|na|the)\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_PUNCT      = re.compile(r"[^a-z0-9 ]")


def normalise(text: str) -> str:
    """Lower-case, strip punctuation and common legal suffixes."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _PUNCT.sub(" ", t)
    t = _STRIP_LEGAL.sub(" ", t)
    t = _WHITESPACE.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # --- load data ----------------------------------------------------------
    firm_df   = pd.read_csv(FIRM_ID_PATH)
    master_df = pd.read_csv(MASTER_PATH, usecols=["SECP_NAME", "SECP_NAME_MAPPED"], on_bad_lines="skip")

    firm_col = firm_df.columns[0]          # "Banks Tracked"
    firm_names: list[str] = firm_df[firm_col].dropna().tolist()

    # Normalised versions for fast lookup
    firm_names_norm: list[str] = [normalise(n) for n in firm_names]
    norm_to_original: dict[str, str] = {
        normalise(n): n for n in firm_names
    }
    exact_set: set[str] = set(firm_names_norm)

    # --- check separator ----------------------------------------------------
    # master_mapping.csv uses tab separation
    mapped_col = "SECP_NAME_MAPPED"
    if mapped_col not in master_df.columns:
        raise ValueError(
            f"Expected column '{mapped_col}' not found. "
            f"Available columns: {master_df.columns.tolist()}"
        )

    mapped_names: list[str] = master_df[mapped_col].fillna("").tolist()

    # --- matching -----------------------------------------------------------
    in_firm_id:    list[bool]  = []
    match_types:   list[str]   = []
    best_matches:  list[str]   = []
    scores:        list[float] = []

    print(f"Matching {len(mapped_names):,} master rows against "
          f"{len(firm_names):,} firm_id entries …")

    for raw_name in mapped_names:
        norm = normalise(raw_name)

        # 1. Exact match
        if norm in exact_set:
            in_firm_id.append(True)
            match_types.append("exact")
            best_matches.append(norm_to_original[norm])
            scores.append(100.0)
            continue

        # 2. Fuzzy match (token_sort_ratio is robust to word-order differences)
        if norm:
            result = process.extractOne(
                norm,
                firm_names_norm,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
        else:
            result = None

        if result is not None:
            matched_norm, score, _ = result
            in_firm_id.append(True)
            match_types.append("fuzzy")
            best_matches.append(norm_to_original.get(matched_norm, matched_norm))
            scores.append(round(score, 1))
        else:
            in_firm_id.append(False)
            match_types.append("")
            best_matches.append("")
            scores.append(0.0)

    # --- attach columns and save -------------------------------------------
    master_df["IN_FIRM_ID"]         = in_firm_id
    master_df["FIRM_ID_MATCH_TYPE"] = match_types
    master_df["FIRM_ID_BEST_MATCH"] = best_matches
    master_df["FIRM_ID_SCORE"]      = scores

    master_df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

    # --- summary ------------------------------------------------------------
    total   = len(master_df)
    matched = int(master_df["IN_FIRM_ID"].sum())
    exact   = (master_df["FIRM_ID_MATCH_TYPE"] == "exact").sum()
    fuzzy   = (master_df["FIRM_ID_MATCH_TYPE"] == "fuzzy").sum()

    print(f"\n{'='*55}")
    print(f"  Total rows        : {total:>8,}")
    print(f"  Matched (IN_FIRM_ID=True) : {matched:>8,}")
    print(f"    – exact         : {exact:>8,}")
    print(f"    – fuzzy         : {fuzzy:>8,}")
    print(f"  Not matched       : {total - matched:>8,}")
    print(f"{'='*55}")
    print(f"\nOutput written to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
