"""
Re-apply tracked-bank enrichment (Firm_ID) to an existing full_results CSV
without re-running the full pipeline.

Usage:
    python -m ucc_mapping_project.scripts.reenrich_universe
    # or, to target a specific file:
    python -m ucc_mapping_project.scripts.reenrich_universe --input path/to/full_results.csv
"""
import argparse
import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent          # ucc_mapping_project/
TRACKED_BANKS_FILE = _HERE / "Inputs" / "tracked_banks.csv"
OUTPUT_DIR    = _HERE / "Outputs"


# ── helpers ──────────────────────────────────────────────────────────────────
def _majority_firm(sub: pd.DataFrame) -> pd.Series:
    """Return the most common Firm_ID for a tracked bank name."""
    if sub["Firm_ID"].notna().any():
        firm = sub["Firm_ID"].dropna().mode().iloc[0]
    else:
        firm = pd.NA
    return pd.Series({"Firm_ID": firm})


def build_lookups(tracked_banks: pd.DataFrame):
    """Build a case-insensitive Firm_ID lookup from tracked banks."""
    tracked_banks = tracked_banks.copy()
    tracked_banks["_key"] = tracked_banks["SECP_NAME"].str.upper().str.strip()

    banks_by_name = (
        tracked_banks[tracked_banks["_key"].notna()]
        .groupby("_key", as_index=False)
        .apply(_majority_firm, include_groups=False)
    )
    return banks_by_name.set_index("_key")["Firm_ID"]


def enrich(df: pd.DataFrame, secp_to_firm_ci) -> pd.DataFrame:
    """Add / overwrite Firm_ID from tracked banks."""
    df = df.copy()

    # Drop stale columns so we start fresh
    df = df.drop(columns=["Firm_ID", "GU_COMPANY_NAME_universe"], errors="ignore")
    df["Firm_ID"] = pd.NA

    # Pass 1 – SECP_NAME_MAPPED is authoritative
    if "SECP_NAME_MAPPED" in df.columns:
        mapped_upper = df["SECP_NAME_MAPPED"].str.upper().str.strip()

        df["Firm_ID"] = mapped_upper.map(secp_to_firm_ci)

        p1 = int(df["Firm_ID"].notna().sum())
        print(f"  Pass 1 (SECP_NAME_MAPPED): {p1:,} Firm_IDs resolved")

    # Pass 2 – raw SECP_NAME fallback for rows with no mapped name
    secp_upper = df["SECP_NAME"].str.upper().str.strip()

    fill_firm = df["Firm_ID"].isna() & secp_upper.map(secp_to_firm_ci).notna()

    df.loc[fill_firm, "Firm_ID"] = secp_upper.map(secp_to_firm_ci)[fill_firm]

    p2 = int(fill_firm.sum())
    print(f"  Pass 2 (raw SECP_NAME fallback): {p2:,} additional Firm_IDs")

    total = int(df["Firm_ID"].notna().sum())
    pct   = total / len(df) * 100 if len(df) else 0
    print(f"  Total: {total:,} / {len(df):,} rows have Firm_ID ({pct:.1f}%)")

    return df


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Re-enrich existing output CSV with tracked-bank Firm_ID.")
    parser.add_argument("--input",  default=None, help="Path to full_results CSV (defaults to most recent)")
    parser.add_argument("--output", default=None, help="Output path (defaults to Outputs/full_results_enriched_<ts>.csv)")
    args = parser.parse_args()

    # ── find input ───────────────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(
            glob.glob(str(OUTPUT_DIR / "full_results_*.csv")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No full_results_*.csv found in {OUTPUT_DIR}")
        input_path = Path(candidates[0])

    print(f"Input : {input_path}")

    # ── load data ────────────────────────────────────────────────────────────
    print("\nLoading tracked banks...")
    tracked_banks = pd.read_csv(
        TRACKED_BANKS_FILE,
        sep="\t",
        encoding="latin-1",
        usecols=["Bank Name", "Firm ID"],
        dtype={"Firm ID": "Int64"},
    ).rename(columns={"Bank Name": "SECP_NAME", "Firm ID": "Firm_ID"})
    print(f"  {len(tracked_banks):,} tracked bank rows loaded")

    print("Loading results CSV...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df):,} result rows loaded")

    # ── build lookups & enrich ───────────────────────────────────────────────
    print("\nBuilding lookups...")
    secp_to_firm_ci = build_lookups(tracked_banks)

    print("Enriching...")
    df = enrich(df, secp_to_firm_ci)

    # ── write output ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUTPUT_DIR / f"full_results_enriched_{ts}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
