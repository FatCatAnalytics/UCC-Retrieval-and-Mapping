"""
Re-apply our-universe enrichment (Firm_ID + GU_COMPANY_NAME_universe) to an
existing full_results CSV without re-running the full pipeline.

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
UNIVERSE_FILE = _HERE / "Inputs" / "our universe.csv"
OUTPUT_DIR    = _HERE / "Outputs"


# ── helpers ──────────────────────────────────────────────────────────────────
def _majority_firm(sub: pd.DataFrame) -> pd.Series:
    """Return the most common Firm_ID (and its GU_COMPANY_NAME) for a group."""
    if sub["Firm_ID"].notna().any():
        firm = sub["Firm_ID"].dropna().mode().iloc[0]
        gu_rows = sub.loc[sub["Firm_ID"] == firm, "GU_COMPANY_NAME"]
        gu = gu_rows.dropna().iloc[0] if gu_rows.notna().any() else pd.NA
    else:
        firm = pd.NA
        gu_rows = sub["GU_COMPANY_NAME"].dropna()
        gu = gu_rows.iloc[0] if not gu_rows.empty else pd.NA
    return pd.Series({"Firm_ID": firm, "GU_COMPANY_NAME": gu})


def build_lookups(universe: pd.DataFrame):
    """Build case-insensitive lookup dicts from the universe dataframe."""
    universe = universe.copy()
    universe["_key"] = universe["SECP_NAME"].str.upper().str.strip()

    univ_by_secp = (
        universe[universe["_key"].notna()]
        .groupby("_key", as_index=False)
        .apply(_majority_firm, include_groups=False)
    )
    secp_to_firm_ci = univ_by_secp.set_index("_key")["Firm_ID"]
    secp_to_gu_ci   = univ_by_secp.set_index("_key")["GU_COMPANY_NAME"]

    univ_gu = universe[
        universe["GU_COMPANY_NAME"].notna() & universe["Firm_ID"].notna()
    ].copy()
    univ_gu["_gu_key"] = univ_gu["GU_COMPANY_NAME"].str.upper().str.strip()
    gu_to_firm_ci = (
        univ_gu.groupby("_gu_key")["Firm_ID"]
        .agg(lambda x: x.mode().iloc[0])
    )

    return secp_to_firm_ci, secp_to_gu_ci, gu_to_firm_ci


def enrich(df: pd.DataFrame, secp_to_firm_ci, secp_to_gu_ci, gu_to_firm_ci) -> pd.DataFrame:
    """Add / overwrite Firm_ID and GU_COMPANY_NAME_universe columns."""
    df = df.copy()

    # Drop stale columns so we start fresh
    df = df.drop(columns=["Firm_ID", "GU_COMPANY_NAME_universe"], errors="ignore")
    df["Firm_ID"]                  = pd.NA
    df["GU_COMPANY_NAME_universe"] = pd.NA

    # Pass 1 – SECP_NAME_MAPPED is authoritative
    if "SECP_NAME_MAPPED" in df.columns:
        mapped_upper = df["SECP_NAME_MAPPED"].str.upper().str.strip()

        df["Firm_ID"]                  = mapped_upper.map(secp_to_firm_ci)
        df["GU_COMPANY_NAME_universe"] = mapped_upper.map(secp_to_gu_ci)

        fill_gu = df["Firm_ID"].isna() & mapped_upper.map(gu_to_firm_ci).notna()
        df.loc[fill_gu, "Firm_ID"] = mapped_upper.map(gu_to_firm_ci)[fill_gu]

        p1 = int(df["Firm_ID"].notna().sum())
        print(f"  Pass 1 (SECP_NAME_MAPPED): {p1:,} Firm_IDs resolved")

    # Pass 2 – raw SECP_NAME fallback for rows with no mapped name
    secp_upper = df["SECP_NAME"].str.upper().str.strip()

    fill_firm = df["Firm_ID"].isna() & secp_upper.map(secp_to_firm_ci).notna()
    fill_gu   = df["GU_COMPANY_NAME_universe"].isna() & secp_upper.map(secp_to_gu_ci).notna()

    df.loc[fill_firm, "Firm_ID"]                  = secp_upper.map(secp_to_firm_ci)[fill_firm]
    df.loc[fill_gu,   "GU_COMPANY_NAME_universe"] = secp_upper.map(secp_to_gu_ci)[fill_gu]

    p2 = int(fill_firm.sum())
    print(f"  Pass 2 (raw SECP_NAME fallback): {p2:,} additional Firm_IDs")

    total = int(df["Firm_ID"].notna().sum())
    pct   = total / len(df) * 100 if len(df) else 0
    print(f"  Total: {total:,} / {len(df):,} rows have Firm_ID ({pct:.1f}%)")

    return df


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Re-enrich existing output CSV with universe Firm_ID.")
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
    print("\nLoading universe...")
    universe = pd.read_csv(
        UNIVERSE_FILE,
        sep="\t",
        encoding="latin-1",
        usecols=["SECP_NAME", "GU_COMPANY_NAME", "Firm_ID"],
        dtype={"Firm_ID": "Int64"},
    )
    print(f"  {len(universe):,} universe rows loaded")

    print("Loading results CSV...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df):,} result rows loaded")

    # ── build lookups & enrich ───────────────────────────────────────────────
    print("\nBuilding lookups...")
    secp_to_firm_ci, secp_to_gu_ci, gu_to_firm_ci = build_lookups(universe)

    print("Enriching...")
    df = enrich(df, secp_to_firm_ci, secp_to_gu_ci, gu_to_firm_ci)

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
