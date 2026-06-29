"""Shared CLI / parameter plumbing for the Databricks task entrypoints."""
from __future__ import annotations

import argparse
from typing import Dict, List

from .config import PipelineConfig


def _kv_list_to_dict(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--conf expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


def build_config(conf_pairs: List[str] | None) -> PipelineConfig:
    """Build a PipelineConfig from --conf key=value job parameters."""
    return PipelineConfig.from_overrides(_kv_list_to_dict(conf_pairs or []))


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--conf",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a PipelineConfig field (repeatable). "
        "E.g. --conf ucc_file=/Volumes/main/ucc/data/Inputs/raw.csv",
    )
    return parser
