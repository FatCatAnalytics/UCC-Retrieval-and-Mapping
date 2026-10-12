"""Run the entire pipeline in a single process (recommended default).

This is the simplest and fastest way to run on Databricks: one task, all stages
executed in topological order with intermediates held in memory (no pickle
round-trips). The dict caches (``normalized_master_data``,
``master_embeddings_cache``) never touch disk, so this avoids re-encoding
embeddings between stages.

Usage:
    python -m ucc_pipeline.run_pipeline \
        --conf ucc_file=/Volumes/main/ucc/data/Inputs/raw.csv \
        --conf master_file=/Volumes/main/ucc/data/Inputs/master_list.csv \
        --conf universe_file=/Volumes/main/ucc/data/Inputs/tracked_banks.csv \
        --conf output_dir=/Volumes/main/ucc/data/Outputs
"""
from __future__ import annotations

from .cli import base_parser, build_config
from .registry import PIPELINE, run_stage


def main(argv=None) -> None:
    parser = base_parser("Run the full UCC mapping pipeline in one process")
    args = parser.parse_args(argv)

    config = build_config(args.conf)
    cache: dict = {}
    for stage in PIPELINE:
        print(f"[ucc_pipeline] running stage: {stage.name}")
        run_stage(stage, config, cache=cache)
    print("[ucc_pipeline] pipeline complete")


if __name__ == "__main__":
    main()
