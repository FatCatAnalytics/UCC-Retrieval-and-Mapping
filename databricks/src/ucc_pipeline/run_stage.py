"""Run a single pipeline stage — the entrypoint each Databricks Job task uses.

Usage (Databricks python_wheel_task or spark_python_task):
    python -m ucc_pipeline.run_stage --stage fuzzy_match_results \
        --conf ucc_file=/Volumes/main/ucc/data/Inputs/raw.csv

Upstream outputs are read from, and this stage's output is written to, the
staging volume (see io_utils.STAGE_DIR). Task ordering is enforced by the
``depends_on`` wiring in databricks.yml.
"""
from __future__ import annotations

from .cli import base_parser, build_config
from .registry import STAGES_BY_NAME, run_stage


def main(argv=None) -> None:
    parser = base_parser("Run one UCC mapping pipeline stage")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES_BY_NAME))
    args = parser.parse_args(argv)

    config = build_config(args.conf)
    stage = STAGES_BY_NAME[args.stage]
    run_stage(stage, config)
    print(f"[ucc_pipeline] stage '{stage.name}' complete")


if __name__ == "__main__":
    main()
