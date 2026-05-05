"""Centralized path / table-name conventions for Databricks tasks.

All path-building goes through this module so that swapping volumes or
schemas only requires editing one file.
"""
from __future__ import annotations

from .config import TaskConfig


# ---------------------------------------------------------------------------
# Volume paths
# ---------------------------------------------------------------------------

def bootstrap_master_path(cfg: TaskConfig) -> str:
    return f"{cfg.landing_root}/bootstrap/master_list.csv"


def bootstrap_tracked_banks_path(cfg: TaskConfig) -> str:
    return f"{cfg.landing_root}/bootstrap/tracked_banks.csv"


def source_ucc_path(cfg: TaskConfig) -> str:
    return f"{cfg.landing_root}/{cfg.source_month}/source/ucc.csv"


def source_dir(cfg: TaskConfig) -> str:
    return f"{cfg.landing_root}/{cfg.source_month}/source"


def review_return_path(cfg: TaskConfig, filename: str = "completed_review.xlsx") -> str:
    return f"{cfg.landing_root}/{cfg.source_month}/review_return/{filename}"


def review_return_dir(cfg: TaskConfig) -> str:
    return f"{cfg.landing_root}/{cfg.source_month}/review_return"


def review_export_dir(cfg: TaskConfig) -> str:
    return f"{cfg.gold_export_root}/{cfg.source_month}/review_out"


def review_export_xlsx(cfg: TaskConfig) -> str:
    return f"{review_export_dir(cfg)}/review_workbook_{cfg.source_month}.xlsx"


def final_export_dir(cfg: TaskConfig) -> str:
    return f"{cfg.gold_export_root}/{cfg.source_month}/final"


# ---------------------------------------------------------------------------
# Table names — namespaced under the catalog/schemas in TaskConfig
# ---------------------------------------------------------------------------

def t_ucc_raw(cfg: TaskConfig) -> str:
    return cfg.table(cfg.bronze_schema, "ucc_raw")


def t_review_workbook_raw(cfg: TaskConfig) -> str:
    return cfg.table(cfg.bronze_schema, "review_workbook_raw")


def t_ucc_clean(cfg: TaskConfig) -> str:
    return cfg.table(cfg.silver_schema, "ucc_clean")


def t_master_list_history(cfg: TaskConfig) -> str:
    return cfg.table(cfg.silver_schema, "master_list_history")


def t_master_list_current(cfg: TaskConfig) -> str:
    return cfg.table(cfg.silver_schema, "master_list_current")


def t_review_decisions(cfg: TaskConfig) -> str:
    return cfg.table(cfg.silver_schema, "review_decisions")


def t_review_decision_errors(cfg: TaskConfig) -> str:
    return cfg.table(cfg.silver_schema, "review_decision_errors")


def t_exact_mapped(cfg: TaskConfig) -> str:
    return cfg.table(cfg.gold_schema, "exact_mapped")


def t_review_candidates(cfg: TaskConfig) -> str:
    return cfg.table(cfg.gold_schema, "review_candidates")


def t_final_ucc_mapping(cfg: TaskConfig) -> str:
    return cfg.table(cfg.gold_schema, "final_ucc_mapping")


def t_monthly_summary(cfg: TaskConfig) -> str:
    return cfg.table(cfg.gold_schema, "monthly_summary")


def t_pipeline_runs(cfg: TaskConfig) -> str:
    return cfg.table(cfg.audit_schema, "pipeline_runs")


def t_source_files(cfg: TaskConfig) -> str:
    return cfg.table(cfg.audit_schema, "source_files")


def t_review_workbooks(cfg: TaskConfig) -> str:
    return cfg.table(cfg.audit_schema, "review_workbooks")


def t_master_list_changes(cfg: TaskConfig) -> str:
    return cfg.table(cfg.audit_schema, "master_list_changes")
