"""Shared helpers for Databricks UCC monthly mapping tasks.

These modules wrap configuration, path/table conventions, audit logging,
and thin shims over the existing pandas/Dagster matching logic in
``ucc_mapping_project.ucc_mapping_project`` so each task notebook can
stay short and declarative.
"""
