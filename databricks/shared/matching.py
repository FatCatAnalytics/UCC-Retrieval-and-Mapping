"""Thin shims around the existing Dagster matching logic.

These functions rewrap the per-stage logic from
``ucc_mapping_project.ucc_mapping_project.assets`` so it can be called
from plain Python (no Dagster context, no PipelineConfig resource).

Where the underlying asset is a pure pandas function, we copy the body
into a context-free helper. Where it is too tangled to lift cleanly,
we leave a TODO and fall back to importing the asset's ``op`` directly
inside a ``build_op_context`` shim.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ucc_mapping_project.ucc_mapping_project import assets as _assets
from ucc_mapping_project.ucc_mapping_project.utils import normalize_name
from ucc_mapping_project.ucc_mapping_project.resources import PipelineConfig

log = logging.getLogger(__name__)


# Re-export key constants so task notebooks can stay terse
RAW_MATCH_COLUMNS = _assets.RAW_MATCH_COLUMNS
STATE_COLUMNS = _assets.STATE_COLUMNS
NO_MASTER_STATE_PAIR = _assets.NO_MASTER_STATE_PAIR


# ---------------------------------------------------------------------------
# Dagster-context shim
# ---------------------------------------------------------------------------

class _LogShim:
    def info(self, msg: str) -> None:
        log.info(msg)

    def warning(self, msg: str) -> None:
        log.warning(msg)

    def error(self, msg: str) -> None:
        log.error(msg)

    def debug(self, msg: str) -> None:
        log.debug(msg)


@dataclass
class _ContextShim:
    """Mimics the bits of ``AssetExecutionContext`` the assets actually use."""
    log: _LogShim


def _ctx() -> _ContextShim:
    return _ContextShim(log=_LogShim())


# ---------------------------------------------------------------------------
# Pandas-only wrappers around assets
# ---------------------------------------------------------------------------

def make_pipeline_config(
    *,
    ucc_file: str,
    master_file: str,
    universe_file: str,
    output_dir: str,
    embedding_model: str,
    fuzzy_auto_threshold: int = 96,
    fuzzy_review_threshold: int = 85,
    embed_auto_threshold: float = 0.95,
    embed_high_threshold: float = 0.90,
    embed_medium_threshold: float = 0.85,
    embed_top_k: int = 20,
    embed_batch_size: int = 256,
    fuzzy_batch_size: int = 1024,
    fuzzy_workers: int = -1,
    cpu_threads: int = -1,
    embedding_workers: int = 4,
    early_stop_threshold: float = 0.98,
    max_candidates_per_query: int = 50,
) -> PipelineConfig:
    """Construct a :class:`PipelineConfig` using only kwargs (no env)."""
    return PipelineConfig(
        ucc_file=ucc_file,
        master_file=master_file,
        universe_file=universe_file,
        output_dir=output_dir,
        embedding_model=embedding_model,
        fuzzy_auto_threshold=fuzzy_auto_threshold,
        fuzzy_review_threshold=fuzzy_review_threshold,
        embed_auto_threshold=embed_auto_threshold,
        embed_high_threshold=embed_high_threshold,
        embed_medium_threshold=embed_medium_threshold,
        embed_top_k=embed_top_k,
        embed_batch_size=embed_batch_size,
        fuzzy_batch_size=fuzzy_batch_size,
        fuzzy_workers=fuzzy_workers,
        cpu_threads=cpu_threads,
        embedding_workers=embedding_workers,
        early_stop_threshold=early_stop_threshold,
        max_candidates_per_query=max_candidates_per_query,
    )


# ---------------------------------------------------------------------------
# Stage wrappers — delegate to the .compute_fn() of each Dagster asset.
# This avoids having to re-implement the logic and keeps both
# Dagster + Databricks pipelines on the same code path.
# ---------------------------------------------------------------------------

def _call_asset(asset_def, **kwargs):
    """Invoke a Dagster ``@asset`` fn with our shim context.

    The Dagster ``AssetsDefinition`` exposes its underlying function via
    ``op.compute_fn.decorated_fn``. We call that directly so the asset
    body runs as a normal Python function.
    """
    try:
        fn = asset_def.op.compute_fn.decorated_fn
    except AttributeError:
        # Dagster API surface has changed across versions — try alternates.
        fn = getattr(asset_def, "_op_def", asset_def).compute_fn.decorated_fn
    return fn(context=_ctx(), **kwargs)


def load_ucc_raw(config: PipelineConfig) -> pd.DataFrame:
    """Wrapper around :func:`assets.ucc_raw_data`."""
    out = _call_asset(_assets.ucc_raw_data, config=config)
    return getattr(out, "value", out)


def load_master_mapping(config: PipelineConfig) -> pd.DataFrame:
    """Wrapper around :func:`assets.master_mapping_data`."""
    out = _call_asset(_assets.master_mapping_data, config=config)
    return getattr(out, "value", out)


def load_tracked_banks(config: PipelineConfig) -> pd.DataFrame:
    """Wrapper around :func:`assets.our_universe_data`."""
    out = _call_asset(_assets.our_universe_data, config=config)
    return getattr(out, "value", out)


def build_normalized_master(master_df: pd.DataFrame) -> dict:
    """Wrapper around :func:`assets.normalized_master_data`."""
    return _call_asset(_assets.normalized_master_data, master_mapping_data=master_df)


def build_master_embeddings_cache(
    config: PipelineConfig, normalized_master: dict
) -> dict:
    return _call_asset(
        _assets.master_embeddings_cache,
        config=config,
        normalized_master_data=normalized_master,
    )


def stage_exact_raw(ucc_df: pd.DataFrame, normalized_master: dict) -> pd.DataFrame:
    return _call_asset(
        _assets.exact_match_raw_results,
        ucc_raw_data=ucc_df,
        normalized_master_data=normalized_master,
    )


def stage_exact_normalized(prev_df: pd.DataFrame, normalized_master: dict) -> pd.DataFrame:
    return _call_asset(
        _assets.exact_match_normalized_results,
        exact_match_raw_results=prev_df,
        normalized_master_data=normalized_master,
    )


def stage_fuzzy(
    config: PipelineConfig, prev_df: pd.DataFrame, normalized_master: dict
) -> pd.DataFrame:
    return _call_asset(
        _assets.fuzzy_match_results,
        config=config,
        exact_match_normalized_results=prev_df,
        normalized_master_data=normalized_master,
    )


def stage_embedding(
    config: PipelineConfig,
    prev_df: pd.DataFrame,
    normalized_master: dict,
    master_cache: dict,
) -> pd.DataFrame:
    return _call_asset(
        _assets.advanced_embedding_results,
        config=config,
        fuzzy_match_results=prev_df,
        normalized_master_data=normalized_master,
        master_embeddings_cache=master_cache,
    )


def stage_clustered(
    config: PipelineConfig, prev_df: pd.DataFrame, master_cache: dict
) -> pd.DataFrame:
    return _call_asset(
        _assets.clustered_results,
        config=config,
        advanced_embedding_results=prev_df,
        master_embeddings_cache=master_cache,
    )


def stage_merged_clusters(
    config: PipelineConfig, prev_df: pd.DataFrame, master_cache: dict
) -> pd.DataFrame:
    return _call_asset(
        _assets.merged_cluster_results,
        config=config,
        clustered_results=prev_df,
        master_embeddings_cache=master_cache,
    )


def stage_suggested(
    config: PipelineConfig,
    prev_df: pd.DataFrame,
    normalized_master: dict,
    master_cache: dict,
) -> pd.DataFrame:
    return _call_asset(
        _assets.suggested_mapping_results,
        config=config,
        merged_cluster_results=prev_df,
        normalized_master_data=normalized_master,
        master_embeddings_cache=master_cache,
    )


def stage_universe_enriched(
    suggested_df: pd.DataFrame, universe_df: pd.DataFrame
) -> pd.DataFrame:
    return _call_asset(
        _assets.universe_enriched_results,
        suggested_mapping_results=suggested_df,
        our_universe_data=universe_df,
    )


# ---------------------------------------------------------------------------
# Convenience: classify rows by output bucket
# ---------------------------------------------------------------------------

def split_by_confidence(df: pd.DataFrame):
    """Return (auto_df, review_df, new_entity_df) using the same logic as
    :func:`assets.pipeline_outputs`.
    """
    auto_mask = df["confidence"] == "AUTO"
    review_mask = df["confidence"].isin(["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW"])
    new_mask = df["confidence"] == "NEW_ENTITY"
    return df[auto_mask].copy(), df[review_mask].copy(), df[new_mask].copy()


def is_master_match(method: pd.Series) -> pd.Series:
    s = method.astype("string")
    return method.notna() & (
        s.str.startswith("Exact Match", na=False)
        | s.str.startswith("Fuzzy Match", na=False)
        | s.str.startswith("Advanced Match", na=False)
    )
