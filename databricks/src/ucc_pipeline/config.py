"""Plain-dataclass replacement for the Dagster ``PipelineConfig`` resource.

Field names and defaults mirror ``ucc_mapping_project/resources.py`` exactly so
that ``pipeline_logic.py`` (the ported asset bodies) reads them unchanged via
attribute access (``config.fuzzy_auto_threshold`` etc.).

On Databricks, file-path fields normally point at Unity Catalog **Volumes**
(``/Volumes/<catalog>/<schema>/<volume>/...``) and are supplied as Job
parameters; everything else can stay on its default. Use
``PipelineConfig.from_overrides({...})`` to apply job parameters / widget values
on top of the defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Dict


# Base directory for default file paths. On Databricks set UCC_DATA_ROOT to a
# Unity Catalog Volume path, e.g. /Volumes/main/ucc/data.
_DATA_ROOT = os.environ.get("UCC_DATA_ROOT", "/Volumes/main/ucc/data")


@dataclass
class PipelineConfig:
    """Central configuration for the mapping pipeline (Databricks port)."""

    # --- File paths (point these at UC Volumes via job params) ---
    ucc_file: str = f"{_DATA_ROOT}/Inputs/ucc.csv"
    master_file: str = f"{_DATA_ROOT}/Inputs/Master Mapping.csv"
    universe_file: str = f"{_DATA_ROOT}/Inputs/Universe.csv"
    output_dir: str = f"{_DATA_ROOT}/Outputs"

    # --- Embedding model (local volume path or HuggingFace id) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Matching thresholds ---
    fuzzy_auto_threshold: int = 96
    fuzzy_review_threshold: int = 85
    fuzzy_min_threshold: int = 80

    embed_auto_threshold: float = 0.95
    embed_high_threshold: float = 0.90
    embed_medium_threshold: float = 0.85
    embed_top_k: int = 20

    # --- Clustering parameters ---
    cluster_sim_threshold: float = 0.88
    cluster_merge_threshold: float = 0.88
    cluster_k_neighbors: int = 20
    cluster_fuzzy_guard: int = 80
    cluster_merge_iterations: int = 1
    cluster_merge_k: int = 10
    cluster_max_size: int = 500

    # --- Processing parameters ---
    embed_batch_size: int = 256
    search_batch_size: int = 4096
    fuzzy_batch_size: int = 1024
    fuzzy_workers: int = -1

    # --- CPU threading ---
    cpu_threads: int = -1
    embedding_workers: int = 4

    # --- Early termination ---
    early_stop_threshold: float = 0.98
    max_candidates_per_query: int = 50

    # --- Lead-bank designation (Stage 8) ---
    lead_bank_threshold: float = 0.66
    filing_type_column: str = ""
    debtor_company_column: str = ""

    # --- Clustering performance ---
    cluster_memmap_threshold_mb: int = 512
    cluster_hnsw_threshold: int = 200000
    cluster_hnsw_m: int = 32
    cluster_hnsw_ef_search: int = 128
    cluster_hnsw_ef_construction: int = 200

    # --- Stage 4 advanced-match perf/accuracy ---
    advanced_use_per_state_faiss: bool = True
    advanced_per_state_min_size: int = 64
    query_embed_cache_enabled: bool = True
    query_embed_cache_path: str = f"{_DATA_ROOT}/.cache/query_embeddings"

    # --- Cross-encoder rerank (REVIEW_MEDIUM band) ---
    advanced_cross_encoder_enabled: bool = True
    advanced_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    advanced_cross_encoder_top_n: int = 10
    advanced_cross_encoder_promote_to_auto: float = 0.92
    advanced_cross_encoder_promote_to_high: float = 0.80

    @classmethod
    def from_overrides(cls, overrides: Dict[str, Any] | None = None) -> "PipelineConfig":
        """Build a config from defaults, applying only known overrides.

        Values arriving from Databricks Job parameters / widgets are strings, so
        each override is coerced to the dataclass field's declared type.
        """
        overrides = dict(overrides or {})
        type_by_name = {f.name: f.type for f in fields(cls)}
        clean: Dict[str, Any] = {}
        for key, raw in overrides.items():
            if key not in type_by_name or raw is None or raw == "":
                continue
            clean[key] = _coerce(type_by_name[key], raw)
        return cls(**clean)


def _coerce(declared_type: Any, value: Any) -> Any:
    t = str(declared_type)
    if value is None:
        return None
    if "bool" in t:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    return str(value)
