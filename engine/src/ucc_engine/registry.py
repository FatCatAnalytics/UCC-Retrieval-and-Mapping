"""The pipeline DAG as data.

Each ``Stage`` mirrors one Dagster asset. ``deps`` are listed in the exact
positional order the core function expects (after ``context`` and the optional
``config``). ``PIPELINE`` is a valid topological order. ``weight`` is a rough
relative cost used only for the progress bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .core import pipeline_logic as L


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    group: str
    needs_config: bool
    deps: List[str] = field(default_factory=list)
    kind: str = "frame"  # frame | dict | sink
    weight: int = 1
    # Name of a boolean PipelineConfig field that switches this stage off. A
    # switched-off stage is replaced by ``passthrough`` so downstream stages
    # still find their input under this stage's name.
    enabled_by: str = ""


# Columns Stages 5-7 add; downstream code tolerates them being empty.
CLUSTER_COLUMNS: Dict[str, object] = {
    "cluster_id": None, "cluster_label": None, "cluster_size": np.nan,
    "suggested_mapping": None, "suggestion_score": np.nan, "suggestion_confidence": None,
    "alternative_suggestions": None,
}


def is_enabled(stage: "Stage", config) -> bool:
    return not stage.enabled_by or bool(getattr(config, stage.enabled_by, True))


def passthrough(stage: "Stage", cache: Dict[str, object]) -> pd.DataFrame:
    """Stand-in for a disabled frame stage: forward the upstream frame unchanged
    apart from the columns the stage would have added. For the clustering group
    this also labels still-unmatched rows ``NEW_ENTITY`` exactly as Stage 5 does
    before grouping them, so exports and the updated master are unaffected."""
    src = cache[stage.deps[0]]
    assert isinstance(src, pd.DataFrame), f"{stage.name}: passthrough needs a frame input"
    df = src.copy(deep=False)
    df.attrs = {}  # never carry model handles / embeddings on the results frame
    if stage.group == "clustering":
        for col, empty in CLUSTER_COLUMNS.items():
            if col not in df.columns:
                df[col] = pd.Series(empty, index=df.index, dtype="float64" if empty is np.nan else "object")
        if stage.name == "clustered_results":
            df.loc[L._pending_match_mask(df), "confidence"] = "NEW_ENTITY"
    return df


PIPELINE: List[Stage] = [
    Stage("ucc_raw_data", "Load UCC filings", "ingestion", True, [], "frame", 1),
    Stage("master_mapping_data", "Load master mapping", "ingestion", True, [], "frame", 1),
    Stage("normalized_master_data", "Normalise master", "ingestion", False, ["master_mapping_data"], "dict", 2),
    Stage("our_universe_data", "Load universe", "ingestion", True, [], "frame", 1),
    Stage("master_embeddings_cache", "Embed master (local model)", "preprocessing", True,
          ["normalized_master_data"], "dict", 6),
    Stage("exact_match_raw_results", "Stage 1 · exact match (raw)", "matching", False,
          ["ucc_raw_data", "normalized_master_data"], "frame", 1),
    Stage("exact_match_normalized_results", "Stage 2 · exact match (normalised)", "matching", False,
          ["exact_match_raw_results", "normalized_master_data"], "frame", 1),
    Stage("fuzzy_match_results", "Stage 3 · fuzzy match", "matching", True,
          ["exact_match_normalized_results", "normalized_master_data", "master_embeddings_cache"], "frame", 6),
    Stage("advanced_embedding_results", "Stage 4 · embedding match + rerank", "matching", True,
          ["fuzzy_match_results", "normalized_master_data", "master_embeddings_cache"], "frame", 28),
    Stage("clustered_results", "Stage 5 · cluster unmatched", "clustering", True,
          ["advanced_embedding_results", "master_embeddings_cache"], "frame", 16, enabled_by="clustering_enabled"),
    Stage("merged_cluster_results", "Stage 6 · merge clusters", "clustering", True,
          ["clustered_results", "master_embeddings_cache"], "frame", 10, enabled_by="clustering_enabled"),
    Stage("suggested_mapping_results", "Stage 7 · suggest mappings", "clustering", True,
          ["merged_cluster_results", "normalized_master_data", "master_embeddings_cache"], "frame", 10,
          enabled_by="clustering_enabled"),
    Stage("universe_enriched_results", "Join universe Firm IDs", "reporting", False,
          ["suggested_mapping_results", "our_universe_data"], "frame", 1),
    Stage("updated_master_data", "Build updated master", "reporting", True,
          ["master_mapping_data", "universe_enriched_results"], "frame", 1),
    Stage("merged_raw_with_firm_id", "Merge raw filings with Firm IDs", "reporting", True,
          ["updated_master_data"], "frame", 3),
    Stage("lead_bank_designation", "Stage 8 · lead-bank designation", "reporting", True,
          ["universe_enriched_results"], "frame", 3),
    Stage("pipeline_outputs", "Write output files", "reporting", True,
          ["universe_enriched_results"], "sink", 3),
]

STAGES_BY_NAME: Dict[str, Stage] = {s.name: s for s in PIPELINE}


def total_weight(config=None) -> int:
    return sum(s.weight for s in PIPELINE if config is None or is_enabled(s, config))

# Frames worth keeping in memory after a run (everything else is freed).
KEEP_AFTER_RUN = ("universe_enriched_results", "updated_master_data", "lead_bank_designation")


def run_stage(stage: Stage, config, ctx, cache: Dict[str, object]):
    """Execute one stage in-process, reading/writing the shared ``cache``."""
    if not is_enabled(stage, config):
        result = passthrough(stage, cache)
        cache[stage.name] = result
        return result
    fn = getattr(L, stage.name)
    args = [ctx]
    if stage.needs_config:
        args.append(config)
    args.extend(cache[d] for d in stage.deps)
    result = fn(*args)
    cache[stage.name] = result
    return result
