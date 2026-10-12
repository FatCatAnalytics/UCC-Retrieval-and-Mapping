"""The pipeline DAG as data, plus a generic stage runner.

Each entry mirrors one Dagster asset from the original ``assets.py``:

    Stage(name, needs_config, deps, kind, publish_table)

* ``deps`` are upstream stage names, **in the exact order** the ported logic
  function expects its positional arguments (after ``context`` and the optional
  ``config``).
* ``kind`` is "frame" (pandas DataFrame output), "dict" (in-memory cache) or
  "sink" (writes files, no usable return).
* ``publish_table`` -> when set, the frame output is also written to a Unity
  Catalog Delta table of that name.

The order of ``PIPELINE`` is a valid topological order of the DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import pipeline_logic as L
from .dagster_compat import AssetExecutionContext
from . import io_utils


@dataclass(frozen=True)
class Stage:
    name: str
    needs_config: bool
    deps: List[str] = field(default_factory=list)
    kind: str = "frame"            # frame | dict | sink
    publish_table: Optional[str] = None


PIPELINE: List[Stage] = [
    # ---- ingestion ----
    Stage("ucc_raw_data", True, [], "frame"),
    Stage("master_mapping_data", True, [], "frame"),
    Stage("normalized_master_data", False, ["master_mapping_data"], "dict"),
    Stage("our_universe_data", True, [], "frame"),
    # ---- preprocessing ----
    Stage("master_embeddings_cache", True, ["normalized_master_data"], "dict"),
    # ---- matching (Stages 1-4) ----
    Stage("exact_match_raw_results", False,
          ["ucc_raw_data", "normalized_master_data"], "frame"),
    Stage("exact_match_normalized_results", False,
          ["exact_match_raw_results", "normalized_master_data"], "frame"),
    Stage("fuzzy_match_results", True,
          ["exact_match_normalized_results", "normalized_master_data",
           "master_embeddings_cache"], "frame"),
    Stage("advanced_embedding_results", True,
          ["fuzzy_match_results", "normalized_master_data",
           "master_embeddings_cache"], "frame"),
    # ---- clustering (Stages 5-7) ----
    Stage("clustered_results", True,
          ["advanced_embedding_results", "master_embeddings_cache"], "frame"),
    Stage("merged_cluster_results", True,
          ["clustered_results", "master_embeddings_cache"], "frame"),
    Stage("suggested_mapping_results", True,
          ["merged_cluster_results", "normalized_master_data",
           "master_embeddings_cache"], "frame"),
    # ---- reporting ----
    Stage("universe_enriched_results", False,
          ["suggested_mapping_results", "our_universe_data"], "frame",
          publish_table="universe_enriched_results"),
    Stage("updated_master_data", True,
          ["master_mapping_data", "universe_enriched_results"], "frame",
          publish_table="updated_master_data"),
    Stage("merged_raw_with_firm_id", True,
          ["updated_master_data"], "frame",
          publish_table="merged_raw_with_firm_id"),
    Stage("lead_bank_designation", True,
          ["universe_enriched_results"], "frame",
          publish_table="lead_bank_designation"),
    Stage("pipeline_outputs", True,
          ["universe_enriched_results"], "sink"),
]

STAGES_BY_NAME: Dict[str, Stage] = {s.name: s for s in PIPELINE}


def run_stage(stage: Stage, config, ctx=None, cache: Optional[Dict] = None):
    """Execute one stage.

    ``cache`` (optional) is an in-memory {name: output} map used by the
    single-process driver. When ``cache`` is None, upstream outputs are loaded
    from the staging volume and this stage's output is persisted back to it.
    """
    ctx = ctx or AssetExecutionContext(stage.name)
    fn = getattr(L, stage.name)

    # Resolve upstream inputs (positional order matters).
    if cache is not None:
        inputs = [cache[d] for d in stage.deps]
    else:
        inputs = [io_utils.load_stage(d) for d in stage.deps]

    args = [ctx]
    if stage.needs_config:
        args.append(config)
    args.extend(inputs)

    result = fn(*args)

    if cache is not None:
        cache[stage.name] = result
    elif stage.kind != "sink":
        io_utils.save_stage(result, stage.name)

    if stage.publish_table and stage.kind == "frame" and result is not None:
        io_utils.write_delta(result, stage.publish_table)

    return result
