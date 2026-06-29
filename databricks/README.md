# UCC Retrieval & Mapping — Databricks port

This folder is a Databricks-native reconstruction of the Dagster pipeline in
`../ucc_mapping_project`. The original Dagster code is left untouched; this is an
additive `databricks/` package you wire into Unity Catalog + Databricks Jobs.

## Design

The matching/clustering logic is **identical** to the proven Dagster
implementation. `src/ucc_pipeline/pipeline_logic.py` is a byte-for-byte copy of
the original `assets.py` with only three import lines changed:

| original `assets.py` | port |
|---|---|
| `from dagster import asset, AssetExecutionContext, Output, MetadataValue, MaterializeResult` | `from .dagster_compat import ...` |
| `from .resources import PipelineConfig` | `from .config import PipelineConfig` |
| *(added)* | `from __future__ import annotations` (keeps return annotations lazy) |

`dagster_compat.py` provides tiny shims so each `@asset` function becomes a plain
callable: `asset` is a pass-through decorator, `Output(value=...)` returns the
value, `AssetExecutionContext.log` maps to stdlib logging, `MetadataValue`/
`MaterializeResult` are no-ops. This means **zero edits to the matching logic** —
the highest-fidelity migration possible.

Orchestration (the implicit Dagster asset DAG) becomes explicit:
`registry.py` encodes the DAG as data (each asset, its upstream deps, and the
output kind), and the two CLI entrypoints execute it.

### Package layout

```
databricks/
├── databricks.yml          # Asset Bundle: 2 jobs (single-task + 17-task DAG)
├── pyproject.toml          # builds the ucc_pipeline wheel + console entrypoints
├── requirements.txt        # non-default deps (rapidfuzz, sentence-transformers, faiss-cpu)
└── src/ucc_pipeline/
    ├── pipeline_logic.py    # the 17 asset functions (copied from assets.py, unchanged logic)
    ├── dagster_compat.py    # Dagster shims (asset/Output/context/MetadataValue)
    ├── config.py            # PipelineConfig dataclass (mirrors resources.py)
    ├── registry.py          # the DAG as data + generic stage runner
    ├── io_utils.py          # staging-volume pickle I/O + UC Delta publishing
    ├── cli.py               # --conf key=value parameter plumbing
    ├── run_pipeline.py      # entrypoint: run whole DAG in one process (recommended)
    ├── run_stage.py         # entrypoint: run a single stage (used by the staged job)
    ├── constants.py, utils.py, embedding_cache.py, cross_encoder_rerank.py  # copied unchanged
    └── models/data_structures.py                                            # copied unchanged
```

## Data passing between tasks

Stage outputs are not all tabular: `normalized_master_data` and
`master_embeddings_cache` are Python dicts of numpy arrays / FAISS structures,
and some result frames carry list-valued columns. So intermediates are exchanged
as **pickle files on a Unity Catalog Volume** (`UCC_STAGE_DIR`), which is
lossless. Curated result frames are *additionally* published as **UC Delta
tables** (`universe_enriched_results`, `updated_master_data`,
`merged_raw_with_firm_id`, `lead_bank_designation`). The single-task job keeps
everything in memory and skips the pickle round-trips entirely.

## Configuration & secrets

All tuning lives in `config.PipelineConfig` (mirrors the Dagster
`PipelineConfig` resource). Override any field with `--conf key=value` job
parameters. Environment variables (set as cluster `spark_env_vars`):

| Env var | Meaning | Default |
|---|---|---|
| `UCC_DATA_ROOT` | UC Volume root with `Inputs/` and `Outputs/` | `/Volumes/main/ucc/data` |
| `UCC_STAGE_DIR` | UC Volume dir for intermediate pickles | `$UCC_DATA_ROOT/_staging` |
| `UCC_CATALOG` / `UCC_SCHEMA` | UC target for published Delta tables | `main` / `ucc` |

**Secrets:** the pipeline needs none (no DBs, no APIs, no cloud keys). The only
network access is the optional HuggingFace model download for
`embedding_model` (default `BAAI/bge-small-en-v1.5`) and the optional
cross-encoder. To run fully offline, stage those models on a Volume and point
`--conf embedding_model=/Volumes/.../bge-small-en-v1.5`. If you later add a
secret, read it via `io_utils.get_dbutils().secrets.get(scope, key)`.

## Deploy & run

Prerequisites: Databricks CLI ≥ 0.218, a UC catalog/schema, and a Volume holding
the input CSVs under `Inputs/` (`raw.csv`, `master_list.csv`, `tracked_banks.csv`).

```bash
cd databricks

# 1) point the bundle at your workspace (edit databricks.yml `host`, or):
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=...

# 2) (optional) override variables for your cloud/UC
#    --var catalog=main --var schema=ucc --var data_root=/Volumes/main/ucc/data
#    --var node_type_id=i3.xlarge --var spark_version=15.4.x-cpu-ml-scala2.12

# 3) deploy (builds the wheel via `python -m build`) and run
databricks bundle deploy -t dev
databricks bundle run ucc_mapping_pipeline_single -t dev     # recommended
# or the fully-wired DAG:
databricks bundle run ucc_mapping_pipeline_staged -t dev
```

Run locally / in a notebook without the bundle:

```bash
pip install -r requirements.txt && pip install -e .
python -m ucc_pipeline.run_pipeline \
  --conf ucc_file=/Volumes/main/ucc/data/Inputs/raw.csv \
  --conf master_file=/Volumes/main/ucc/data/Inputs/master_list.csv \
  --conf universe_file=/Volumes/main/ucc/data/Inputs/tracked_banks.csv \
  --conf output_dir=/Volumes/main/ucc/data/Outputs
```

## Old Dagster asset → new Databricks task mapping

Every asset maps 1:1 to a `pipeline_logic` function and a `--stage` task. The
single-task job runs them all in topological order.

| # | Dagster asset (group) | `--stage` value | Upstream deps | Output |
|---|---|---|---|---|
| 1 | `ucc_raw_data` (ingestion) | `ucc_raw_data` | — | DataFrame |
| 2 | `master_mapping_data` (ingestion) | `master_mapping_data` | — | DataFrame |
| 3 | `normalized_master_data` (ingestion) | `normalized_master_data` | master_mapping_data | dict (cache) |
| 4 | `our_universe_data` (ingestion) | `our_universe_data` | — | DataFrame |
| 5 | `master_embeddings_cache` (preprocessing) | `master_embeddings_cache` | normalized_master_data | dict (cache) |
| 6 | `exact_match_raw_results` — Stage 1 (matching) | `exact_match_raw_results` | ucc_raw_data, normalized_master_data | DataFrame |
| 7 | `exact_match_normalized_results` — Stage 2 | `exact_match_normalized_results` | exact_match_raw_results, normalized_master_data | DataFrame |
| 8 | `fuzzy_match_results` — Stage 3 | `fuzzy_match_results` | exact_match_normalized_results, normalized_master_data, master_embeddings_cache | DataFrame |
| 9 | `advanced_embedding_results` — Stage 4 | `advanced_embedding_results` | fuzzy_match_results, normalized_master_data, master_embeddings_cache | DataFrame |
| 10 | `clustered_results` — Stage 5 (clustering) | `clustered_results` | advanced_embedding_results, master_embeddings_cache | DataFrame |
| 11 | `merged_cluster_results` — Stage 6 | `merged_cluster_results` | clustered_results, master_embeddings_cache | DataFrame |
| 12 | `suggested_mapping_results` — Stage 7 | `suggested_mapping_results` | merged_cluster_results, normalized_master_data, master_embeddings_cache | DataFrame |
| 13 | `universe_enriched_results` (reporting) | `universe_enriched_results` | suggested_mapping_results, our_universe_data | DataFrame → UC table |
| 14 | `updated_master_data` (reporting) | `updated_master_data` | master_mapping_data, universe_enriched_results | DataFrame → UC table |
| 15 | `merged_raw_with_firm_id` (reporting) | `merged_raw_with_firm_id` | updated_master_data (re-reads raw CSV) | DataFrame → UC table |
| 16 | `lead_bank_designation` — Stage 8 (reporting) | `lead_bank_designation` | universe_enriched_results | DataFrame → UC table |
| 17 | `pipeline_outputs` (reporting) | `pipeline_outputs` | universe_enriched_results | CSV/summary files in `output_dir` |

## Notes / follow-ups

- The original `resources.py` default filenames (`ucc.csv`, `Master Mapping.csv`,
  `Universe.csv`) differ from the shipped sample CSVs (`raw.csv`,
  `master_list.csv`, `tracked_banks.csv`); the bundle passes the sample names via
  `--conf`. Adjust for your production inputs.
- For very large inputs, the in-process single-task job is still appropriate —
  the heavy lifting (FAISS, sentence-transformers) is single-node CPU work, not
  Spark. Size the driver accordingly (`num_workers: 0`, large driver).
- To materialize *more* intermediates as Delta tables, add `publish_table=...`
  to the relevant `Stage` in `registry.py`.
