# ucc-engine

Headless version of the UCC → master-mapping pipeline, with a local HTTP API.
It is the backend ("sidecar") of the desktop app in `../desktop`, and also a
plain CLI.

* **Same logic as the Dagster project.** `src/ucc_engine/core/` is *generated*
  from `ucc_mapping_project/.../assets.py` by `scripts/sync_core.py` (Dagster
  decorators and the `SentenceTransformer(...)` loads are swapped for shims).
  Never edit `core/` by hand - re-run the script when `assets.py` changes.
  `pytest` fails if `core/` is stale.
* **No Dagster, no network inference.** The two small models (the
  `bge-small-en-v1.5` embedder and the `ms-marco-MiniLM` cross-encoder) run on
  the local CPU through [`fastembed`](https://github.com/qdrant/fastembed)
  (ONNX Runtime). Install the `st` extra to use the original torch
  `sentence-transformers` models instead. There is no LLM anywhere.
* **Inputs are pluggable.** `CsvDataSource` today; `SqlDataSource` (SQLAlchemy,
  three queries) is implemented and waiting for a connection string.

## Setup

```bash
cd engine
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,sql,build]"
source .venv/bin/activate

ucc-engine prefetch-models            # one-time download into the local model cache
pytest                                # fast plumbing tests (no models needed)
```

## CLI

```bash
# whole pipeline on CSVs, streaming the log; add --sample 3000 for a quick run
ucc-engine run --ucc Inputs/ucc.csv --master "Inputs/Master Mapping.csv" \
               --universe Inputs/Universe.csv --sample 3000 --set fuzzy_auto_threshold=95

# local API for the desktop app (prints UCC_ENGINE_PORT=<n> first)
ucc-engine serve --port 8765
```

Runs live under the per-user app data dir (`~/Library/Application Support/ucc-engine/runs/<id>/`
on macOS; override with `UCC_ENGINE_HOME`). Each run has `inputs/` (only when
sampled or from SQL), `outputs/` (the same files the Dagster `pipeline_outputs`
asset writes, plus `updated_master_*.csv`, `merged_raw_with_firm_id_*.csv`,
lead-bank files) and `run.json` (status, timings, summary, log tail).

## HTTP API (127.0.0.1 only)

| Method & path | Purpose |
|---|---|
| `GET /health` | version, available backends, active run |
| `GET /config/defaults` | `PipelineConfig` defaults + schema for a settings form |
| `POST /models/prefetch` | download models now |
| `POST /sources/preview` | validate a CSV/SQL source: columns present/missing, row counts, 5-row preview |
| `GET/POST /runs` | list runs / start one `{source, config, sample_rows, name}` (409 if one is running) |
| `GET /runs/{id}` | status, progress (0-1), per-stage state, summary |
| `GET /runs/{id}/logs?after=N` | incremental log lines |
| `POST /runs/{id}/cancel` | stop after the current stage |
| `GET /runs/{id}/summary` | the numbers from the Dagster summary file, as JSON |
| `GET /runs/{id}/review` | paged review queue (`confidence`, `search`, `method`, `offset`, `limit`) |
| `GET /runs/{id}/clusters` | new-entity clusters, one row per cluster |
| `GET /runs/{id}/outputs[/{name}]` | list / download output files |
| `POST /runs/{id}/promote-master` | copy this run's `updated_master` over a destination (timestamped backup kept) |
| `DELETE /runs/{id}` | remove a finished run |

## Configuration

`config.PipelineConfig` mirrors the Dagster `resources.PipelineConfig`
field-for-field (thresholds, clustering, performance, lead-bank, rerank), plus:

| Field | Default | Meaning |
|---|---|---|
| `embedding_backend` | `auto` | `fastembed` if installed, else `sentence-transformers` |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | 384-dim; keep in sync with any persisted query cache |
| `model_dir` | `""` | directory of pre-downloaded model files for air-gapped machines (disables downloads) |
| `model_cache_dir` | `<app data>/models` | where models are downloaded |
| `model_threads` | `0` | ONNX / torch intra-op threads (0 = library default) |

## Packaging the sidecar

```bash
python scripts/build_sidecar.py --dev-stub   # dev: shell stub -> engine/.venv (for `tauri dev`)
python scripts/build_sidecar.py              # release: PyInstaller one-file binary ->
                                             #   ../desktop/src-tauri/binaries/ucc-engine-<target-triple>
```

Model weights are not inside the binary; they download on first use. To ship
them, copy the contents of `model_cache_dir` next to the app and set `model_dir`.
