# UCC Mapping - desktop app

Tauri 2 shell (Rust) + React/TypeScript UI around the Python engine in
`../engine`. The app starts the engine as a sidecar, talks to it over
`http://127.0.0.1:<port>`, and shows:

1. **Setup** - pick the UCC filings, Master Mapping and Universe CSVs (or enter
   a SQL connection + three queries), check the inputs, optionally run on a
   random sample, tweak thresholds.
2. **Run** - the 17 stages with live status/timings and the streamed log;
   cancel after the current stage.
3. **Results** - summary (auto / review / new entities / Firm-ID coverage,
   match-method and confidence breakdowns, stage timings), the review queue
   and new-entity clusters as searchable tables, the output files, and
   **Promote as new master** (copies the run's `updated_master_*.csv` over your
   Master Mapping file, keeping a timestamped backup).

No LLM and no network inference: the two small models run on the local CPU.

## Prerequisites

* Rust (stable), Node 20+, and the engine environment (`../engine/README.md`).
* macOS: Xcode Command Line Tools. Linux/Windows: see the Tauri 2 prerequisites.

## Develop

```bash
# one-time: engine venv + dev sidecar stub (a shell script that runs engine/.venv/bin/ucc-engine)
(cd ../engine && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,sql,build]")
python3 ../engine/scripts/build_sidecar.py --dev-stub

npm install
npm run app:dev            # = tauri dev: Vite on :1420 + the window; engine spawned via the stub
```

Alternatives when iterating on the engine:

```bash
UCC_ENGINE_URL=http://127.0.0.1:8765 npm run app:dev      # connect to an engine you started yourself
UCC_ENGINE_CMD=../engine/.venv/bin/ucc-engine npm run app:dev   # spawn that executable instead of the sidecar
VITE_ENGINE_URL=http://127.0.0.1:8765 npm run dev          # UI only, in a normal browser (no native dialogs)
```

## Build an installer

```bash
(cd ../engine && .venv/bin/python scripts/build_sidecar.py)   # freezes the engine -> src-tauri/binaries/ucc-engine-<triple>
npm run app:build                                             # -> src-tauri/target/release/bundle/
```

The frozen engine is ~300 MB (Python + pandas + faiss + ONNX Runtime, no torch).
Model weights download on first use into the user's app-data folder; for
offline machines ship them alongside and set `model_dir` in the advanced settings.

## Layout

```
desktop/
├── src/                    React UI (App, views/Setup|Run|Results, api.ts client, types.ts)
├── src-tauri/
│   ├── src/lib.rs          sidecar lifecycle, port discovery, file dialogs, reveal/open
│   ├── tauri.conf.json     externalBin = binaries/ucc-engine
│   ├── capabilities/       webview permissions (core only)
│   └── binaries/           ucc-engine-<target-triple> (dev stub or frozen binary; git-ignored)
└── package.json
```
