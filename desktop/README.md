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
   match-method and confidence breakdowns, stage timings), the **review queue**
   where you approve, correct-then-approve, or reject each suggestion (approved
   rows are appended to the updated master with **Add N approved to updated
   master**; decisions persist per run), new-entity clusters when clustering is
   on, the output files, and **Promote as new master** (applies pending approvals,
   then copies the run's `updated_master_*.csv` over your Master Mapping file,
   keeping a timestamped backup).

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

Installers are built *on the target OS* - the Python engine is frozen with
PyInstaller, which does not cross-compile.

### Windows (NSIS `.exe`) - CI

`.github/workflows/windows-installer.yml` runs on `windows-latest`: installs
the engine, runs its tests, freezes the sidecar, builds the app with
`--bundles nsis`, and uploads **`UCC Mapping_<version>_x64-setup.exe`** as the
`UCC-Mapping-windows-installer` artifact. It triggers on pushes touching
`engine/` or `desktop/`, on *Run workflow* in the Actions tab, and on `v*` tags
(which also attach the installer to a GitHub Release).

The installer is per-user (`installMode: currentUser`, no admin rights
needed) and embeds the WebView2 bootstrapper (Windows 10/11 already ship
WebView2, so it normally does nothing). It is unsigned, so SmartScreen shows
"Windows protected your PC" → *More info* → *Run anyway* the first time; add a
code-signing certificate under `bundle.windows.certificateThumbprint` to remove that.

### Windows - locally

```powershell
cd engine
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[build,sql,dev]"
.venv\Scripts\python.exe scripts\build_sidecar.py   # -> desktop\src-tauri\binaries\ucc-engine-x86_64-pc-windows-msvc.exe
cd ..\desktop
npm ci
npm run app:build -- --bundles nsis                 # -> src-tauri\target\release\bundle\nsis\*.exe
```

Prerequisites: Rust (MSVC toolchain), Node 20+, Python 3.11 + uv, and the
Visual Studio C++ Build Tools (Tauri's Windows prerequisites).

### macOS

```bash
(cd ../engine && .venv/bin/python scripts/build_sidecar.py)   # -> src-tauri/binaries/ucc-engine-<triple>
npm run app:build                                             # -> src-tauri/target/release/bundle/{macos,dmg}
```

### Notes

* The frozen engine is a ~80 MB one-file executable that unpacks to a temp
  folder on launch (first start ~15-30 s; the UI shows "starting"). On quit the
  app closes the engine's stdin and it exits by itself so that temp folder is
  cleaned up; it is only force-killed if it does not stop within 3 s.
* Model weights are not in the installer. They download on first use into the
  user's app-data folder, or point `model_dir` (advanced settings) at a
  pre-filled folder or network share - see "Where the models live" in
  `../engine/README.md`.

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
