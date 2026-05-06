# UCC Mapping — Databricks Phase 1

This folder contains task notebooks and shared helpers for running the
UCC monthly mapping pipeline on Databricks. The intent is that the
**code lives in this repo** and is imported into Databricks via
**Repos** (Workspace → Repos → Add Repo → connect this Git repo). No
copy-paste into the workspace UI.

## Layout

```
databricks/
├── README.md                     ← you are here
├── requirements-databricks.txt   ← libraries for serverless notebook installs
├── shared/                       ← reusable helpers (config, paths, audit, …)
│   ├── __init__.py
│   ├── config.py                 ← TaskConfig + widget plumbing
│   ├── paths.py                  ← Volume paths + UC table names
│   ├── audit.py                  ← pipeline_runs / source_files / changes log
│   ├── io.py                     ← CSV reads, Delta upserts, partition overwrite
│   ├── matching.py               ← thin shims over Dagster matching stages
│   └── validation.py             ← schema + light QC checks
└── tasks/                        ← runnable .py notebooks
    ├── 00_setup_hugging_face_model.py
    ├── 00_validate_inputs.py
    ├── 01_ingest_ucc_raw.py
    ├── 02_clean_ucc.py
    ├── 03_load_master_current.py
    ├── 04_exact_match.py
    ├── 05_fuzzy_embedding_match.py
    ├── 06_export_review_workbook.py
    ├── 08_ingest_review_workbook.py
    ├── 09_validate_review_decisions.py
    ├── 10_update_master_list.py
    ├── 11_publish_final_mapping.py
    └── 12_write_audit_summary.py
```

> Task `07` is intentionally not present — that step is the offline
> business review (operators fill out the Excel workbook produced by
> `06`). When the workbook returns, ingestion resumes at `08`.

## Volume layout (recommended)

```
/Volumes/ucc/bronze/landing/bootstrap/master_list.csv
/Volumes/ucc/bronze/landing/bootstrap/tracked_banks.csv
/Volumes/ucc/bronze/landing/{source_month}/source/ucc.csv
/Volumes/ucc/bronze/landing/{source_month}/review_return/completed_review.xlsx
/Volumes/ucc/gold/exports/{source_month}/review_out/review_workbook_{source_month}.xlsx
/Volumes/ucc/gold/exports/{source_month}/final/...
/Volumes/ucc/audit/artifacts/models/bge-small-en-v1.5/
```

`source_month` is `YYYY-MM` (for example `2026-04`). Operators upload
the source CSV to the month-stamped `source/` folder; the pipeline
writes back to `review_out/` and `final/`.

## Tables

| Layer  | Table                                        | Partition / key                  |
|--------|----------------------------------------------|----------------------------------|
| bronze | `ucc.bronze.ucc_raw`                         | `source_month`                   |
| bronze | `ucc.bronze.review_workbook_raw`             | `source_month, review_run_id`    |
| silver | `ucc.silver.ucc_clean`                       | `source_month`                   |
| silver | `ucc.silver.master_list_current`             | (overwrite)                      |
| silver | `ucc.silver.master_list_history`             | append-only                      |
| silver | `ucc.silver.review_decisions`                | `source_month, review_run_id`    |
| silver | `ucc.silver.review_decision_errors`          | `source_month, review_run_id`    |
| gold   | `ucc.gold.exact_mapped`                      | `source_month`                   |
| gold   | `ucc.gold.review_candidates`                 | `source_month`                   |
| gold   | `ucc.gold.final_ucc_mapping`                 | `source_month`                   |
| gold   | `ucc.gold.monthly_summary`                   | `source_month`                   |
| audit  | `ucc.audit.pipeline_runs`                    | append, `pipeline_run_id`        |
| audit  | `ucc.audit.source_files`                     | append, `pipeline_run_id`        |
| audit  | `ucc.audit.review_workbooks`                 | append, `review_run_id`          |
| audit  | `ucc.audit.master_list_changes`              | append, `review_run_id`          |

The notebooks `CREATE SCHEMA IF NOT EXISTS` and create the audit
tables on first use, so a clean workspace can run end-to-end without
separate DDL scripts.

## Job parameters / widgets

Every task notebook auto-declares the same widgets via
`databricks.shared.config.declare_widgets`. When defining a Databricks
Job, set these as **job-level parameters** so all tasks share them:

| Widget            | Required | Example                            |
|-------------------|----------|------------------------------------|
| `source_month`    | yes      | `2026-04`                          |
| `catalog`         | no       | `ucc` (default)                    |
| `bronze_schema`   | no       | `bronze`                           |
| `silver_schema`   | no       | `silver`                           |
| `gold_schema`     | no       | `gold`                             |
| `audit_schema`    | no       | `audit`                            |
| `landing_root`    | no       | `/Volumes/ucc/bronze/landing`      |
| `gold_export_root`| no       | `/Volumes/ucc/gold/exports`        |
| `model_dir`       | no       | `/Volumes/ucc/audit/artifacts/models/bge-small-en-v1.5` |
| `hf_model_id`     | no       | `BAAI/bge-small-en-v1.5`           |
| `review_run_id`   | only for tasks 09-12 | output of task 08 (e.g. `review_20260507T140212Z`) |

For tasks `08` → `09` → `10` → `11` → `12`, wire the job so task 08's
`dbutils.notebook.exit` value (the new `review_run_id`) is passed to
the next tasks via `{{tasks.<task_key>.values.review_run_id}}`.

## Task ordering

### Phase 1a — initial pipeline (matching → review export)

```
00_setup_hugging_face_model       (one-time / when model changes)
00_validate_inputs
01_ingest_ucc_raw
02_clean_ucc
03_load_master_current            (force_bootstrap=true on first month only)
04_exact_match
05_fuzzy_embedding_match
06_export_review_workbook
```

After 06, business reviewers download
`review_workbook_{source_month}.xlsx`, fill in `decision`,
`decided_mapping`, and `decided_firm_id`, and upload it to
`/Volumes/ucc/bronze/landing/{source_month}/review_return/completed_review.xlsx`.

### Phase 1b — review-return processing

```
08_ingest_review_workbook         (emits review_run_id)
09_validate_review_decisions
10_update_master_list
11_publish_final_mapping
12_write_audit_summary
```

## How the matching tasks reuse the existing repo

Tasks `04` and `05` import the original Dagster asset functions from
`ucc_mapping_project.ucc_mapping_project.assets` and invoke them with
a small `_ContextShim` (in `databricks/shared/matching.py`). This
keeps Dagster and Databricks on the **same code path** for
normalization, fuzzy, embedding, clustering, and Firm_ID enrichment.

The shim does have to write the in-memory `master_list_current` and
tracked-banks data to a tmp CSV before calling the Dagster loaders
(because those loaders take file paths, not DataFrames). This is
intentional for Phase 1; in Phase 2 we'd refactor the loaders to
accept DataFrames directly.

### Importing from a Databricks Repo

When this repo is added under Workspace → Repos, each notebook adds
`<repo_root>` to `sys.path` so:

```python
from databricks.shared.config import load_config
from ucc_mapping_project.ucc_mapping_project.utils import normalize_name
```

both work without any extra config. If you move the repo under a
nested workspace folder, update the `_REPO_ROOT` calculation at the
top of each task — it currently assumes the path
`<repo_root>/databricks/tasks/<task>.py`.

## Running locally for development

For quick local sanity checks (no Databricks required):

```bash
cd ucc_mapping_project
pip install -e ".[dev]"
pip install -r ../databricks/requirements-databricks.txt
python -c "from databricks.shared import paths, config; print(config.WIDGET_DEFS)"
```

The notebooks themselves require a Spark/dbutils runtime, but the
`shared/` modules are plain Python and unit-testable.

## Outstanding TODOs / Phase-2 cleanup

- Loaders in `assets.py` still take file paths; refactor them to
  accept DataFrames so we can avoid the tmp-CSV detour in tasks 04/05.
- `09_validate_review_decisions` uses a deliberately lenient schema —
  tighten once the operator workbook format is locked in.
- Add a `gh actions` smoke-test that runs `02_clean_ucc` on a 100-row
  fixture against a local Spark session.
- Wire the job's `review_run_id` plumbing in a `databricks.yml` /
  bundle so the job graph is reproducible from this repo.
