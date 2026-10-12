# Databricks Job Runbook

This branch is structured to run as a Databricks Asset Bundle job from the `databricks/` folder.

## Recommended job

Use `ucc_mapping_pipeline_single` first. It runs the full DAG in one Python wheel task and keeps intermediate objects in memory, which is faster and avoids serialising FAISS/numpy-heavy objects between tasks.

Use `ucc_mapping_pipeline_staged` only when you specifically want one Databricks task per pipeline stage for observability/repair.

## Required workspace objects

Create or confirm:

- Unity Catalog catalog, default: `main`
- Unity Catalog schema, default: `ucc`
- Unity Catalog volume/root path, default: `/Volumes/main/ucc/data`
- Input files under `${data_root}/Inputs/`:
  - `raw.csv`
  - `master_list.csv`
  - `tracked_banks.csv`
- Output directory: `${data_root}/Outputs`
- Foundation Model API endpoint access for the run identity, default endpoint: `databricks-bge-large-en-v1.5`

## Deploy

```bash
cd databricks
python -m pip install --upgrade build databricks-cli

databricks bundle validate -t dev \
  --var catalog=main \
  --var schema=ucc \
  --var data_root=/Volumes/main/ucc/data

databricks bundle deploy -t dev \
  --var catalog=main \
  --var schema=ucc \
  --var data_root=/Volumes/main/ucc/data
```

## Run

```bash
databricks bundle run ucc_mapping_pipeline_single -t dev
```

To override paths at run time, use the job parameters in the Databricks Jobs UI or pass them when triggering the job through the API/CLI:

- `raw_file`
- `master_file`
- `universe_file`
- `output_dir`
- `embedding_backend`
- `embedding_endpoint`

## Production notes

- The staged DAG uses a run-specific staging directory: `${data_root}/_staging/{{job.run_id}}`. This prevents stale pickle files from previous runs being reused during retries or repairs.
- Published Unity Catalog Delta tables are controlled in `src/ucc_pipeline/registry.py` via `publish_table`.
- Default published tables are:
  - `universe_enriched_results`
  - `updated_master_data`
  - `merged_raw_with_firm_id`
  - `lead_bank_designation`
- If the Foundation Model API endpoint is unavailable, either grant the job run identity access or set `embedding_backend=local` and make sure the required local model dependencies are available on the cluster.

## Smoke test

After deployment, run with a small sample input first and check:

1. Job run completes successfully.
2. Files are written to `${data_root}/Outputs`.
3. Delta tables exist under `${catalog}.${schema}`.
4. The latest `updated_master_*` and `merged_raw_with_firm_id_*` files have the expected row counts.
