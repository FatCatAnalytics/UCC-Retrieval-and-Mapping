# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Stage HuggingFace embedding model
# MAGIC
# MAGIC One-time (or rare) task: download `BAAI/bge-small-en-v1.5` and stage
# MAGIC it onto the audit volume so subsequent tasks load from the volume
# MAGIC (faster + no network egress per run).
# MAGIC
# MAGIC **Job parameters**
# MAGIC - `hf_model_id` (default `BAAI/bge-small-en-v1.5`)
# MAGIC - `model_dir` — destination directory under
# MAGIC   `/Volumes/ucc/audit/artifacts/models/...`
# MAGIC
# MAGIC Run only when the model is missing or being upgraded.

# COMMAND ----------

# MAGIC %pip install --quiet huggingface_hub sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Make the shared/ helpers importable when this notebook is run from a
# Databricks Repo. The repo layout expected:
#   <repo_root>/databricks/tasks/00_setup_hugging_face_model.py
#   <repo_root>/databricks/shared/...
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath("__file__")) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from databricks.shared.config import load_config

cfg = load_config(dbutils)
print(f"Pipeline run: {cfg.pipeline_run_id}")
print(f"Target model dir: {cfg.model_dir}")
print(f"HF model id:      {cfg.hf_model_id}")

# COMMAND ----------

import os
from huggingface_hub import snapshot_download

os.makedirs(cfg.model_dir, exist_ok=True)

local_path = snapshot_download(
    repo_id=cfg.hf_model_id,
    local_dir=cfg.model_dir,
    local_dir_use_symlinks=False,  # Volumes do not support symlinks reliably
)
print(f"Model staged at: {local_path}")

# COMMAND ----------

# Smoke test the model loads from the volume
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(cfg.model_dir, device="cpu")
emb = model.encode(["JPMorgan Chase Bank", "JP MORGAN CHASE BK"], normalize_embeddings=True)
print("Embedding shape:", emb.shape)
print("Cosine sim:", float((emb[0] * emb[1]).sum()))

# COMMAND ----------

dbutils.notebook.exit(cfg.model_dir)
