#!/usr/bin/env python3
"""Regenerate ``ucc_engine/core`` from the Dagster project.

The matching / clustering logic lives in exactly one place:
``ucc_mapping_project/ucc_mapping_project/assets.py``. This script copies it
(plus its pure-Python helpers) into the engine and applies a small, mechanical
set of rewrites so the code runs without Dagster or sentence-transformers:

  * ``from dagster import ...``            -> ``from ..compat import ...``
  * ``from .resources import PipelineConfig`` -> ``from ..config import PipelineConfig``
  * ``from .cross_encoder_rerank import`` -> ``from ..rerank import``
  * each ``SentenceTransformer(...)`` load block -> ``load_embedding_model(config, context)``
  * cache key ``model_id=str(config.embedding_model)`` -> the backend's ``model_id``
  * ``try_load_reranker(ce_model_name)`` -> ``try_load_reranker(ce_model_name, config=config)``

These are the same substitutions the Databricks port made by hand; here they are
automated so the engine never drifts from the Dagster logic. Run it whenever
``assets.py`` changes:

    python engine/scripts/sync_core.py          # rewrite
    python engine/scripts/sync_core.py --check  # exit 1 if core/ is stale (CI)
"""
from __future__ import annotations

import argparse
import difflib
import py_compile
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SRC = REPO / "ucc_mapping_project" / "ucc_mapping_project"
DST = REPO / "engine" / "src" / "ucc_engine" / "core"

# Files copied verbatim (they only import from each other / third parties).
VERBATIM = [
    "utils.py",
    "constants.py",
    "embedding_cache.py",
    "models/__init__.py",
    "models/data_structures.py",
]

HEADER = (
    "# GENERATED FILE - DO NOT EDIT.\n"
    "# Source: ucc_mapping_project/ucc_mapping_project/assets.py\n"
    "# Regenerate with: python engine/scripts/sync_core.py\n"
)

_ST_IMPORT = re.compile(r"^(?P<i>[ \t]*)from sentence_transformers import SentenceTransformer[ \t]*$", re.M)
_ST_LOAD = re.compile(
    r"^(?P<i>[ \t]*)try:\n"
    r"(?P=i)[ \t]+model = SentenceTransformer\(config\.embedding_model, device='cpu'\)\n"
    r"(?P=i)except Exception:\n"
    r"(?P=i)[ \t]+model = SentenceTransformer\('BAAI/bge-small-en-v1\.5', device='cpu'\)\n",
    re.M,
)


def transform_assets(text: str) -> str:
    out = text

    out = out.replace(
        "from dagster import asset, AssetExecutionContext, Output, MetadataValue, MaterializeResult",
        "from ..compat import asset, AssetExecutionContext, Output, MetadataValue, MaterializeResult",
    )
    out = out.replace("from .resources import PipelineConfig", "from ..config import PipelineConfig")
    out = out.replace("from .cross_encoder_rerank import", "from ..rerank import")

    n_imports = len(_ST_IMPORT.findall(out))
    out = _ST_IMPORT.sub(lambda m: f"{m.group('i')}from ..embeddings import load_embedding_model", out)
    n_loads = len(_ST_LOAD.findall(out))
    out = _ST_LOAD.sub(lambda m: f"{m.group('i')}model = load_embedding_model(config, context)\n", out)
    if n_imports != n_loads or n_imports == 0:
        raise SystemExit(
            f"sync_core: expected matching SentenceTransformer import/load blocks, "
            f"got imports={n_imports} loads={n_loads}. assets.py changed shape; update the regexes."
        )

    out = out.replace(
        "model_id=str(config.embedding_model),",
        "model_id=str(getattr(model, 'model_id', config.embedding_model)),",
    )
    # Let the reranker see model_cache_dir / model_dir / backend preference.
    if out.count("try_load_reranker(ce_model_name)") != 1:
        raise SystemExit("sync_core: expected exactly one try_load_reranker(ce_model_name) call")
    out = out.replace("try_load_reranker(ce_model_name)", "try_load_reranker(ce_model_name, config=config)")

    # Keep return annotations lazy (shimmed types are plain functions/classes).
    first_import = re.search(r"^(import |from )", out, re.M)
    assert first_import, "assets.py has no imports?"
    out = out[: first_import.start()] + "from __future__ import annotations\n" + out[first_import.start():]

    for forbidden in ("from dagster", "import dagster", "sentence_transformers"):
        if forbidden in out:
            raise SystemExit(f"sync_core: '{forbidden}' still present after transform")
    return HEADER + out


def build() -> dict[Path, str]:
    files: dict[Path, str] = {}
    files[DST / "pipeline_logic.py"] = transform_assets((SRC / "assets.py").read_text())
    for rel in VERBATIM:
        src = SRC / rel
        body = src.read_text() if src.exists() else ""
        files[DST / rel] = (
            f"# GENERATED FILE - DO NOT EDIT. Source: ucc_mapping_project/ucc_mapping_project/{rel}\n" + body
        )
    files[DST / "__init__.py"] = (
        '"""Generated copy of the Dagster matching logic. See engine/scripts/sync_core.py."""\n'
    )
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify core/ is up to date; do not write")
    args = ap.parse_args()

    files = build()
    stale: list[Path] = []
    for path, content in files.items():
        current = path.read_text() if path.exists() else None
        if current != content:
            stale.append(path)
            if args.check:
                if current is not None:
                    sys.stdout.writelines(
                        difflib.unified_diff(
                            current.splitlines(True), content.splitlines(True),
                            str(path), str(path) + " (expected)", n=1,
                        )
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

    if args.check:
        if stale:
            print(f"sync_core: {len(stale)} stale file(s): " + ", ".join(p.name for p in stale))
            return 1
        print("sync_core: core/ is up to date")
        return 0

    for path in files:
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
    print(f"sync_core: wrote {len(stale)} file(s) into {DST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
