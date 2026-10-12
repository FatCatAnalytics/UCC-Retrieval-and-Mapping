"""Fast, model-free tests for the engine plumbing (the pipeline itself is exercised by `ucc-engine run`)."""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]


def test_core_is_in_sync_with_dagster_assets():
    """core/ must be regenerated whenever assets.py changes."""
    r = subprocess.run([sys.executable, str(ENGINE / "scripts" / "sync_core.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_core_imports_without_dagster_or_torch():
    import ucc_engine.core.pipeline_logic as L

    src = Path(L.__file__).read_text()
    assert "from dagster" not in src and "sentence_transformers" not in src
    assert callable(L.fuzzy_match_results) and callable(L.updated_master_data)


def test_config_mirrors_dagster_fields_and_coerces_overrides():
    from ucc_engine.config import PipelineConfig

    cfg = PipelineConfig.from_overrides({"fuzzy_auto_threshold": "95", "embed_auto_threshold": "0.9",
                                         "advanced_cross_encoder_enabled": "false", "unknown_key": 1, "cpu_threads": ""})
    assert cfg.fuzzy_auto_threshold == 95 and cfg.embed_auto_threshold == 0.9
    assert cfg.advanced_cross_encoder_enabled is False
    assert cfg.cpu_threads == -1  # blank override ignored -> default
    assert not hasattr(cfg, "unknown_key")
    names = {f["name"] for f in PipelineConfig.schema()}
    # Every attribute the core *reads* (code, not docstrings) must exist on the engine config.
    import ast

    import ucc_engine.core.pipeline_logic as L

    tree = ast.parse(Path(L.__file__).read_text())
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config"
    }
    assert used <= names, f"core reads config fields missing from PipelineConfig: {sorted(used - names)}"


def test_shared_model_survives_dataframe_copy():
    """pandas deep-copies df.attrs; the core stores the model there."""
    from ucc_engine.embeddings import _SharedModel

    class Fake(_SharedModel):
        pass

    m = Fake()
    df = pd.DataFrame({"a": [1]})
    df.attrs["embedding_model"] = m
    assert df.copy().attrs["embedding_model"] is m
    assert copy.deepcopy({"model": m})["model"] is m


def test_compat_context_forwards_logs():
    from ucc_engine.compat import AssetExecutionContext, MetadataValue, Output

    seen = []
    ctx = AssetExecutionContext("stage", sink=lambda lvl, msg: seen.append((lvl, msg)))
    ctx.log.info("hello %s", "world")
    ctx.log.warning("careful")
    assert seen == [("INFO", "hello world"), ("WARNING", "careful")]
    assert Output(value=42, metadata={"x": MetadataValue.int(1)}) == 42


def test_csv_datasource_validates_columns(tmp_path: Path):
    from ucc_engine.datasources import CsvSourceSpec, DataSourceError, make_datasource

    good = tmp_path / "ucc.csv"
    good.write_text("DEBT_STATE,SECP_STATE,SECP_NAME\nCA,NY,ACME BANK\n")
    master = tmp_path / "master.csv"
    master.write_text('DEBT_STATE,SECP_STATE,SECP_NAME,SECP_NAME_MAPPED,Firm ID\nCA,NY,ACME BANK,Acme Bank,1\n')
    bad = tmp_path / "bad.csv"
    bad.write_text("Bank Name,Firm ID\nAcme,1\n")

    ds = make_datasource(CsvSourceSpec(ucc_path=str(good), master_path=str(bad)))
    desc = ds.describe()
    assert desc["inputs"]["ucc"]["missing_columns"] == []
    assert set(desc["inputs"]["master"]["missing_columns"]) == {"DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED"}
    assert desc["inputs"]["universe"] == {"present": False}

    ds = make_datasource({"type": "csv", "ucc_path": str(good), "master_path": str(master)})
    mat = ds.materialize(tmp_path / "run", sample_rows=0)
    assert mat.ucc_file == str(good) and mat.universe_file == "" and mat.rows == {"ucc": 1, "master": 1, "universe": 0}

    with pytest.raises(DataSourceError):
        make_datasource(CsvSourceSpec(ucc_path=str(tmp_path / "nope.csv"), master_path=str(master))).materialize(tmp_path)


def test_sql_spec_redacts_password():
    from ucc_engine.datasources import SqlSourceSpec, public_spec

    spec = SqlSourceSpec(url="postgresql://user:s3cret@db.local/ucc", ucc_query="select 1", master_query="select 1")
    assert "s3cret" not in public_spec(spec)["url"]


def test_registry_is_topologically_ordered_and_matches_core_signatures():
    import inspect

    import ucc_engine.core.pipeline_logic as L
    from ucc_engine.registry import PIPELINE

    seen: set[str] = set()
    for st in PIPELINE:
        assert set(st.deps) <= seen, f"{st.name} depends on {set(st.deps) - seen} before they run"
        params = [p for p in inspect.signature(getattr(L, st.name)).parameters if p != "context"]
        expected = (["config"] if st.needs_config else []) + st.deps
        assert params == expected, f"{st.name}: core wants {params}, registry passes {expected}"
        seen.add(st.name)


def test_clustering_stages_are_skipped_by_default_but_keep_the_dag_wired():
    import numpy as np

    from ucc_engine.config import PipelineConfig
    from ucc_engine.registry import CLUSTER_COLUMNS, STAGES_BY_NAME, is_enabled, run_stage, total_weight

    cfg = PipelineConfig()
    assert cfg.clustering_enabled is False
    clustering = [STAGES_BY_NAME[n] for n in ("clustered_results", "merged_cluster_results", "suggested_mapping_results")]
    assert all(not is_enabled(s, cfg) for s in clustering)
    assert all(is_enabled(s, PipelineConfig(clustering_enabled=True)) for s in clustering)
    assert total_weight(cfg) < total_weight(PipelineConfig(clustering_enabled=True))

    # Frame as it leaves Stage 4: one matched row, one pending, one NO_MASTER_STATE_PAIR.
    df = pd.DataFrame({
        "SECP_NAME": ["A", "B", "C"], "SECP_NAME_NORMALIZED": ["a", "b", "c"],
        "SECP_NAME_MAPPED": ["Acme", None, None], "confidence": ["AUTO", None, "NO_MASTER_STATE_PAIR"],
    })
    df.attrs["embedding_model"] = object()
    cache = {"advanced_embedding_results": df, "master_embeddings_cache": {}, "normalized_master_data": {}}
    for s in clustering:
        out = run_stage(s, cfg, None, cache)
    assert out is cache["suggested_mapping_results"]
    assert out["confidence"].tolist() == ["AUTO", "NEW_ENTITY", "NO_MASTER_STATE_PAIR"]
    assert set(CLUSTER_COLUMNS) <= set(out.columns) and out["cluster_id"].isna().all()
    assert np.isnan(out["cluster_size"].mean())  # what pipeline_outputs computes for the summary file
    assert out.attrs == {} and df.attrs  # input untouched, output carries no model handle


def test_api_health_and_validation(tmp_path: Path):
    from fastapi.testclient import TestClient

    from ucc_engine.api import create_app
    from ucc_engine.runner import RunStore

    client = TestClient(create_app(RunStore(tmp_path / "runs")))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/config/defaults").json()["defaults"]["fuzzy_auto_threshold"] == 96
    assert client.get("/runs/nope").status_code == 404
    r = client.post("/runs", json={"source": {"type": "csv", "ucc_path": "/nope.csv", "master_path": "/nope2.csv"}, "sample_rows": -1})
    assert r.status_code == 422  # sample_rows must be >= 0
