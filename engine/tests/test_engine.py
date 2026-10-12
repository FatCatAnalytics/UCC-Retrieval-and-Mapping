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


def _fake_finished_run(tmp_path: Path):
    """A succeeded run with an in-memory results frame + updated master, no models involved."""
    from ucc_engine.config import PipelineConfig
    from ucc_engine.datasources import CsvSourceSpec
    from ucc_engine.runner import PipelineRun

    universe = tmp_path / "universe.csv"
    universe.write_text("Bank Name,Firm ID\nAcme Bank,101\nZeta Bank,202\n")
    spec = CsvSourceSpec(ucc_path=str(tmp_path / "u.csv"), master_path=str(tmp_path / "m.csv"), universe_path=str(universe))
    run = PipelineRun(spec, PipelineConfig(), runs_root=tmp_path / "runs", name="t")
    run.config.universe_file = str(universe)
    run.status = "succeeded"
    results = pd.DataFrame({
        "DEBT_STATE": ["ca", "NY", "TX", "FL"], "SECP_STATE": ["CA", "NY", "TX", "FL"],
        "SECP_NAME": ["ACME BK NA", "ZETA BANK TRUSTEE", "OTHER CO", "KNOWN"],
        "SECP_NAME_MAPPED": ["Acme Bank", "Zeta Bank", "Unknown Co", "Known"],
        "match_method": ["Advanced Match"] * 3 + ["Exact Match (Raw)"],
        "match_score": [0.91, 0.88, 0.86, 1.0],
        "confidence": ["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW", "AUTO"],
        "Firm_ID": [101, None, None, 5],
    })
    run._cache["universe_enriched_results"] = results
    master = pd.DataFrame({"DEBT_STATE": ["FL"], "SECP_STATE": ["FL"], "SECP_NAME": ["KNOWN"],
                           "SECP_NAME_MAPPED": ["Known"], "Firm ID": [5], "_source": ["original_master"]})
    run._cache["updated_master_data"] = master
    run.output_dir.mkdir(parents=True, exist_ok=True)
    master.drop(columns="_source").to_csv(run.output_dir / "updated_master_20260101_000000.csv", index=False)
    master.to_csv(run.output_dir / "updated_master_20260101_000000_with_source.csv", index=False)
    run.summary = run._build_summary()
    return run


def test_review_decisions_flow_into_updated_master(tmp_path: Path):
    from ucc_engine.runner import PipelineRun

    run = _fake_finished_run(tmp_path)
    page = run.review()
    assert page["total"] == 3 and all("key" in r and r["decision"] is None for r in page["rows"])
    k_acme, k_other, k_zeta = (r["key"] for r in sorted(page["rows"], key=lambda r: r["SECP_NAME"]))
    assert k_acme == "CA|CA|ACME BK NA"  # states normalised like the core

    # approve as suggested, approve with a corrected name (Firm ID re-resolved from the Universe), reject one
    out = run.set_decisions([
        {"key": k_acme, "decision": "approved"},
        {"key": k_other, "decision": "approved", "mapped_name": "Zeta Bank"},
        {"key": k_zeta, "decision": "rejected"},
    ])
    assert out["counts"] == {"approved": 2, "rejected": 1, "pending": 0, "applied": 0, "unapplied_approved": 2}
    assert run.decisions[k_other]["firm_id"] == 202 and run.decisions[k_acme]["firm_id"] == 101
    assert run.review(decision="approved")["total"] == 2 and run.review(decision="pending")["total"] == 0
    with pytest.raises(KeyError):
        run.set_decisions([{"key": "FL|FL|KNOWN", "decision": "approved"}])  # AUTO row is not reviewable
    with pytest.raises(ValueError):
        run.set_decisions([{"key": k_acme, "decision": "maybe"}])

    # promote applies pending approvals first; the promoted file contains exactly master + approved rows
    dest = tmp_path / "Master Mapping.csv"
    dest.write_text("x\n")
    res = run.promote_master(str(dest))
    promoted = pd.read_csv(dest)
    assert list(promoted.columns) == ["DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED", "Firm ID"]
    assert promoted["SECP_NAME"].tolist() == ["KNOWN", "ACME BK NA", "OTHER CO"]
    assert promoted.set_index("SECP_NAME").loc["OTHER CO", "SECP_NAME_MAPPED"] == "Zeta Bank"
    assert promoted["Firm ID"].tolist() == [5, 101, 202]
    assert res["backup"] and Path(res["backup"]).read_text() == "x\n"
    assert run.decision_counts()["unapplied_approved"] == 0 and run.summary["updated_master"]["approved_rows"] == 2

    # re-applying is idempotent (no duplicate approved rows), and decisions survive a restart
    run.apply_decisions()
    latest = pd.read_csv(run._latest("updated_master_*.csv", exclude="_with_source"))
    assert len(latest) == 3
    run._persist()
    restored = PipelineRun.load(run.run_dir)
    assert restored is not None and restored.decisions.keys() == run.decisions.keys()

    # clearing a decision
    run.set_decisions([{"key": k_zeta, "decision": None}])
    assert run.decision_counts()["pending"] == 1

    # a corrected name that is not in the Universe gets a blank Firm ID (not the suggestion's),
    # and the blank must not demote the column to floats ("101.0") in the CSV
    run.set_decisions([{"key": k_zeta, "decision": "approved", "mapped_name": "Unknown Bank"}])
    assert run.decisions[k_zeta]["firm_id"] is None
    run.apply_decisions()
    text = Path(run._latest("updated_master_*.csv", exclude="_with_source")).read_text()
    assert ",101\n" in text and ".0" not in text
    assert any(line.endswith(",Unknown Bank,") for line in text.splitlines())


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
