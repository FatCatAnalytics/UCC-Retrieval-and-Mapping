"""Local HTTP API consumed by the Tauri desktop app (binds to 127.0.0.1 only)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PipelineConfig, app_data_dir
from .datasources import CsvSourceSpec, DataSourceError, SqlSourceSpec, SourceSpec, make_datasource
from .embeddings import available_backends, prefetch_models
from .runner import REVIEW_BANDS, RunStore


class RunRequest(BaseModel):
    source: SourceSpec = Field(..., discriminator="type")
    config: Dict[str, Any] = Field(default_factory=dict, description="PipelineConfig overrides")
    sample_rows: int = Field(0, ge=0, description="0 = all UCC rows; N = random sample for a quick run")
    name: str = ""


class PromoteRequest(BaseModel):
    destination: str = Field(..., description="Path the updated master should overwrite (a backup is kept)")


class PreviewRequest(BaseModel):
    source: SourceSpec = Field(..., discriminator="type")


class DecisionItem(BaseModel):
    key: str = Field(..., description="`key` of a review-queue row")
    decision: Optional[Literal["approved", "rejected"]] = Field(None, description="null clears the decision")
    mapped_name: Optional[str] = Field(None, description="override the suggested SECP_NAME_MAPPED")
    firm_id: Optional[Any] = Field(None, description="override the Firm ID (default: Universe lookup by mapped name)")


class DecisionsRequest(BaseModel):
    items: List[DecisionItem] = Field(..., min_length=1)


def create_app(store: Optional[RunStore] = None) -> FastAPI:
    store = store or RunStore()
    app = FastAPI(title="UCC Mapping Engine", version=__version__)
    app.add_middleware(  # Tauri webview origins (tauri://localhost, http://tauri.localhost, Vite dev)
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ---------------------------------------------------------------- meta
    @app.get("/health")
    def health() -> Dict[str, Any]:
        active = store.active()
        return {
            "status": "ok",
            "version": __version__,
            "pid": os.getpid(),
            "data_dir": str(app_data_dir()),
            "backends": available_backends(),
            "active_run": active.id if active else None,
        }

    @app.get("/config/defaults")
    def config_defaults() -> Dict[str, Any]:
        return {"defaults": PipelineConfig().to_dict(), "schema": PipelineConfig.schema()}

    @app.post("/models/prefetch")
    def models_prefetch(overrides: Dict[str, Any] | None = None) -> Dict[str, str]:
        try:
            return prefetch_models(PipelineConfig.from_overrides(overrides or {}))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Model prefetch failed: {e}") from e

    # ------------------------------------------------------------- sources
    @app.post("/sources/preview")
    def sources_preview(req: PreviewRequest) -> Dict[str, Any]:
        try:
            return make_datasource(req.source).describe()
        except DataSourceError as e:
            raise HTTPException(400, str(e)) from e

    # ---------------------------------------------------------------- runs
    @app.get("/runs")
    def runs_list() -> List[Dict[str, Any]]:
        return store.list()

    @app.post("/runs", status_code=201)
    def runs_create(req: RunRequest) -> Dict[str, Any]:
        if store.active() is not None:
            raise HTTPException(409, "A run is already in progress")
        cfg = PipelineConfig.from_overrides(req.config)
        run = store.create(req.source, cfg, sample_rows=req.sample_rows, name=req.name).start()
        return run.snapshot()

    def _get(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, f"Unknown run {run_id}")
        return run

    @app.get("/runs/{run_id}")
    def runs_get(run_id: str) -> Dict[str, Any]:
        return _get(run_id).snapshot()

    @app.delete("/runs/{run_id}")
    def runs_delete(run_id: str) -> Dict[str, bool]:
        _get(run_id)
        if not store.delete(run_id):
            raise HTTPException(409, "Cannot delete a run that is in progress")
        return {"deleted": True}

    @app.post("/runs/{run_id}/cancel")
    def runs_cancel(run_id: str) -> Dict[str, Any]:
        run = _get(run_id)
        run.cancel()
        return run.snapshot()

    @app.get("/runs/{run_id}/logs")
    def runs_logs(run_id: str, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
        run = _get(run_id)
        lines = run.logs(after=after, limit=limit)
        return {"lines": lines, "last_seq": lines[-1]["seq"] if lines else after, "status": run.status}

    @app.get("/runs/{run_id}/summary")
    def runs_summary(run_id: str) -> Dict[str, Any]:
        run = _get(run_id)
        if run.summary is None and run.status == "succeeded":
            run.summary = run._build_summary()
        return run.summary or {}

    @app.get("/runs/{run_id}/review")
    def runs_review(
        run_id: str,
        confidence: Optional[List[str]] = Query(None, description=f"any of {list(REVIEW_BANDS)}"),
        search: str = "",
        method: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        sort: str = "match_score",
        descending: bool = True,
        decision: Literal["", "pending", "approved", "rejected"] = "",
    ) -> Dict[str, Any]:
        return _get(run_id).review(confidence=confidence, search=search, method=method, offset=offset,
                                   limit=limit, sort=sort, descending=descending, decision=decision)

    # ----------------------------------------------------- review decisions
    @app.get("/runs/{run_id}/decisions")
    def runs_decisions(run_id: str) -> Dict[str, Any]:
        run = _get(run_id)
        return {"counts": run.decision_counts(), "decisions": run.decisions}

    @app.put("/runs/{run_id}/decisions")
    def runs_set_decisions(run_id: str, req: DecisionsRequest) -> Dict[str, Any]:
        try:
            return _get(run_id).set_decisions([i.model_dump() for i in req.items])
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (ValueError, RuntimeError) as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/runs/{run_id}/apply-decisions")
    def runs_apply_decisions(run_id: str) -> Dict[str, Any]:
        try:
            return _get(run_id).apply_decisions()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/runs/{run_id}/clusters")
    def runs_clusters(
        run_id: str, search: str = "", min_size: int = Query(1, ge=1),
        offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return _get(run_id).clusters(search=search, min_size=min_size, offset=offset, limit=limit)

    @app.get("/runs/{run_id}/outputs")
    def runs_outputs(run_id: str) -> List[Dict[str, Any]]:
        return _get(run_id).outputs()

    @app.get("/runs/{run_id}/outputs/{name}")
    def runs_output_file(run_id: str, name: str) -> FileResponse:
        run = _get(run_id)
        path = (run.output_dir / Path(name).name).resolve()
        if run.output_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "No such output file")
        return FileResponse(path, filename=path.name)

    @app.post("/runs/{run_id}/promote-master")
    def runs_promote(run_id: str, req: PromoteRequest) -> Dict[str, Any]:
        try:
            return _get(run_id).promote_master(req.destination)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    return app


# `uvicorn ucc_engine.api:app`
app = create_app()
