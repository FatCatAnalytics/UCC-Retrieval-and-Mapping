"""In-process pipeline runner with progress, log streaming, cancellation and results.

A ``PipelineRun`` executes the stages in ``registry.PIPELINE`` on a worker
thread. State is snapshot-able at any time (for the HTTP API) and persisted to
``<run_dir>/run.json`` so finished runs survive an engine restart.
"""
from __future__ import annotations

import gc
import json
import math
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .compat import AssetExecutionContext
from .config import PipelineConfig, app_data_dir
from .datasources import DataSource, SourceSpec, make_datasource, public_spec
from .registry import KEEP_AFTER_RUN, PIPELINE, TOTAL_WEIGHT, Stage, run_stage

REVIEW_BANDS = ("REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW")
REVIEW_COLUMNS = [
    "DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_NORMALIZED", "SECP_NAME_MAPPED",
    "matched_master_raw", "match_method", "match_score", "confidence",
    "ambiguous_master_candidates", "Firm_ID", "cluster_id", "cluster_label", "cluster_size",
    "suggested_mapping", "suggestion_score", "suggestion_confidence", "alternative_suggestions",
]
MAX_LOG_LINES = 5000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Make pandas/numpy scalars JSON-serialisable; NaN -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            v = value.item()
            return None if isinstance(v, float) and math.isnan(v) else v
        except Exception:  # noqa: BLE001
            return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def frame_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


@dataclass
class StageState:
    name: str
    label: str
    group: str
    status: str = "pending"  # pending | running | done | failed | skipped
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    seconds: Optional[float] = None
    rows: Optional[int] = None


@dataclass
class LogLine:
    seq: int
    ts: str
    level: str
    stage: str
    message: str


class RunCancelled(Exception):
    pass


class PipelineRun:
    def __init__(
        self,
        source_spec: SourceSpec,
        config: PipelineConfig,
        *,
        runs_root: Path,
        sample_rows: int = 0,
        run_id: Optional[str] = None,
        name: str = "",
    ):
        self.id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.name = name or self.id
        self.created_at = _now()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.status = "queued"  # queued | running | succeeded | failed | cancelled
        self.error: Optional[str] = None
        self.sample_rows = int(sample_rows or 0)

        self.run_dir = runs_root / self.id
        self.output_dir = self.run_dir / "outputs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source_spec = source_spec
        self.source: DataSource = make_datasource(source_spec)
        self.config = config
        self.config.output_dir = str(self.output_dir)
        self.input_rows: Dict[str, int] = {}

        self.stages: List[StageState] = [StageState(s.name, s.label, s.group) for s in PIPELINE]
        self.current_stage: Optional[str] = None
        self._logs: List[LogLine] = []
        self._seq = 0
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cache: Dict[str, Any] = {}
        self.summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ logging
    def _log(self, level: str, message: str, stage: str = "engine") -> None:
        with self._lock:
            self._seq += 1
            self._logs.append(LogLine(self._seq, _now(), level, stage, message))
            if len(self._logs) > MAX_LOG_LINES:
                del self._logs[: len(self._logs) - MAX_LOG_LINES]

    def logs(self, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(l) for l in self._logs if l.seq > after][:limit]

    # ------------------------------------------------------------------ control
    def start(self) -> "PipelineRun":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=f"ucc-run-{self.id}", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()
        self._log("WARNING", "Cancellation requested; the run stops after the current stage.")

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --------------------------------------------------------------------- run
    def _run(self) -> None:
        self.status, self.started_at = "running", _now()
        self._persist()
        try:
            self._log("INFO", f"Materialising inputs from {self.source.kind} source"
                              + (f" (sample {self.sample_rows:,} UCC rows)" if self.sample_rows else ""))
            inputs = self.source.materialize(self.run_dir, self.sample_rows)
            self.config.ucc_file = inputs.ucc_file
            self.config.master_file = inputs.master_file
            self.config.universe_file = inputs.universe_file or str(self.run_dir / "inputs" / "__no_universe__.csv")
            self.input_rows = inputs.rows
            self._log("INFO", "Inputs: " + ", ".join(f"{k}={v:,}" for k, v in inputs.rows.items()))

            for stage, state in zip(PIPELINE, self.stages):
                if self._cancel.is_set():
                    raise RunCancelled()
                self._exec_stage(stage, state)

            self.summary = self._build_summary()
            self.status = "succeeded"
            self._log("INFO", "Run complete.")
        except RunCancelled:
            self.status = "cancelled"
            for st in self.stages:
                if st.status == "pending":
                    st.status = "skipped"
        except Exception as e:  # noqa: BLE001
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
            self._log("ERROR", self.error)
            self._log("DEBUG", traceback.format_exc())
            for st in self.stages:
                if st.status == "running":
                    st.status = "failed"
                elif st.status == "pending":
                    st.status = "skipped"
        finally:
            self.finished_at = _now()
            self.current_stage = None
            self._release_memory()
            self._persist()

    def _exec_stage(self, stage: Stage, state: StageState) -> None:
        state.status, state.started_at = "running", _now()
        self.current_stage = stage.name
        t0 = time.perf_counter()
        ctx = AssetExecutionContext(stage.name, sink=lambda lvl, msg: self._log(lvl, msg, stage.name))
        self._log("INFO", f"▶ {stage.label}", stage.name)
        try:
            result = run_stage(stage, self.config, ctx, self._cache)
        except Exception:
            state.status, state.finished_at = "failed", _now()
            state.seconds = round(time.perf_counter() - t0, 2)
            raise
        state.seconds = round(time.perf_counter() - t0, 2)
        state.finished_at, state.status = _now(), "done"
        if isinstance(result, pd.DataFrame):
            state.rows = int(len(result))
        self._log("INFO", f"✓ {stage.label} in {state.seconds:.1f}s"
                          + (f" · {state.rows:,} rows" if state.rows is not None else ""), stage.name)
        self._persist()

    def _release_memory(self) -> None:
        self._cache = {k: v for k, v in self._cache.items() if k in KEEP_AFTER_RUN}
        gc.collect()

    # ---------------------------------------------------------------- snapshot
    def progress(self) -> float:
        done = sum(s.weight for s, st in zip(PIPELINE, self.stages) if st.status == "done")
        return round(done / TOTAL_WEIGHT, 4)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "progress": self.progress(),
                "current_stage": self.current_stage,
                "sample_rows": self.sample_rows,
                "input_rows": self.input_rows,
                "source": public_spec(self.source_spec),
                "run_dir": str(self.run_dir),
                "output_dir": str(self.output_dir),
                "stages": [asdict(s) for s in self.stages],
                "summary": self.summary,
                "config": self.config.to_dict(),
            }

    def _persist(self) -> None:
        data = self.snapshot()
        data["log_tail"] = self.logs(after=max(0, self._seq - 300))
        tmp = self.run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(data, indent=1, default=str))
        tmp.replace(self.run_dir / "run.json")

    # ----------------------------------------------------------------- results
    def _enriched(self) -> Optional[pd.DataFrame]:
        df = self._cache.get("universe_enriched_results")
        if isinstance(df, pd.DataFrame):
            return df
        # After a restart: rebuild from the full_results export.
        files = sorted(self.output_dir.glob("full_results_*.csv"))
        if files:
            df = pd.read_csv(files[-1], low_memory=False)
            self._cache["universe_enriched_results"] = df
            return df
        return None

    def _build_summary(self) -> Dict[str, Any]:
        df = self._enriched()
        if df is None:
            return {}
        conf = df["confidence"].fillna("UNKNOWN")
        method = df["match_method"].fillna("Unmatched")
        new_mask = conf.eq("NEW_ENTITY")
        clusters = df.loc[new_mask, "cluster_id"].dropna() if "cluster_id" in df.columns else pd.Series(dtype=float)
        out: Dict[str, Any] = {
            "total_records": int(len(df)),
            "auto_mapped": int(conf.eq("AUTO").sum()),
            "review_needed": int(conf.isin(REVIEW_BANDS).sum()),
            "new_entities": int(new_mask.sum()),
            "no_master_state_pair": int(conf.eq("NO_MASTER_STATE_PAIR").sum()),
            "by_match_method": {str(k): int(v) for k, v in method.value_counts().items()},
            "by_confidence": {str(k): int(v) for k, v in conf.value_counts().items()},
            "clusters": int(clusters.nunique()),
            "avg_cluster_size": round(float(new_mask.sum() / clusters.nunique()), 1) if clusters.nunique() else 0.0,
            "records_with_firm_id": int(df["Firm_ID"].notna().sum()) if "Firm_ID" in df.columns else 0,
            "unique_firm_ids": int(df["Firm_ID"].nunique()) if "Firm_ID" in df.columns else 0,
        }
        um = self._cache.get("updated_master_data")
        if isinstance(um, pd.DataFrame):
            src = um["_source"].astype(str) if "_source" in um.columns else pd.Series(dtype=str)
            out["updated_master"] = {
                "rows": int(len(um)),
                "original_rows": int(src.eq("original_master").sum()),
                "added_rows": int(src.str.startswith("pipeline_").sum()),
                "file": self._latest("updated_master_*.csv", exclude="_with_source"),
            }
        lb = self._cache.get("lead_bank_designation")
        if isinstance(lb, pd.DataFrame):
            out["lead_bank"] = {"rows": int(len(lb)), "columns": [str(c) for c in lb.columns][:12]}
        out["stage_seconds"] = {s.name: s.seconds for s in self.stages if s.seconds is not None}
        out["total_seconds"] = round(sum(s.seconds or 0 for s in self.stages), 1)
        return out

    def _latest(self, pattern: str, exclude: str = "") -> Optional[str]:
        files = [p for p in sorted(self.output_dir.glob(pattern)) if not exclude or exclude not in p.name]
        return str(files[-1]) if files else None

    def review(
        self,
        *,
        confidence: Optional[Iterable[str]] = None,
        search: str = "",
        method: str = "",
        offset: int = 0,
        limit: int = 100,
        sort: str = "match_score",
        descending: bool = True,
    ) -> Dict[str, Any]:
        df = self._enriched()
        if df is None:
            return {"total": 0, "rows": [], "offset": offset, "limit": limit}
        bands = list(confidence) if confidence else list(REVIEW_BANDS)
        sub = df[df["confidence"].isin(bands)]
        if method:
            sub = sub[sub["match_method"].astype(str).str.contains(method, case=False, na=False)]
        if search:
            s = search.strip()
            mask = sub["SECP_NAME"].astype(str).str.contains(s, case=False, na=False, regex=False)
            if "SECP_NAME_MAPPED" in sub.columns:
                mask |= sub["SECP_NAME_MAPPED"].astype(str).str.contains(s, case=False, na=False, regex=False)
            if "suggested_mapping" in sub.columns:
                mask |= sub["suggested_mapping"].astype(str).str.contains(s, case=False, na=False, regex=False)
            sub = sub[mask]
        if sort in sub.columns:
            sub = sub.sort_values(sort, ascending=not descending, na_position="last")
        cols = [c for c in REVIEW_COLUMNS if c in sub.columns]
        page = sub.iloc[offset: offset + limit][cols]
        return {"total": int(len(sub)), "offset": offset, "limit": limit, "columns": cols,
                "rows": frame_records(page)}

    def clusters(self, *, search: str = "", min_size: int = 1, offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        """New-entity clusters (Stage 5-7) summarised one row per cluster."""
        df = self._enriched()
        if df is None or "cluster_id" not in df.columns:
            return {"total": 0, "rows": [], "offset": offset, "limit": limit}
        sub = df[df["confidence"].eq("NEW_ENTITY") & df["cluster_id"].notna()]
        agg = (
            sub.groupby("cluster_id", sort=False)
            .agg(
                cluster_label=("cluster_label", "first"),
                cluster_size=("SECP_NAME", "size"),
                suggested_mapping=("suggested_mapping", "first") if "suggested_mapping" in sub.columns else ("SECP_NAME", "first"),
                suggestion_score=("suggestion_score", "max") if "suggestion_score" in sub.columns else ("SECP_NAME", "size"),
                suggestion_confidence=("suggestion_confidence", "first") if "suggestion_confidence" in sub.columns else ("SECP_NAME", "first"),
                states=("SECP_STATE", lambda s: ", ".join(sorted({str(x) for x in s.dropna()})[:6])),
                sample_names=("SECP_NAME", lambda s: " | ".join(list(dict.fromkeys(s.astype(str)))[:5])),
            )
            .reset_index()
        )
        agg = agg[agg["cluster_size"] >= max(1, min_size)]
        if search:
            s = search.strip()
            agg = agg[
                agg["cluster_label"].astype(str).str.contains(s, case=False, na=False, regex=False)
                | agg["sample_names"].astype(str).str.contains(s, case=False, na=False, regex=False)
                | agg["suggested_mapping"].astype(str).str.contains(s, case=False, na=False, regex=False)
            ]
        agg = agg.sort_values("cluster_size", ascending=False)
        page = agg.iloc[offset: offset + limit]
        return {"total": int(len(agg)), "offset": offset, "limit": limit,
                "columns": [str(c) for c in page.columns], "rows": frame_records(page)}

    def outputs(self) -> List[Dict[str, Any]]:
        files = []
        for p in sorted(self.output_dir.iterdir()) if self.output_dir.exists() else []:
            if p.is_file():
                files.append({"name": p.name, "path": str(p), "size_bytes": p.stat().st_size,
                              "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})
        return files

    def promote_master(self, destination: str) -> Dict[str, Any]:
        """Copy this run's updated master over ``destination`` (keeping a timestamped backup)."""
        src = self._latest("updated_master_*.csv", exclude="_with_source")
        if not src:
            raise FileNotFoundError("This run produced no updated_master_*.csv")
        dest = Path(destination).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if dest.exists():
            backup = dest.with_name(f"{dest.stem}.backup_{datetime.now():%Y%m%d_%H%M%S}{dest.suffix}")
            shutil.copy2(dest, backup)
        shutil.copy2(src, dest)
        self._log("INFO", f"Promoted {Path(src).name} -> {dest}" + (f" (backup: {backup.name})" if backup else ""))
        self._persist()
        return {"source": src, "destination": str(dest), "backup": str(backup) if backup else None}

    # --------------------------------------------------------------- restore
    @classmethod
    def load(cls, run_dir: Path) -> Optional["PipelineRun"]:
        meta = run_dir / "run.json"
        if not meta.is_file():
            return None
        data = json.loads(meta.read_text())
        spec = make_datasource(data["source"]).spec
        cfg = PipelineConfig.from_overrides(data.get("config") or {})
        run = cls(spec, cfg, runs_root=run_dir.parent, sample_rows=data.get("sample_rows", 0),
                  run_id=data["id"], name=data.get("name", ""))
        run.created_at = data.get("created_at", run.created_at)
        run.started_at, run.finished_at = data.get("started_at"), data.get("finished_at")
        run.status = data.get("status", "unknown")
        if run.status in ("queued", "running"):  # engine died mid-run
            run.status, run.error = "failed", data.get("error") or "Engine stopped before the run finished"
        run.error = run.error or data.get("error")
        run.input_rows = data.get("input_rows", {})
        run.summary = data.get("summary")
        run.stages = [StageState(**s) for s in data.get("stages", [])] or run.stages
        for line in data.get("log_tail", []):
            run._logs.append(LogLine(**line))
            run._seq = max(run._seq, line["seq"])
        return run


class RunStore:
    """All runs known to this engine, newest first."""

    def __init__(self, root: Optional[Path] = None):
        self.root = (root or app_data_dir() / "runs").expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._runs: Dict[str, PipelineRun] = {}
        self._lock = threading.Lock()
        for d in sorted(self.root.iterdir()):
            if d.is_dir():
                run = PipelineRun.load(d)
                if run is not None:
                    self._runs[run.id] = run

    def create(self, spec: SourceSpec, config: PipelineConfig, *, sample_rows: int = 0, name: str = "") -> PipelineRun:
        run = PipelineRun(spec, config, runs_root=self.root, sample_rows=sample_rows, name=name)
        with self._lock:
            self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Optional[PipelineRun]:
        return self._runs.get(run_id)

    def list(self) -> List[Dict[str, Any]]:
        snaps = [r.snapshot() for r in self._runs.values()]
        for s in snaps:
            s.pop("config", None)
        return sorted(snaps, key=lambda s: s["created_at"], reverse=True)

    def active(self) -> Optional[PipelineRun]:
        return next((r for r in self._runs.values() if r.status in ("queued", "running")), None)

    def delete(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status in ("queued", "running"):
            return False
        shutil.rmtree(run.run_dir, ignore_errors=True)
        with self._lock:
            del self._runs[run_id]
        return True
