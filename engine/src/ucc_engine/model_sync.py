"""Copy the model files from a network share into the local model folder.

The share (``model_share_dir``) holds the models in fastembed's cache layout -
exactly what ``ucc-engine prefetch-models --set model_cache_dir=<share>`` writes:

    \\\\fileserver\\apps\\ucc-models\\
        models--Qdrant--bge-small-en-v1.5-onnx-Q\\...
        models--Xenova--ms-marco-MiniLM-L-6-v2\\...

``sync_models_from_share`` mirrors those folders into ``model_cache_dir`` on
this PC: a file is copied when it is missing locally, has a different size, or
is newer on the share. Up to date → nothing is read but directory listings.
After a sync the engine loads the models from the local folder in offline mode.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

MODEL_FOLDER_PREFIX = "models--"
MANIFEST = ".synced_from.json"

Log = Callable[[str, str], None]


def _nolog(_level: str, _msg: str) -> None:
    pass


def _needs_copy(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    try:
        a, b = src.stat(), dst.stat()
    except OSError:
        return True
    return a.st_size != b.st_size or a.st_mtime > b.st_mtime + 1.0


def model_folders(root: Path) -> List[Path]:
    """The model folders directly under ``root`` (``models--*``)."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(MODEL_FOLDER_PREFIX))


def local_models_present(cache_dir: str) -> bool:
    return bool(model_folders(Path(cache_dir)))


def sync_models_from_share(share_dir: str, cache_dir: str, log: Log = _nolog) -> Dict[str, Any]:
    """Mirror the model folders on ``share_dir`` into ``cache_dir``.

    Returns ``{"share", "local", "models": [names], "copied": n, "bytes": n,
    "seconds": s, "status": "updated"|"up_to_date"|"share_unreachable"}``.
    Raises ``FileNotFoundError`` only when the share is unreachable *and* no
    model is present locally (there is then nothing to run with).
    """
    share, local = Path(share_dir), Path(cache_dir)
    t0 = time.perf_counter()
    folders = model_folders(share)
    if not folders:
        if local_models_present(cache_dir):
            log("WARNING", f"Model share {share_dir!r} is not reachable or holds no models; using the local copy in {cache_dir}")
            return {"share": share_dir, "local": cache_dir, "models": [p.name for p in model_folders(local)],
                    "copied": 0, "bytes": 0, "seconds": 0.0, "status": "share_unreachable"}
        raise FileNotFoundError(
            f"Model share {share_dir!r} is not reachable (or holds no 'models--*' folders) and no models are present "
            f"locally in {cache_dir}. Connect to the network share once, or point model_dir at a folder that has the models."
        )

    local.mkdir(parents=True, exist_ok=True)
    copied = 0
    nbytes = 0
    for folder in folders:
        for root, _dirs, files in os.walk(folder):
            rel_root = Path(root).relative_to(share)
            (local / rel_root).mkdir(parents=True, exist_ok=True)
            for name in files:
                src = Path(root) / name
                dst = local / rel_root / name
                if _needs_copy(src, dst):
                    tmp = dst.with_name(dst.name + ".part")
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                    copied += 1
                    nbytes += dst.stat().st_size
    secs = round(time.perf_counter() - t0, 1)
    status = "updated" if copied else "up_to_date"
    (local / MANIFEST).write_text(json.dumps({"share": str(share), "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                              "models": [p.name for p in folders], "copied": copied}, indent=2))
    if copied:
        log("INFO", f"Copied {copied} model file(s) ({nbytes / 1e6:.0f} MB) from {share} to {local} in {secs}s")
    else:
        log("INFO", f"Models in {local} are up to date with {share}")
    return {"share": str(share), "local": str(local), "models": [p.name for p in folders],
            "copied": copied, "bytes": nbytes, "seconds": secs, "status": status}


def sync_status(share_dir: str, cache_dir: str) -> Dict[str, Any]:
    """What the UI shows: is the share reachable, what is local, when was it last synced."""
    local = Path(cache_dir)
    manifest: Dict[str, Any] = {}
    try:
        manifest = json.loads((local / MANIFEST).read_text())
    except Exception:  # noqa: BLE001
        pass
    share_models = [p.name for p in model_folders(Path(share_dir))] if share_dir else []
    return {
        "share": share_dir,
        "share_reachable": bool(share_models),
        "share_models": share_models,
        "local": str(local),
        "local_models": [p.name for p in model_folders(local)],
        "last_synced": manifest.get("synced_at"),
        "last_synced_from": manifest.get("share"),
    }
