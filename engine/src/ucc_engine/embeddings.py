"""Local embedding backends.

Both backends are ordinary pip packages that run on the CPU of the machine the
engine is installed on; nothing is sent over the network at inference time.

* ``fastembed``             - ONNX Runtime, no torch, ~⅓ the size and faster on
                              CPU. Default when installed.
* ``sentence-transformers`` - the original torch model used by the Dagster
                              project; bit-for-bit parity with past runs.

Each backend exposes the subset of the ``SentenceTransformer`` interface the
generated core calls:  ``encode(texts, batch_size=, normalize_embeddings=, ...)``
returning a float32 ``(n, dim)`` array, plus a stable ``model_id`` used to key
the persistent query-embedding cache.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

# fastembed registers the same weights under a different repo id.
_FASTEMBED_ALIASES = {
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def available_backends() -> dict[str, bool]:
    return {
        "fastembed": _available("fastembed"),
        "sentence-transformers": _available("sentence_transformers"),
    }


def _as_list(texts: Any) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    return ["" if t is None else str(t) for t in texts]


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class _SharedModel:
    """Copy-proof base: the core stores the model in ``DataFrame.attrs`` and
    pandas deep-copies ``attrs`` on every ``df.copy()``. A loaded model is
    immutable and shared, so copying returns the same instance (this also
    removes the hidden cost of deep-copying a transformer per stage)."""

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):  # pickling (e.g. multiprocessing) is never meaningful
        raise TypeError(f"{type(self).__name__} is a process-local model and cannot be pickled")


def fastembed_location(cache_dir: str, model_dir: str = "") -> dict[str, Any]:
    """Where fastembed should look for model files.

    ``model_dir`` is an *offline* model folder in the normal cache layout
    (``models--Qdrant--bge-small-en-v1.5-onnx-Q/``, ``models--Xenova--ms-marco-MiniLM-L-6-v2/``
    - exactly what ``ucc-engine prefetch-models`` writes). It may be a network
    share / UNC path; the files are only ever read. With it set, nothing is
    downloaded. Without it, ``cache_dir`` is used and missing models download.
    """
    if model_dir:
        return {"cache_dir": model_dir, "local_files_only": True}
    return {"cache_dir": cache_dir}


class FastEmbedBackend(_SharedModel):
    """``fastembed.TextEmbedding`` wrapped to look like ``SentenceTransformer``."""

    def __init__(self, model_name: str, cache_dir: str, model_dir: str = "", threads: int = 0):
        from fastembed import TextEmbedding  # lazy: heavy import

        name = _FASTEMBED_ALIASES.get(model_name, model_name)
        kwargs: dict[str, Any] = {"model_name": name, **fastembed_location(cache_dir, model_dir)}
        if threads > 0:
            kwargs["threads"] = threads
        self._model = TextEmbedding(**kwargs)
        self.model_id = f"fastembed:{name}"
        self._dim: int | None = None

    def encode(
        self,
        texts: Any,
        batch_size: int = 256,
        normalize_embeddings: bool = False,
        **_ignored,
    ) -> np.ndarray:
        items = _as_list(texts)
        if not items:
            return np.empty((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        vecs = list(self._model.embed(items, batch_size=max(1, int(batch_size))))
        mat = np.ascontiguousarray(np.vstack(vecs), dtype=np.float32)
        self._dim = mat.shape[1]
        if normalize_embeddings:
            mat = np.ascontiguousarray(_l2(mat), dtype=np.float32)
        return mat

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["probe"]).shape[1])
        return self._dim


class SentenceTransformerBackend(_SharedModel):
    """Thin wrapper over the original torch model."""

    def __init__(self, model_name: str, cache_dir: str, model_dir: str = "", threads: int = 0):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        if threads > 0:
            try:
                import torch

                torch.set_num_threads(threads)
            except Exception:  # noqa: BLE001
                pass
        self._model = SentenceTransformer(
            model_name, device="cpu", cache_folder=model_dir or cache_dir or None, local_files_only=bool(model_dir)
        )
        self.model_id = f"st:{model_name}"

    def encode(self, texts: Any, batch_size: int = 256, normalize_embeddings: bool = False, **_ignored) -> np.ndarray:
        items = _as_list(texts)
        if not items:
            return np.empty((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        mat = self._model.encode(
            items,
            batch_size=max(1, int(batch_size)),
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
        )
        return np.ascontiguousarray(mat, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


def resolve_backend(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    have = available_backends()
    if req == "auto":
        if have["fastembed"]:
            return "fastembed"
        if have["sentence-transformers"]:
            return "sentence-transformers"
        raise RuntimeError(
            "No local embedding backend installed. `pip install fastembed` "
            "(recommended) or `pip install sentence-transformers`."
        )
    if req in ("fastembed", "fe"):
        if not have["fastembed"]:
            raise RuntimeError("embedding_backend=fastembed but fastembed is not installed")
        return "fastembed"
    if req in ("sentence-transformers", "st", "sentence_transformers", "local"):
        if not have["sentence-transformers"]:
            raise RuntimeError("embedding_backend=sentence-transformers but it is not installed")
        return "sentence-transformers"
    raise RuntimeError(f"Unknown embedding_backend {requested!r}")


_MODEL_CACHE: dict[tuple, Any] = {}


_SYNCED: set[tuple[str, str]] = set()


def model_location(config: Any, context: Any = None) -> tuple[str, str]:
    """``(cache_dir, model_dir)`` to load models from, honouring the three ways
    a site can provide them:

    1. ``model_dir`` set       - read in place (local folder or network share), offline.
    2. ``model_share_dir`` set - copy anything missing/newer from the share into
                                 ``model_cache_dir`` (once per process), then load
                                 the local copy offline.
    3. neither                 - ``model_cache_dir``; missing models download.
    """
    model_dir = str(getattr(config, "model_dir", "") or "")
    cache_dir = str(getattr(config, "model_cache_dir", "") or Path.home() / ".cache" / "ucc-engine" / "models")
    share_dir = str(getattr(config, "model_share_dir", "") or "")
    if model_dir:
        if not Path(model_dir).is_dir():
            raise FileNotFoundError(f"model_dir {model_dir!r} does not exist or is not reachable")
        return cache_dir, model_dir
    if share_dir:
        from .model_sync import sync_models_from_share

        if (share_dir, cache_dir) not in _SYNCED:
            log = (lambda lvl, msg: getattr(context.log, lvl.lower(), context.log.info)(msg)) if context is not None else (lambda _l, _m: None)
            sync_models_from_share(share_dir, cache_dir, log)
            _SYNCED.add((share_dir, cache_dir))
        return cache_dir, cache_dir  # local copy, offline
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir, ""


def load_embedding_model(config: Any, context: Any = None):
    """Factory used by the generated core in place of ``SentenceTransformer(...)``.

    Models are memoised per (backend, model, dir) for the life of the process so
    the five call sites in the core share one loaded model.
    """
    backend = resolve_backend(getattr(config, "embedding_backend", "auto"))
    model_name = getattr(config, "embedding_model", "BAAI/bge-small-en-v1.5")
    cache_dir, model_dir = model_location(config, context)
    threads = int(getattr(config, "model_threads", 0) or 0)
    key = (backend, model_name, model_dir, cache_dir, threads)

    model = _MODEL_CACHE.get(key)
    if model is None:
        if context is not None:
            context.log.info(f"Loading embedding model {model_name!r} via {backend} (local CPU)")
        cls = FastEmbedBackend if backend == "fastembed" else SentenceTransformerBackend
        model = cls(model_name, cache_dir=cache_dir, model_dir=model_dir, threads=threads)
        _MODEL_CACHE[key] = model
    return model


def prefetch_models(config: Any) -> dict[str, str]:
    """Download (if needed) the embedding + reranker models so a first run is offline-safe."""
    from .rerank import try_load_reranker

    out: dict[str, str] = {}
    model = load_embedding_model(config)
    out["embedding"] = model.model_id
    cache_dir, model_dir = model_location(config)
    out["location"] = model_dir or cache_dir
    rr = try_load_reranker(getattr(config, "advanced_cross_encoder_model", ""), config=config)
    out["reranker"] = getattr(rr, "model_id", "unavailable") if rr is not None else "unavailable"
    return out
