"""Cross-encoder reranker for the Stage 4 ``REVIEW_MEDIUM`` band - local backends only.

Keeps the exact interface the generated core imports
(``try_load_reranker``, ``score_pairs``, ``promote_confidence``) while serving
the model through ``fastembed.rerank.cross_encoder.TextCrossEncoder`` (ONNX) or
``sentence_transformers.CrossEncoder`` (torch), whichever is installed.

Degrades gracefully: if no backend/model can be loaded ``try_load_reranker``
returns ``None`` and Stage 4 proceeds with the rerank as a no-op.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .embeddings import _SharedModel, available_backends

# sentence-transformers id -> fastembed (ONNX export) id
_FASTEMBED_RERANKERS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Xenova/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base": "BAAI/bge-reranker-base",
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class FastEmbedReranker(_SharedModel):
    def __init__(self, model_name: str, cache_dir: str, model_dir: str = "", threads: int = 0):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy

        from .embeddings import fastembed_location

        name = _FASTEMBED_RERANKERS.get(model_name, model_name)
        kwargs: dict[str, Any] = {"model_name": name, **fastembed_location(cache_dir, model_dir)}
        if threads > 0:
            kwargs["threads"] = threads
        self._model = TextCrossEncoder(**kwargs)
        self.model_id = f"fastembed:{name}"

    def score(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        # TextCrossEncoder.rerank scores one query against many docs, so group
        # the arbitrary (query, candidate) pairs by query and scatter back.
        by_query: dict[str, list[int]] = defaultdict(list)
        for i, (q, _) in enumerate(pairs):
            by_query[q].append(i)
        out = [0.0] * len(pairs)
        for q, idxs in by_query.items():
            docs = [pairs[i][1] for i in idxs]
            raw = list(self._model.rerank(q, docs, batch_size=64))
            for i, logit in zip(idxs, raw):
                out[i] = _sigmoid(float(logit))
        return out


class SentenceTransformerReranker(_SharedModel):
    def __init__(self, model_name: str, cache_dir: str, model_dir: str = "", threads: int = 0):
        from sentence_transformers import CrossEncoder  # lazy

        self._model = CrossEncoder(
            model_name, device="cpu", max_length=128, cache_folder=model_dir or cache_dir or None,
            local_files_only=bool(model_dir),
        )
        self.model_id = f"st:{model_name}"

    def score(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        raw = self._model.predict([list(p) for p in pairs], batch_size=64, show_progress_bar=False)
        return [_sigmoid(float(x)) for x in raw]


_RERANKER_CACHE: dict[tuple, Any] = {}


def try_load_reranker(model_name: str, device: str = "cpu", config: Any = None):
    """Return a reranker handle or ``None`` (rerank becomes a no-op)."""
    if not model_name:
        return None
    backend_pref = str(getattr(config, "embedding_backend", "auto") or "auto").lower()
    from .embeddings import model_location

    try:
        cache_dir, model_dir = model_location(config)
    except FileNotFoundError:
        return None  # no models anywhere: rerank becomes a no-op, the embedding loader reports the error
    threads = int(getattr(config, "model_threads", 0) or 0)
    key = (model_name, backend_pref, cache_dir, model_dir, threads)
    if key in _RERANKER_CACHE:
        return _RERANKER_CACHE[key]

    have = available_backends()
    order: list[str] = []
    if backend_pref in ("sentence-transformers", "st", "local"):
        order = ["sentence-transformers", "fastembed"]
    else:
        order = ["fastembed", "sentence-transformers"]

    handle = None
    for backend in order:
        if not have.get(backend):
            continue
        try:
            cls = FastEmbedReranker if backend == "fastembed" else SentenceTransformerReranker
            handle = cls(model_name, cache_dir=cache_dir, model_dir=model_dir, threads=threads)
            break
        except Exception:  # noqa: BLE001 - fall through to next backend / None
            handle = None
    _RERANKER_CACHE[key] = handle
    return handle


def score_pairs(model, pairs: Sequence[Sequence[str]]) -> List[float]:
    """Sigmoid-mapped relevance in [0, 1] per pair; ``[]`` on any failure (no-op)."""
    if not pairs or model is None:
        return []
    try:
        scores = model.score(list(pairs))
    except Exception:  # noqa: BLE001
        return []
    return scores if len(scores) == len(pairs) else []


def promote_confidence(
    current_band: str,
    rerank_score: float,
    promote_to_auto: float,
    promote_to_high: float,
) -> Optional[str]:
    """New band iff promotion is warranted, else ``None``. Never demotes."""
    if current_band != "REVIEW_MEDIUM":
        return None
    if rerank_score >= promote_to_auto:
        return "AUTO"
    if rerank_score >= promote_to_high:
        return "REVIEW_HIGH"
    return None
