# =============================================================================
# CROSS-ENCODER RERANKER FOR REVIEW_MEDIUM BAND
# =============================================================================
# Stage 4's weighted-sum scorer (embed + fuzzy + ngram + phonetic + acronym) is
# very good at separating clear matches from clear non-matches but struggles in
# the REVIEW_MEDIUM band, where the top candidate is plausible but not certain.
#
# A cross-encoder reads (query, candidate) as a *pair* and returns a calibrated
# relevance score. It's much slower than a bi-encoder per pair, but we only
# invoke it on the small subset of rows that landed in REVIEW_MEDIUM, scoring
# their top-N (default 10) candidates -- typically a few hundred to a few
# thousand pairs per pipeline run, easily handled on CPU.
#
# Decision policy (never demotes):
#   - If the cross-encoder agrees with Stage 4's best pick and gives it a high
#     score, promote AUTO or REVIEW_HIGH.
#   - If the cross-encoder prefers a DIFFERENT candidate strongly, swap to it
#     (only when its score clears the high threshold; otherwise leave the
#     original pick alone -- we never make things worse).
#   - If the cross-encoder is uncertain, leave the Stage 4 verdict unchanged.
#
# This module degrades gracefully: if the model can't be loaded (no internet,
# missing local copy), `try_load_reranker` returns None and Stage 4 acts as if
# rerank were disabled.
# =============================================================================
from __future__ import annotations

import math
from typing import List, Optional, Sequence


def _is_databricks_endpoint(model_name: str) -> bool:
    """Heuristic: a Databricks serving endpoint, not a local cross-encoder id.

    Databricks Foundation Model / served endpoints are referenced by a bare
    endpoint name (e.g. ``databricks-bge-reranker-large-v2``). Local
    sentence-transformers ids look like ``cross-encoder/ms-marco-...``.
    """
    if not model_name:
        return False
    return model_name.startswith("databricks-") or model_name.startswith("endpoints:/")


class DatabricksReranker:
    """Reranker backed by a Databricks serving endpoint (mlflow.deployments).

    Kept behind the same ``try_load_reranker`` / ``score_pairs`` interface used by
    Stage 4 so pipeline_logic.py is unchanged. ``score`` returns relevance in
    [0, 1] -- one per (query, candidate) pair. Any failure returns an empty list
    so Stage 4 treats the rerank as a no-op.
    """

    def __init__(self, endpoint: str, client=None):
        self.endpoint = str(endpoint)
        self._client = client

    def _get_client(self):
        if self._client is None:
            from mlflow.deployments import get_deploy_client  # lazy import

            self._client = get_deploy_client("databricks")
        return self._client

    def score(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        if not pairs:
            return []
        client = self._get_client()
        queries = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        try:
            resp = client.predict_endpoint(
                self.endpoint, {"query": queries, "text": texts}
            )
        except Exception:
            # Alternative payload shape used by some reranker deployments.
            try:
                resp = client.predict_endpoint(
                    self.endpoint,
                    {"inputs": [{"query": q, "text": t} for q, t in zip(queries, texts)]},
                )
            except Exception:
                return []
        return _extract_rerank_scores(resp, len(pairs))


def _extract_rerank_scores(resp, n: int) -> List[float]:
    """Pull n relevance scores out of a reranker endpoint response."""
    data = resp
    if isinstance(resp, dict):
        for key in ("predictions", "data", "scores", "results"):
            if key in resp and resp[key] is not None:
                data = resp[key]
                break
    try:
        out: List[float] = []
        for item in data:
            if isinstance(item, dict):
                val = item.get("score", item.get("relevance_score", item.get("value")))
            else:
                val = item
            out.append(float(val))
    except Exception:
        return []
    return out if len(out) == n else []


def try_load_reranker(model_name: str, device: str = "cpu"):
    """Try to load the reranker. Return a model handle or None on failure.

    If ``model_name`` names a Databricks serving endpoint we return a
    :class:`DatabricksReranker`; otherwise we load a local
    ``sentence_transformers.CrossEncoder`` (imported lazily). On any failure
    return None so Stage 4 acts as if rerank were disabled.
    """
    if _is_databricks_endpoint(model_name):
        try:
            reranker = DatabricksReranker(model_name)
            reranker._get_client()  # fail fast if mlflow/client unavailable
            return reranker
        except Exception:
            return None

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        return None

    try:
        model = CrossEncoder(model_name, device=device, max_length=128)
        return model
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for cross-encoder logit output."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_pairs(model, pairs: Sequence[Sequence[str]]) -> List[float]:
    """Run the cross-encoder over a batch of (query, candidate) pairs.

    Returns a list of sigmoid-mapped relevance scores in [0, 1] (one per pair).
    Returns an empty list on any failure -- caller should treat that as a
    no-op rerank.
    """
    if not pairs:
        return []

    # Databricks-served reranker already returns calibrated [0, 1] scores.
    if isinstance(model, DatabricksReranker):
        return model.score(list(pairs))

    try:
        # CrossEncoder.predict returns raw logits; we sigmoid for [0, 1] range.
        raw = model.predict(list(pairs), batch_size=64, show_progress_bar=False)
    except Exception:
        return []

    try:
        return [_sigmoid(float(x)) for x in raw]
    except Exception:
        return []


def promote_confidence(
    current_band: str,
    rerank_score: float,
    promote_to_auto: float,
    promote_to_high: float,
) -> Optional[str]:
    """Return a new confidence band IFF promotion is warranted, else None.

    Never demotes. The caller treats `None` as "leave the existing band alone".
    """
    if current_band != "REVIEW_MEDIUM":
        return None
    if rerank_score >= promote_to_auto:
        return "AUTO"
    if rerank_score >= promote_to_high:
        return "REVIEW_HIGH"
    return None
