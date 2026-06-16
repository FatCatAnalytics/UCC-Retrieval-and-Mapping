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


def try_load_reranker(model_name: str, device: str = "cpu"):
    """Try to load the cross-encoder model. Return the model or None on failure.

    We import sentence_transformers lazily so that environments without it
    don't pay the import cost.
    """
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
