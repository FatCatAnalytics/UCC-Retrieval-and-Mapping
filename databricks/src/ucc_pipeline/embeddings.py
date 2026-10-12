# =============================================================================
# EMBEDDING BACKEND ABSTRACTION
# =============================================================================
# The matching/clustering pipeline (pipeline_logic.py) consumes embeddings via a
# single ``SentenceTransformer.encode(list_of_texts, ...)`` style call and, in a
# couple of spots, ``model.get_sentence_embedding_dimension()``. This module
# provides drop-in replacements for that object so the FAISS index build,
# cosine-similarity search and UnionFind clustering downstream stay byte-for-byte
# identical -- only *where the vectors come from* changes.
#
# Two backends:
#   * ``databricks`` (default) -- calls a Databricks Foundation Model API serving
#     endpoint (e.g. ``databricks-bge-large-en-v1.5``) through
#     ``mlflow.deployments``. This is the Databricks-native path and needs no
#     extra deps beyond ``mlflow`` (bundled in Databricks Runtime ML).
#   * ``local`` -- the original ``sentence-transformers`` path, kept so the
#     pipeline still runs outside Databricks (CI, laptops) unchanged.
#
# Both backends return **float32 numpy arrays** shaped ``(n_texts, dim)`` and,
# when ``normalize_embeddings=True`` (the pipeline always passes this), L2-
# normalised rows -- exactly what ``IndexFlatIP`` expects for cosine similarity.
# =============================================================================
from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence

import numpy as np


# Databricks serving endpoints cap the number of inputs per request. We chunk
# large encode() calls into sub-batches of this size. Configurable via
# PipelineConfig.embedding_max_batch.
_DEFAULT_MAX_BATCH = 128


def _as_text_list(texts: Any) -> List[str]:
    """Normalise the encode() input to a plain list[str].

    The pipeline always passes a list, but a bare string is tolerated so the
    backend behaves like SentenceTransformer.encode.
    """
    if isinstance(texts, str):
        return [texts]
    return [("" if t is None else str(t)) for t in texts]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe against zero vectors)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class LocalSentenceTransformerBackend:
    """Thin wrapper over ``sentence_transformers.SentenceTransformer``.

    Preserves the original behaviour exactly: encode() forwards every kwarg
    (batch_size, normalize_embeddings, show_progress_bar, device, ...) straight
    through to the underlying model.
    """

    def __init__(self, model_path: str, fallback: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy import

        try:
            self._model = SentenceTransformer(model_path, device="cpu")
            self.model_id = str(model_path)
        except Exception:
            self._model = SentenceTransformer(fallback, device="cpu")
            self.model_id = str(fallback)

    def encode(self, texts: Any, **kwargs) -> np.ndarray:
        out = self._model.encode(texts, **kwargs)
        return np.ascontiguousarray(out, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        try:
            return int(self._model.get_sentence_embedding_dimension())
        except Exception:
            return 384


class DatabricksEmbeddingBackend:
    """Embedding backend backed by a Databricks Foundation Model API endpoint.

    Uses ``mlflow.deployments.get_deploy_client("databricks")`` and calls
    ``predict_endpoint(endpoint, {"input": batch})``. The Databricks
    Foundation Model embedding endpoints are OpenAI-compatible and expect the
    ``input`` key; some custom/older deployments expect ``inputs`` instead, so we
    transparently retry with that key if the first shape is rejected.

    Parameters
    ----------
    endpoint : str
        Serving endpoint name, e.g. ``databricks-bge-large-en-v1.5``.
    dim : int
        Embedding dimensionality of the endpoint (bge-large = 1024,
        bge-small = 384, gte-large = 1024). Used for empty-input shaping and as
        the value returned by ``get_sentence_embedding_dimension``.
    max_batch : int
        Max texts per endpoint request. Larger encode() calls are chunked.
    max_retries : int
        Retry attempts per chunk on transient errors (rate limits, 5xx).
    retry_base_delay : float
        Base seconds for exponential backoff (delay = base * 2**attempt).
    client : optional
        Pre-built deploy client. Mainly for tests; production builds it lazily.
    """

    def __init__(
        self,
        endpoint: str,
        dim: int,
        max_batch: int = _DEFAULT_MAX_BATCH,
        max_retries: int = 4,
        retry_base_delay: float = 0.5,
        client: Any = None,
    ):
        self.endpoint = str(endpoint)
        self.model_id = str(endpoint)
        self.dim = int(dim)
        self.max_batch = max(1, int(max_batch))
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self._client = client
        # Remember which payload key works so we don't pay the retry every call.
        self._input_key: Optional[str] = None

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is None:
            from mlflow.deployments import get_deploy_client  # lazy import

            self._client = get_deploy_client("databricks")
        return self._client

    # --------------------------------------------------------------- one chunk
    def _predict_chunk(self, batch: List[str]) -> np.ndarray:
        client = self._get_client()
        keys_to_try = [self._input_key] if self._input_key else ["input", "inputs"]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            for key in keys_to_try:
                try:
                    resp = client.predict_endpoint(self.endpoint, {key: batch})
                    vecs = _extract_embeddings(resp)
                    if vecs.shape[0] != len(batch):
                        raise ValueError(
                            f"endpoint returned {vecs.shape[0]} vectors for "
                            f"{len(batch)} inputs"
                        )
                    self._input_key = key  # cache the working key
                    return vecs
                except Exception as e:  # noqa: BLE001 -- retry/backoff below
                    last_err = e
                    continue
            # All key shapes failed this attempt -- back off then retry.
            if attempt < self.max_retries:
                time.sleep(self.retry_base_delay * (2 ** attempt))

        raise RuntimeError(
            f"Databricks embedding endpoint {self.endpoint!r} failed after "
            f"{self.max_retries + 1} attempts: {last_err!r}"
        )

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        texts: Any,
        normalize_embeddings: bool = False,
        **_ignored,
    ) -> np.ndarray:
        """Embed ``texts`` and return a float32 ``(n, dim)`` array.

        ``batch_size``, ``show_progress_bar``, ``device`` and ``convert_to_numpy``
        kwargs from the SentenceTransformer call sites are accepted and ignored
        (batching is governed by ``max_batch``; output is always numpy float32).
        """
        items = _as_text_list(texts)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)

        blocks: List[np.ndarray] = []
        for start in range(0, len(items), self.max_batch):
            chunk = items[start:start + self.max_batch]
            blocks.append(self._predict_chunk(chunk))

        mat = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)
        if normalize_embeddings:
            mat = np.ascontiguousarray(_l2_normalize(mat), dtype=np.float32)
        return mat

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def _extract_embeddings(resp: Any) -> np.ndarray:
    """Pull a 2-D float32 array out of whatever shape the endpoint returned.

    Handles the common Databricks / MLflow response shapes:
      * OpenAI-compatible:  {"data": [{"embedding": [...]}, ...]}
      * MLflow predictions: {"predictions": [[...], ...]}  or
                            {"predictions": [{"embedding": [...]}, ...]}
      * {"embeddings": [[...], ...]}
      * a bare list of vectors / list of dicts
    """
    data = resp

    if isinstance(resp, dict):
        for key in ("data", "predictions", "embeddings"):
            if key in resp and resp[key] is not None:
                data = resp[key]
                break

    if data is None:
        raise ValueError("empty embedding response")

    rows: List[Sequence[float]] = []
    for item in data:
        if isinstance(item, dict):
            vec = item.get("embedding")
            if vec is None:
                raise ValueError(f"response item missing 'embedding': {item!r}")
            rows.append(vec)
        else:
            rows.append(item)

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_embedding_model(config: Any, context: Any = None):
    """Factory: build the embedding backend selected by ``config``.

    ``config.embedding_backend`` -> ``"databricks"`` (default) or ``"local"``.
    Returns an object exposing ``encode(...)`` and
    ``get_sentence_embedding_dimension()`` -- a drop-in for SentenceTransformer.
    """
    backend = str(getattr(config, "embedding_backend", "databricks")).strip().lower()

    def _log(msg: str) -> None:
        if context is not None and getattr(context, "log", None) is not None:
            context.log.info(msg)

    if backend == "local":
        model_path = getattr(config, "embedding_model", "BAAI/bge-small-en-v1.5")
        _log(f"Embedding backend=local; loading SentenceTransformer({model_path!r})")
        return LocalSentenceTransformerBackend(model_path)

    endpoint = getattr(config, "embedding_endpoint", "databricks-bge-large-en-v1.5")
    dim = int(getattr(config, "embedding_dim", 1024))
    _log(
        f"Embedding backend=databricks; endpoint={endpoint!r} dim={dim} "
        f"(Foundation Model API)"
    )
    return DatabricksEmbeddingBackend(
        endpoint=endpoint,
        dim=dim,
        max_batch=int(getattr(config, "embedding_max_batch", _DEFAULT_MAX_BATCH)),
        max_retries=int(getattr(config, "embedding_max_retries", 4)),
        retry_base_delay=float(getattr(config, "embedding_retry_base_delay", 0.5)),
    )
