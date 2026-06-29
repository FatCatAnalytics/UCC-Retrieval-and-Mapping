"""Unit tests for the Databricks embedding backend.

These run WITHOUT a real Databricks endpoint -- a fake deploy client records the
batches it receives and returns deterministic vectors, so we can assert on
batching, float32 dtype/shape, normalisation, retries and the local fallback.

Run:  python -m pytest databricks/tests/test_embeddings.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ucc_pipeline.embeddings import (  # noqa: E402
    DatabricksEmbeddingBackend,
    _extract_embeddings,
    _l2_normalize,
    load_embedding_model,
)


class FakeClient:
    """Records every predict_endpoint call and returns OpenAI-shaped vectors."""

    def __init__(self, dim=4, fail_times=0, key="input"):
        self.dim = dim
        self.calls = []          # list of batch lengths seen
        self.payload_keys = []   # which key each call used
        self.fail_times = fail_times
        self.expected_key = key

    def predict_endpoint(self, endpoint, payload):
        # Only accept the expected key; emulate endpoints that 400 on the wrong one.
        if self.expected_key not in payload:
            raise ValueError(f"unexpected payload keys {list(payload)}")
        batch = payload[self.expected_key]
        self.payload_keys.append(self.expected_key)

        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated transient 503")

        self.calls.append(len(batch))
        data = [
            {"embedding": [float(i + 1)] * self.dim}  # constant non-unit vector
            for i, _ in enumerate(batch)
        ]
        return {"object": "list", "data": data}


def _backend(client, dim=4, **kw):
    return DatabricksEmbeddingBackend(
        endpoint="databricks-bge-large-en-v1.5", dim=dim, client=client, **kw
    )


def test_batching_chunks_to_max_batch():
    client = FakeClient(dim=4)
    be = _backend(client, max_batch=3)
    out = be.encode([f"t{i}" for i in range(7)])
    # 7 texts, max_batch 3 -> chunks of 3, 3, 1
    assert client.calls == [3, 3, 1]
    assert out.shape == (7, 4)


def test_returns_float32_2d():
    be = _backend(FakeClient(dim=8))
    out = be.encode(["a", "b"])
    assert out.dtype == np.float32
    assert out.ndim == 2 and out.shape == (2, 8)
    assert out.flags["C_CONTIGUOUS"]


def test_empty_input_shape():
    be = _backend(FakeClient(dim=5), dim=5)
    out = be.encode([])
    assert out.shape == (0, 5)
    assert out.dtype == np.float32


def test_normalize_embeddings_l2():
    be = _backend(FakeClient(dim=4))
    out = be.encode(["x", "y"], normalize_embeddings=True)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_no_normalize_keeps_raw():
    be = _backend(FakeClient(dim=4))
    out = be.encode(["x"], normalize_embeddings=False)
    # first row is all 1.0 -> norm = 2.0, not normalised
    assert np.allclose(np.linalg.norm(out[0]), 2.0)


def test_retry_then_succeed():
    client = FakeClient(dim=4, fail_times=2)
    be = _backend(client, max_retries=3, retry_base_delay=0.0)
    out = be.encode(["a", "b"])
    assert out.shape == (2, 4)


def test_retry_exhausted_raises():
    client = FakeClient(dim=4, fail_times=99)
    be = _backend(client, max_retries=1, retry_base_delay=0.0)
    try:
        be.encode(["a"])
    except RuntimeError as e:
        assert "failed after" in str(e)
    else:
        raise AssertionError("expected RuntimeError after retries exhausted")


def test_input_key_fallback_to_inputs():
    # Endpoint only accepts "inputs"; backend should discover that and cache it.
    client = FakeClient(dim=4, key="inputs")
    be = _backend(client)
    out = be.encode(["a", "b", "c"])
    assert out.shape == (3, 4)
    assert set(client.payload_keys) == {"inputs"}


def test_extract_embeddings_shapes():
    # predictions as list-of-lists
    arr = _extract_embeddings({"predictions": [[1.0, 2.0], [3.0, 4.0]]})
    assert arr.shape == (2, 2) and arr.dtype == np.float32
    # bare list of dicts
    arr2 = _extract_embeddings([{"embedding": [1, 2, 3]}])
    assert arr2.shape == (1, 3)
    # single 1-D vector reshaped to (1, n)
    arr3 = _extract_embeddings({"embeddings": [0.1, 0.2, 0.3]})
    assert arr3.shape == (1, 3)


def test_l2_normalize_handles_zero_rows():
    m = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    n = _l2_normalize(m)
    assert np.allclose(n[0], [0.0, 0.0])      # zero row stays zero (no nan)
    assert np.allclose(np.linalg.norm(n[1]), 1.0)


def test_get_dimension():
    be = _backend(FakeClient(dim=4), dim=1024)
    assert be.get_sentence_embedding_dimension() == 1024


class _Cfg:
    embedding_backend = "databricks"
    embedding_endpoint = "databricks-gte-large-en-v1.5"
    embedding_dim = 1024
    embedding_max_batch = 64
    embedding_max_retries = 2
    embedding_retry_base_delay = 0.1
    embedding_model = "BAAI/bge-small-en-v1.5"


def test_factory_builds_databricks_backend():
    be = load_embedding_model(_Cfg())
    assert isinstance(be, DatabricksEmbeddingBackend)
    assert be.endpoint == "databricks-gte-large-en-v1.5"
    assert be.dim == 1024
    assert be.max_batch == 64


def test_factory_local_imports():
    # local backend must at least be selectable; importing sentence-transformers
    # may fail in this env, so we only assert the factory dispatches there.
    cfg = _Cfg()
    cfg.embedding_backend = "local"
    try:
        be = load_embedding_model(cfg)
    except Exception as e:
        # acceptable: sentence-transformers not installed in CI
        assert "sentence_transformers" in repr(e) or "SentenceTransformer" in repr(e)
    else:
        assert hasattr(be, "encode") and hasattr(be, "get_sentence_embedding_dimension")


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
