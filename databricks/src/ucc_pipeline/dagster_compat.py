"""Lightweight Dagster compatibility shims for the Databricks port.

The pipeline logic in ``pipeline_logic.py`` is a 1:1 copy of the original
Dagster ``assets.py`` — only its import lines were changed to point here. That
keeps the matching/clustering logic byte-for-byte identical to the proven
Dagster implementation while removing the Dagster runtime dependency.

These shims make the original ``@asset`` functions behave as plain callables:

* ``asset``               -> pass-through decorator (drops UI metadata).
* ``AssetExecutionContext`` -> object exposing ``.log`` backed by stdlib logging.
* ``Output(value, ...)``  -> returns ``value`` directly, so assets that did
  ``return Output(value=df, metadata=...)`` simply return ``df``.
* ``MaterializeResult(metadata=...)`` -> returns the metadata dict.
* ``MetadataValue``       -> no-op typed-value constructors (identity).

Each task entrypoint builds an ``AssetExecutionContext`` and calls the logic
function directly, e.g. ``pipeline_logic.ucc_raw_data(ctx, config)``.
"""
from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def asset(*decorator_args, **decorator_kwargs):
    """Pass-through replacement for ``dagster.asset``.

    Supports both ``@asset`` and ``@asset(description=..., group_name=...)``.
    """
    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]

    def _decorate(fn):
        return fn

    return _decorate


class _ContextLogger:
    def __init__(self, name: str = "ucc_pipeline"):
        self._log = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        self._log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log.debug(msg, *args, **kwargs)


class AssetExecutionContext:
    """Minimal stand-in providing the ``.log`` interface used by the assets."""

    def __init__(self, name: str = "ucc_pipeline"):
        self.log = _ContextLogger(name)


def Output(value=None, metadata=None, **_kwargs):
    """Unwrap Dagster ``Output`` to the bare value."""
    return value


def MaterializeResult(metadata=None, **_kwargs):
    """Return the metadata dict in place of a Dagster MaterializeResult."""
    return metadata or {}


class MetadataValue:
    """No-op typed-metadata constructors (identity)."""

    @staticmethod
    def int(value):
        return value

    @staticmethod
    def float(value):
        return value

    @staticmethod
    def text(value):
        return value

    @staticmethod
    def md(value):
        return value

    @staticmethod
    def json(value):
        return value

    @staticmethod
    def path(value):
        return value

    @staticmethod
    def bool(value):
        return value

    @staticmethod
    def url(value):
        return value
