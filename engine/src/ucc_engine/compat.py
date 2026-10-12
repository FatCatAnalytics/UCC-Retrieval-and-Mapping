"""Dagster compatibility shims so the generated ``core.pipeline_logic`` runs as plain functions.

* ``asset``               -> pass-through decorator
* ``Output(value, ...)``  -> the bare value
* ``MaterializeResult``   -> the metadata dict
* ``MetadataValue``       -> identity constructors
* ``AssetExecutionContext`` -> exposes ``.log``; every message is forwarded to an
  optional sink so the runner can stream logs to the UI.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

_logger = logging.getLogger("ucc_engine.core")

LogSink = Callable[[str, str], None]  # (level, message)


def asset(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn


class _ContextLogger:
    __slots__ = ("_stage", "_sink")

    def __init__(self, stage: str, sink: Optional[LogSink]):
        self._stage = stage
        self._sink = sink

    def _emit(self, level: str, msg, *args) -> None:
        text = str(msg) % args if args else str(msg)
        getattr(_logger, level)("[%s] %s", self._stage, text)
        if self._sink is not None:
            self._sink(level.upper(), text)

    def debug(self, msg, *args, **_):
        self._emit("debug", msg, *args)

    def info(self, msg, *args, **_):
        self._emit("info", msg, *args)

    def warning(self, msg, *args, **_):
        self._emit("warning", msg, *args)

    def error(self, msg, *args, **_):
        self._emit("error", msg, *args)


class AssetExecutionContext:
    """Minimal stand-in: the ported logic only uses ``context.log``."""

    def __init__(self, stage: str = "ucc_engine", sink: Optional[LogSink] = None):
        self.stage = stage
        self.log = _ContextLogger(stage, sink)


def Output(value=None, metadata=None, **_):
    return value


def MaterializeResult(metadata=None, **_):
    return metadata or {}


class MetadataValue:
    int = staticmethod(lambda v: v)
    float = staticmethod(lambda v: v)
    text = staticmethod(lambda v: v)
    md = staticmethod(lambda v: v)
    json = staticmethod(lambda v: v)
    path = staticmethod(lambda v: v)
    bool = staticmethod(lambda v: v)
    url = staticmethod(lambda v: v)
