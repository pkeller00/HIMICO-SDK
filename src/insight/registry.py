"""Registries so new feature extractors and prediction models plug in by name.

    from insight.registry import register_model
    from insight.models.base import Predictor

    @register_model("my-model")
    class MyModel(Predictor):
        features = "uni"
        ...

Third-party packages can also expose models through the `insight.models` entry point group.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, TypeVar

T = TypeVar("T")
_MODELS: dict[str, type] = {}
_FEATURES: dict[str, type] = {}


def register_model(name: str) -> Callable[[T], T]:
    def deco(cls):
        _MODELS[name] = cls
        cls.name = name
        return cls
    return deco


def register_features(name: str) -> Callable[[T], T]:
    def deco(cls):
        _FEATURES[name] = cls
        cls.name = name
        return cls
    return deco


def _load_builtin():
    from . import models  # noqa: F401  (registers built-ins)
    from .stages import features  # noqa: F401
    for ep in entry_points(group="insight.models"):
        if ep.name not in _MODELS:
            try:
                _MODELS[ep.name] = ep.load()
            except Exception:  # a broken plugin must not break the SDK
                pass


def get_model(name: str) -> type:
    _load_builtin()
    if name not in _MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_MODELS)}")
    return _MODELS[name]


def get_features(name: str) -> type:
    _load_builtin()
    if name not in _FEATURES:
        raise KeyError(f"unknown feature extractor {name!r}; available: {sorted(_FEATURES)}")
    return _FEATURES[name]


def models() -> dict[str, type]:
    _load_builtin()
    return dict(_MODELS)


def feature_extractors() -> dict[str, type]:
    _load_builtin()
    return dict(_FEATURES)
