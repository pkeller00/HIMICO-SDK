"""Use a model hosted by `insight serve` instead of local weights.

    Pipeline(backend="http://gpu-box:8080")          # token, if the server needs one: INSIGHT_SERVER_TOKEN
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from ..safe_graph import dump_graph
from .base import Prediction, Predictor


class RemoteError(RuntimeError):
    pass


def _request(url: str, data: bytes | None = None, timeout: float = 600):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    tok = os.environ.get("INSIGHT_SERVER_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", "")
        except Exception:
            msg = ""
        raise RemoteError(f"{url}: HTTP {e.code} {msg}".strip()) from None
    except urllib.error.URLError as e:
        raise RemoteError(f"cannot reach {url} ({e.reason}); is `insight serve` running?") from None


def ping(base: str) -> dict:
    return _request(base.rstrip("/") + "/ping", timeout=10)


class RemotePredictor(Predictor):
    """Sends each patient graph to `<base>/predict/<model>` and wraps the answer as a Prediction."""

    def __init__(self, base: str, model: str, model_cls):
        self.base, self.model = base.rstrip("/"), model
        self.features, self.dthresh, self.local_name = model_cls.features, model_cls.dthresh, model_cls.local_name
        info = ping(self.base)
        if model not in info.get("models", []):
            raise RemoteError(f"server {self.base} does not offer {model!r} (has {info.get('models')})")

    def predict(self, graph) -> Prediction:
        r = _request(f"{self.base}/predict/{self.model}", dump_graph(graph))
        known = {"patient_id", "score", "risk", "risk_group", "folds", "local"}
        return Prediction(patient=str(getattr(graph, "id", "")), score=r["score"], risk=r["risk"], group=r.get("risk_group"),
                          folds=r.get("folds", []), local=None if r.get("local") is None else np.asarray(r["local"], np.float32),
                          extra={k: v for k, v in r.items() if k not in known and not k.startswith("fold")})
