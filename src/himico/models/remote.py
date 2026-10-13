"""Run a model on a HIMICO server instead of local weights.

    Pipeline()                                       # backend "auto": local weights if present, else the HIMICO server
    Pipeline(backend="himico")                       # always the hosted server (API key: `himico login` / HIMICO_API_KEY)
    Pipeline(backend="http://gpu-box:8080")          # a self-hosted `himico serve`
    Pipeline(backend="local")                        # local weights only (weights_dir or Hugging Face access)

Protocol (served by `himico serve` and by the hosted server):
    GET  /ping                 {"status", "version", "models", "auth"}    no key needed
    GET  /account              {"name", "tier", "models", "usage"?}       key needed
    POST /predict/<model>      body: .g graph bytes -> prediction JSON    key needed
Errors are JSON {"error": ...}: 401 missing/invalid key, 403 model not included in the key's tier, 429 quota or rate
limit (with Retry-After), 503 busy or under maintenance.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import numpy as np

from .. import __version__
from ..auth import CONTACT, KEY_URL, api_key, missing_key_message, server_url
from ..safe_graph import dump_graph
from .base import Prediction, Predictor

log = logging.getLogger("himico")
_RETRY = {429, 502, 503, 504}


class RemoteError(RuntimeError):
    pass


def resolve_backend(backend: str | None, model_cls=None, weights_dir: str | None = None) -> str | None:
    """URL of the server to use, or None for local weights."""
    b = (backend or "auto").strip()
    if b == "local":
        return None
    if b == "himico":
        return server_url()
    if b == "auto":
        from ..weights import available
        return None if model_cls is not None and available(model_cls.weights, weights_dir) else server_url()
    if not b.startswith(("http://", "https://")):
        raise ValueError(f"backend must be 'auto', 'himico', 'local' or a server URL, not {backend!r}")
    return b.rstrip("/")


def _request(url: str, data: bytes | None = None, timeout: float = 600, key: str | None = None, attempts: int = 4):
    u = urlsplit(url)
    base = f"{u.scheme}://{u.netloc}"
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
        req.add_header("User-Agent", f"himico-sdk/{__version__}")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
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
            if e.code in _RETRY and attempt < attempts:
                wait = min(float(e.headers.get("Retry-After") or 2 ** attempt), 120)
                log.warning("%s: HTTP %d %s; retrying in %.0fs", url, e.code, msg, wait)
                time.sleep(wait)
                continue
            if e.code == 401:
                raise RemoteError(f"{base} rejected the API key{f' ({msg})' if msg else ''}. Check it with `himico "
                                  f"account`, or request one at {KEY_URL}.") from None
            if e.code == 403:
                raise RemoteError(f"{base}: {msg or 'your API key does not include this model'}. "
                                  f"See `himico account`; to upgrade contact {CONTACT}.") from None
            if e.code == 429:
                raise RemoteError(f"{base}: {msg or 'request quota reached'}. Re-run later to resume; "
                                  f"see `himico account` for your usage.") from None
            raise RemoteError(f"{url}: HTTP {e.code} {msg}".strip()) from None
        except urllib.error.URLError as e:
            if attempt < attempts and data is not None:
                time.sleep(2 ** attempt)
                continue
            hint = ("the hosted HIMICO server may not be live yet or is under maintenance; use backend 'local' with "
                    f"weights you have been given, or contact {CONTACT}" if base == server_url()
                    else "is `himico serve` running?")
            raise RemoteError(f"cannot reach {base} ({e.reason}); {hint}") from None


def ping(base: str) -> dict:
    return _request(base.rstrip("/") + "/ping", timeout=10, attempts=1)


def account(base: str) -> dict:
    base = base.rstrip("/")
    key = api_key(base)
    if not key:
        raise RemoteError(missing_key_message(base))
    return _request(base + "/account", timeout=15, key=key, attempts=1)


class RemotePredictor(Predictor):
    """Sends each patient graph to `<base>/predict/<model>` and wraps the answer as a Prediction."""

    def __init__(self, base: str, model: str, model_cls):
        self.base, self.model = base.rstrip("/"), model
        self.features, self.dthresh, self.local_name = model_cls.features, model_cls.dthresh, model_cls.local_name
        info = ping(self.base)
        if model not in info.get("models", []):
            raise RemoteError(f"server {self.base} does not offer {model!r} (has {info.get('models')})")
        self.key = api_key(self.base)
        if info.get("auth", True):          # fail before the first patient if the key is missing, wrong or too narrow
            acct = account(self.base)
            if acct.get("models") is not None and model not in acct["models"]:
                raise RemoteError(f"your {acct.get('tier', '')} key for {self.base} does not include {model!r}. "
                                  f"Contact {CONTACT} to add it.")
            log.info("predict: %s as %s (%s tier)", self.base, acct.get("name", "?"), acct.get("tier", "?"))

    def predict(self, graph) -> Prediction:
        r = _request(f"{self.base}/predict/{self.model}", dump_graph(_minimal(graph)), key=self.key)
        known = {"patient_id", "score", "risk", "risk_group", "folds", "local"}
        extra = {k: v for k, v in r.items() if k not in known and not k.startswith("fold")}
        if "n_slides" in extra:
            extra["n_slides"] = len(getattr(graph, "slideOrder", []) or [1])
        return Prediction(patient=str(getattr(graph, "id", "")), score=r["score"], risk=r["risk"], group=r.get("risk_group"),
                          folds=r.get("folds", []), local=None if r.get("local") is None else np.asarray(r["local"], np.float32),
                          extra=extra)


def _minimal(graph):
    """Only what the model needs: features, edges and slide boundaries. No patient or slide IDs, no coordinates."""
    from torch_geometric.data import Data

    g = Data(x=graph.x, edge_index=graph.edge_index)
    if getattr(graph, "slide_ptr", None) is not None:
        g.slide_ptr = graph.slide_ptr
    return g
