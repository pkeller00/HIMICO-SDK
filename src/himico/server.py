"""`himico serve`: host HIMICO models (INSIGHT, ...) behind a small REST API.

Run it on a GPU machine that holds the weights; users run the heavy preprocessing themselves and send only patient
graphs, so neither their slides nor the weights leave their machines. The hosted HIMICO server at Warwick speaks the
same protocol (see himico.models.remote).

    himico keys add "Jane Doe, Uni X" --tier academic       # prints a new API key once; stores only its hash
    himico serve --host 0.0.0.0 --port 8080 --keys keys.json

    GET  /ping                        {"status": "ok", "models": [...], "auth": true}       (no key needed)
    GET  /account                     {"name", "tier", "models"}  for the presented key
    GET  /models                      model names, descriptions and required features
    POST /predict/<model>             body: .g graph bytes  ->  prediction JSON (risk, group, folds, local)

Keys (Authorization: Bearer <key>) come from a keys file (HIMICO_SERVER_KEYS or --keys) of
{"<sha256 of key>": {"name", "tier", "models": [...] or null for all}}, or a single shared HIMICO_SERVER_TOKEN.
Quotas and billing belong in front of this server (the hosted service answers 429 when a quota is reached).

Graphs are opened with the restricted loader in himico.safe_graph, never stored and never logged. One request runs
at a time per model. Without keys the API is open to anyone who can reach the port, so either bind to localhost
(the default) or configure keys.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import __version__
from .registry import get_model, models
from .safe_graph import MAX_BYTES, GraphError, load_graph
from .wsi import device_for

log = logging.getLogger("himico")
TIERS = ("academic", "commercial", "collaborator", "internal")


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ── key file ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_keys(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def add_key(path: str | Path, name: str, tier: str = "academic", model_names: list[str] | None = None) -> str:
    """Create a key, store its hash in the keys file and return the key (shown once, never stored)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    keys = load_keys(path)
    key = "hmk_" + secrets.token_urlsafe(32)
    keys[_hash(key)] = {"name": name, "tier": tier, "models": model_names, "id": secrets.token_hex(4)}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.touch(mode=0o600)
    tmp.write_text(json.dumps(keys, indent=1))
    tmp.replace(p)
    return key


def revoke_key(path: str | Path, key_id: str) -> bool:
    keys = load_keys(path)
    left = {h: r for h, r in keys.items() if r.get("id") != key_id}
    if len(left) == len(keys):
        return False
    Path(path).write_text(json.dumps(left, indent=1))
    return True


# ── server ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class _State:
    device = None
    weights_dir = None
    token = None
    keys: dict = {}
    loaded: dict = {}
    locks: dict = {}
    guard = threading.Lock()

    @classmethod
    def model(cls, name):
        with cls.guard:
            if name not in cls.loaded:
                cls.loaded[name] = get_model(name)(cls.device, cls.weights_dir)
                cls.locks[name] = threading.Lock()
        return cls.loaded[name], cls.locks[name]

    @classmethod
    def auth_required(cls) -> bool:
        return bool(cls.token or cls.keys)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"himico/{__version__}"

    def _send(self, code, body):
        data = json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _account(self) -> dict | None:
        """The caller's key record, or None if the key is missing or wrong."""
        if not _State.auth_required():
            return {"name": "anonymous", "tier": "open", "models": None}
        got = self.headers.get("Authorization", "")
        if not got.startswith("Bearer "):
            return None
        key = got[len("Bearer "):].strip()
        if _State.token and hmac.compare_digest(key, _State.token):
            return {"name": "shared token", "tier": "internal", "models": None}
        return _State.keys.get(_hash(key))

    def do_GET(self):
        if self.path == "/ping":
            return self._send(200, {"status": "ok", "version": __version__, "device": str(_State.device),
                                    "models": sorted(models()), "auth": _State.auth_required()})
        acct = self._account()
        if acct is None:
            return self._send(401, {"error": "missing or invalid API key"})
        if self.path == "/account":
            return self._send(200, {k: acct.get(k) for k in ("name", "tier", "models")})
        if self.path == "/models":
            return self._send(200, {n: {"description": c.description, "features": c.features, "dthresh": c.dthresh,
                                        "paper": c.paper} for n, c in models().items()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        acct = self._account()
        if acct is None:
            return self._send(401, {"error": "missing or invalid API key"})
        if not self.path.startswith("/predict/"):
            return self._send(404, {"error": "not found"})
        name = self.path[len("/predict/"):]
        if name not in models():
            return self._send(404, {"error": f"unknown model {name!r}"})
        if acct.get("models") is not None and name not in acct["models"]:
            return self._send(403, {"error": f"your {acct.get('tier')} key does not include {name!r}"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self._send(413, {"error": "empty or too large upload"})
        raw = self.rfile.read(n)
        try:
            g = load_graph(raw)
            model, lock = _State.model(name)
            with lock:
                pred = model.predict(g)
            body = {**pred.row(), "folds": pred.folds,
                    "local": None if pred.local is None else np.round(pred.local, 5).tolist()}
            log.info("predict %s for key %s (%s)", name, acct.get("id", "-"), acct.get("tier"))
            return self._send(200, body)
        except GraphError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            log.exception("prediction failed")
            return self._send(500, {"error": f"prediction failed ({type(e).__name__})"})
        finally:
            del raw

    def log_message(self, fmt, *args):   # never log request contents
        pass


def serve(host: str = "127.0.0.1", port: int = 8080, device: str = "auto", weights_dir: str | None = None,
          preload: list[str] | None = None, keys: str | None = None):
    _State.device = device_for(device)
    _State.weights_dir = weights_dir
    _State.token = os.environ.get("HIMICO_SERVER_TOKEN") or os.environ.get("INSIGHT_SERVER_TOKEN") or None
    keys = keys or os.environ.get("HIMICO_SERVER_KEYS")
    if keys:
        _State.keys = load_keys(keys)
        log.info("%d API keys loaded from %s", len(_State.keys), keys)
    for name in preload or []:
        _State.model(name)
    if host not in ("127.0.0.1", "localhost") and not _State.auth_required():
        log.warning("serving on %s without API keys: anyone who can reach this port can use the models", host)
    log.info("HIMICO server %s on http://%s:%d (device %s; models %s)", __version__, host, port, _State.device, sorted(models()))
    ThreadingHTTPServer((host, port), _Handler).serve_forever()
