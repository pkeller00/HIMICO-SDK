"""`insight serve`: host INSIGHT models behind a small REST API.

Run it on a GPU machine that holds the weights; collaborators run the heavy preprocessing themselves and send only
patient graphs, so the weights never leave the server.

    insight serve --host 0.0.0.0 --port 8080          # set INSIGHT_SERVER_TOKEN to require a bearer token

    GET  /ping                        {"status": "ok", "models": [...], "device": ...}
    GET  /models                      model names, descriptions and required features
    POST /predict/<model>             body: .g graph bytes  ->  prediction JSON (risk, group, folds, local)

Graphs are opened with the restricted loader in insight.safe_graph, never stored and never logged. One request runs
at a time per model. Without a token the API is open to anyone who can reach the port, so either bind to localhost
(the default) or set INSIGHT_SERVER_TOKEN.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import __version__
from .registry import get_model, models
from .safe_graph import MAX_BYTES, GraphError, load_graph
from .wsi import device_for

log = logging.getLogger("insight")


class _State:
    device = None
    weights_dir = None
    token = None
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


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"insight/{__version__}"

    def _send(self, code, body):
        data = json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self) -> bool:
        if not _State.token:
            return True
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got, f"Bearer {_State.token}")

    def do_GET(self):
        if not self._authorised():
            return self._send(401, {"error": "missing or wrong token"})
        if self.path == "/ping":
            return self._send(200, {"status": "ok", "version": __version__, "device": str(_State.device),
                                    "models": sorted(models()), "loaded": sorted(_State.loaded)})
        if self.path == "/models":
            return self._send(200, {n: {"description": c.description, "features": c.features, "dthresh": c.dthresh,
                                        "paper": c.paper} for n, c in models().items()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorised():
            return self._send(401, {"error": "missing or wrong token"})
        if not self.path.startswith("/predict/"):
            return self._send(404, {"error": "not found"})
        name = self.path[len("/predict/"):]
        if name not in models():
            return self._send(404, {"error": f"unknown model {name!r}"})
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
          preload: list[str] | None = None):
    _State.device = device_for(device)
    _State.weights_dir = weights_dir
    _State.token = os.environ.get("INSIGHT_SERVER_TOKEN") or None
    for name in preload or []:
        _State.model(name)
    if host not in ("127.0.0.1", "localhost") and not _State.token:
        log.warning("serving on %s without INSIGHT_SERVER_TOKEN: anyone who can reach this port can use the models", host)
    log.info("INSIGHT server %s on http://%s:%d (device %s; models %s)", __version__, host, port, _State.device, sorted(models()))
    ThreadingHTTPServer((host, port), _Handler).serve_forever()
