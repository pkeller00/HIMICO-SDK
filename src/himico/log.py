"""Logging: readable lines on the console, structured JSON lines in <workspace>/logs/."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

LOGGER = logging.getLogger("himico")


class _Json(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r.created)), "level": r.levelname, "msg": r.getMessage()}
        for k in ("slide", "patient", "stage", "seconds", "status"):
            if hasattr(r, k):
                d[k] = getattr(r, k)
        if r.exc_info:
            d["error"] = self.formatException(r.exc_info)
        return json.dumps(d)


def setup(workspace: Path | None = None, level: int = logging.INFO, console: bool = True) -> Path | None:
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h)
        h.close()
    if console:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
        LOGGER.addHandler(h)
    path = None
    if workspace is not None:
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        path = workspace / "logs" / f"run_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        fh = logging.FileHandler(path)
        fh.setFormatter(_Json())
        LOGGER.addHandler(fh)
    return path
