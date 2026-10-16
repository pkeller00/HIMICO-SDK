"""Check, before a run, that this SDK is the latest version offered by the HIMICO server.

The server's `/ping` reports `latest_sdk`. If this install is older, the run stops with the upgrade command, so a
cohort is never processed with outdated code. An unreachable server never blocks (the run fails later with its own
message if it needs the server). Skip with `himico run --no-update-check` or HIMICO_NO_UPDATE_CHECK=1.
"""
from __future__ import annotations

import os
import re

from . import __version__
from .auth import SDK_URL


class UpdateRequired(RuntimeError):
    pass


def _v(s: str) -> tuple[int, ...]:
    n = [int(p) for p in re.findall(r"\d+", s)[:3]]
    return tuple(n + [0] * (3 - len(n)))


def check(server: str | None) -> None:
    """Raise UpdateRequired if `server` offers a newer SDK than this one."""
    if not server or os.environ.get("HIMICO_NO_UPDATE_CHECK"):
        return
    from .models.remote import ping

    try:
        latest = ping(server).get("latest_sdk")
    except Exception:
        return
    if latest and _v(str(latest)) > _v(__version__):
        raise UpdateRequired(
            f"a newer HIMICO SDK is available ({__version__} -> {latest}). Upgrade first, then run again:\n"
            f"    pip install --upgrade --force-reinstall --no-deps git+{SDK_URL}\n"
            "(to run anyway: --no-update-check or HIMICO_NO_UPDATE_CHECK=1)")
