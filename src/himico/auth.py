"""API keys for the HIMICO model server at Warwick (and for self-hosted `himico serve` servers).

Model weights for INSIGHT (and later SPARROW and other HIMICO models) are not distributed. Preprocessing runs on your
own hardware; only patch features and graph edges (no images, coordinates or IDs) are sent to the server, which
returns the predictions. Requests carry an API key: free for academic research, licensed for commercial use.

    himico login                      # paste your key once; stored in ~/.config/himico/credentials.json (mode 600)
    export HIMICO_API_KEY=...         # or set it per job (clusters, containers); takes precedence over the file
    export HIMICO_SERVER=https://...  # use another server than the default
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_SERVER = "https://insight.dcs.warwick.ac.uk/api/himico"   # interim host; the dedicated server will replace it
KEY_URL = "https://github.com/pkeller00/HIMICO-SDK#api-key"     # how to request a key (until the server has a sign-up page)
CONTACT = "Piotr.Keller@warwick.ac.uk"


def server_url() -> str:
    """The hosted HIMICO server, unless HIMICO_SERVER points elsewhere."""
    return (os.environ.get("HIMICO_SERVER") or DEFAULT_SERVER).rstrip("/")


def _credentials_path() -> Path:
    if os.environ.get("HIMICO_CONFIG_DIR"):
        return Path(os.environ["HIMICO_CONFIG_DIR"]) / "credentials.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "himico" / "credentials.json"


def _load() -> dict[str, str]:
    p = _credentials_path()
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def api_key(server: str | None = None) -> str | None:
    """Key for `server`: HIMICO_API_KEY, then the key saved by `himico login` for that server, then the legacy
    INSIGHT_SERVER_TOKEN."""
    env = os.environ.get("HIMICO_API_KEY")
    if env:
        return env.strip()
    saved = _load().get((server or server_url()).rstrip("/"))
    return saved or os.environ.get("INSIGHT_SERVER_TOKEN") or None


def save_key(key: str, server: str | None = None) -> Path:
    p = _credentials_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    creds = _load()
    creds[(server or server_url()).rstrip("/")] = key.strip()
    tmp = p.with_suffix(".tmp")
    tmp.touch(mode=0o600)
    os.chmod(tmp, 0o600)
    tmp.write_text(json.dumps(creds, indent=1))
    tmp.replace(p)
    return p


def forget_key(server: str | None = None) -> bool:
    creds = _load()
    if creds.pop((server or server_url()).rstrip("/"), None) is None:
        return False
    _credentials_path().write_text(json.dumps(creds, indent=1))
    return True


def missing_key_message(server: str) -> str:
    return (f"no API key for {server}. Request one at {KEY_URL} (free for academic research; commercial licences: "
            f"{CONTACT}), then run `himico login` or set HIMICO_API_KEY.")
