"""
State management for infracloud.

Persists the active instance's metadata to ~/.infracloud/state.json so that
commands like `infracloud status`, `infracloud down`, and `infracloud url`
work correctly across different terminal sessions.

The state file is a plain JSON object — easy to inspect manually:

    cat ~/.infracloud/state.json

Example state:

    {
      "instance_id": 12345678,
      "stack_name": "ltx-video",
      "ssh_host": "ssh5.vast.ai",
      "ssh_port": 34567,
      "api_ports": {"5000": 38291},
      "cost_per_hr": 0.35,
      "created_at": "2025-04-02T10:00:00Z"
    }
"""

from __future__ import annotations

import json
from pathlib import Path

STATE_DIR = Path.home() / ".infracloud"
STATE_FILE = STATE_DIR / "state.json"


def save_state(data: dict) -> None:
    """Persist instance metadata to ~/.infracloud/state.json.

    Creates ~/.infracloud/ if it does not exist. Overwrites any existing state
    (there is only ever one active instance at a time).

    Args:
        data: Dict containing instance metadata. Expected keys:
              instance_id, stack_name, ssh_host, ssh_port, api_ports,
              cost_per_hr, created_at.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w") as f:
        json.dump(data, f, indent=2)


def load_state() -> dict | None:
    """Read the active instance state from ~/.infracloud/state.json.

    Returns:
        The parsed state dict, or None if no state file exists (meaning there
        is no active instance).
    """
    if not STATE_FILE.exists():
        return None
    with STATE_FILE.open() as f:
        return json.load(f)


def clear_state() -> None:
    """Remove ~/.infracloud/state.json.

    Safe to call even if the file does not exist — will not raise an error.
    """
    STATE_FILE.unlink(missing_ok=True)
