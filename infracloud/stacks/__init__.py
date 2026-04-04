"""
Stack discovery for infracloud.

Stacks are discovered automatically by scanning the ``stacks/`` directory at
the repository root. Any subdirectory that contains a ``pyproject.toml`` with
a ``[tool.infracloud]`` section is a valid stack:

    stacks/
      ltx-video/
        pyproject.toml   ← [tool.infracloud] metadata
        serve.py         ← entrypoint
        uv.lock          ← pinned dependencies
      comfyui/
        pyproject.toml   ← [tool.infracloud] metadata (onstart_mode = "custom")
        onstart.sh       ← custom boot script

Usage::

    infracloud up ltx-video    # discovers stacks/ltx-video/pyproject.toml
    infracloud up comfyui      # discovers stacks/comfyui/pyproject.toml

External stacks can also be passed directly as :class:`~infracloud.stack.Stack`
objects or YAML files — no registration needed.
"""

from __future__ import annotations

from pathlib import Path

from infracloud.stack import Stack

# ``stacks/`` lives at the repo root, three levels up from this file:
#   infracloud/stacks/__init__.py  →  ../../..  →  repo root
_STACKS_ROOT = Path(__file__).resolve().parent.parent.parent / "stacks"


def get(name: str) -> Stack | None:
    """Look up a stack by name (= directory name under ``stacks/``).

    Args:
        name: The stack name, e.g. ``"ltx-video"`` or ``"comfyui"``.

    Returns:
        A :class:`~infracloud.stack.Stack` loaded from
        ``stacks/{name}/pyproject.toml``, or ``None`` if the directory or
        ``pyproject.toml`` does not exist.
    """
    stack_dir = _STACKS_ROOT / name
    if not stack_dir.is_dir():
        return None
    if not (stack_dir / "pyproject.toml").exists():
        return None
    return Stack.from_dir(stack_dir)


def list_stacks() -> list[str]:
    """Return the names of all available built-in stacks.

    Scans ``stacks/`` for subdirectories that contain a ``pyproject.toml``.

    Returns:
        Sorted list of stack names (directory names), e.g.
        ``["comfyui", "ltx-video"]``.
    """
    if not _STACKS_ROOT.is_dir():
        return []
    return sorted(
        d.name
        for d in _STACKS_ROOT.iterdir()
        if d.is_dir() and (d / "pyproject.toml").exists()
    )
