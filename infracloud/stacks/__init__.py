"""
Registry of built-in infracloud stacks.

Built-in stacks can be referenced by name in the CLI or Python API:

    infracloud up ltx-video
    cloud.up("ltx-video")

External projects can register their own stacks at runtime:

    from infracloud.stacks import register
    from infracloud.stack import Stack

    register(Stack(name="my-stack", image="...", ...))
"""

from __future__ import annotations

from infracloud.stack import Stack

BUILTIN_STACKS: dict[str, Stack] = {}


def register(stack: Stack) -> None:
    """Add a Stack to the built-in registry.

    Args:
        stack: The Stack instance to register. Its ``name`` attribute is used
               as the lookup key.
    """
    BUILTIN_STACKS[stack.name] = stack


def get(name: str) -> Stack | None:
    """Look up a built-in stack by name.

    Args:
        name: The stack name, e.g. ``"ltx-video"``.

    Returns:
        The matching :class:`~infracloud.stack.Stack`, or ``None`` if not found.
    """
    return BUILTIN_STACKS.get(name)


# ── Register built-in stacks ──────────────────────────────────────────────────
# Imports are done here (bottom of file) to avoid circular imports, since each
# stack module imports `register` from this file.

from infracloud.stacks import ltx_video as _ltx_video  # noqa: E402, F401
from infracloud.stacks import comfyui as _comfyui      # noqa: E402, F401
