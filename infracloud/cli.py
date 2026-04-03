"""
infracloud CLI — Launch and manage GPU servers on Vast.ai.

Commands:
  up <stack>   Launch a GPU server (built-in stack name or path to .yaml)
  down         Destroy the active instance
  status       Show status of the active instance
  url          Print the server URL (for use in scripts)
  ssh          Open an SSH session to the active instance
  code         Open the active instance in VS Code via Remote-SSH

Examples:

    infracloud up ltx-video
    infracloud up ./my-stack.yaml
    curl $(infracloud url)/generate -d '{"prompt": "a cat"}'
    infracloud ssh
    infracloud down
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import click

from infracloud.state import load_state


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _require_state() -> dict:
    """Return the active instance state, or exit with an error if none exists."""
    state = load_state()
    if state is None:
        click.echo("✗ No hay instancia activa.", err=True)
        sys.exit(1)
    return state


def _build_url(state: dict) -> str:
    """Build the primary HTTP URL from state (public_ip + first api_port)."""
    host = state.get("public_ip") or state["ssh_host"]  # fallback for old state
    first_port = next(iter(state["api_ports"].values()))
    return f"http://{host}:{first_port}"


def _format_uptime(created_at_iso: str) -> str:
    """Return a human-readable uptime string from an ISO 8601 timestamp."""
    try:
        created_at = datetime.fromisoformat(created_at_iso)
        # Ensure tz-aware for comparison
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created_at
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return "desconocido"


# ─── CLI Group ────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="infracloud", prog_name="infracloud")
def cli() -> None:
    """infracloud — gestión de servidores GPU en Vast.ai."""


# ─── up ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("stack")
@click.option(
    "--vram",
    type=int,
    default=None,
    help="Override mínimo de VRAM en GB (ej: --vram 48).",
)
@click.option(
    "--disk",
    type=int,
    default=None,
    help="Override de disco en GB (ej: --disk 100).",
)
@click.option(
    "--offer",
    type=int,
    default=None,
    help=(
        "ID de oferta Vast.ai a usar directamente (salta la búsqueda automática). "
        "Útil para seleccionar una máquina específica desde la UI de Vast.ai."
    ),
)
def up(stack: str, vram: int | None, disk: int | None, offer: int | None) -> None:
    """Lanza un servidor GPU y espera a que esté listo.

    STACK puede ser el nombre de un stack built-in (ej: ltx-video) o la ruta
    a un archivo YAML que define un stack personalizado (ej: ./mi-stack.yaml).

    Para usar una oferta específica de Vast.ai (por ejemplo elegida desde la UI):

        infracloud up ltx-video --offer 12345678
    """
    from infracloud.cloud import InfraCloud
    from infracloud.stack import Stack

    overrides: dict = {}
    if vram is not None:
        overrides["gpu_vram_gb"] = vram
    if disk is not None:
        overrides["disk_gb"] = disk

    # If it looks like a file path, load via Stack.from_yaml()
    stack_arg: Stack | str
    if stack.endswith(".yaml") or stack.endswith(".yml") or os.path.exists(stack):
        try:
            stack_arg = Stack.from_yaml(stack)
        except FileNotFoundError:
            click.echo(f"✗ Archivo no encontrado: {stack}", err=True)
            sys.exit(1)
        except Exception as exc:
            click.echo(f"✗ Error al cargar el stack desde YAML: {exc}", err=True)
            sys.exit(1)
    else:
        stack_arg = stack  # built-in name; InfraCloud will resolve it

    try:
        cloud = InfraCloud()
        server = cloud.up(stack_arg, offer_id=offer, **overrides)
    except RuntimeError as exc:
        click.echo(f"\n✗ {exc}", err=True)
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    click.echo("")
    click.echo(f"  URL:  {server.url}")
    click.echo(f"  SSH:  {server.ssh_command}")
    click.echo(f"  Cost: ${server.cost_per_hr:.2f}/hr")


# ─── down ─────────────────────────────────────────────────────────────────────


@cli.command()
def down() -> None:
    """Destruye la instancia activa y limpia el estado local."""
    from infracloud.cloud import InfraCloud

    state = load_state()
    if state is None:
        click.echo("✗ No hay instancia activa.", err=True)
        sys.exit(1)

    try:
        InfraCloud().down()
    except RuntimeError as exc:
        click.echo(f"✗ {exc}", err=True)
        sys.exit(1)

    click.echo("✓ Instancia destruida.")


# ─── status ───────────────────────────────────────────────────────────────────


@cli.command()
def status() -> None:
    """Muestra el estado de la instancia activa."""
    from infracloud.cloud import InfraCloud

    info = InfraCloud().status()
    if info is None:
        click.echo("No hay instancia activa.")
        return

    url = _build_url(info)
    ssh_host = info.get("ssh_host", "")
    ssh_port = info.get("ssh_port", "")
    ssh_cmd = f"ssh -p {ssh_port} root@{ssh_host}"
    cost = info.get("cost_per_hr", 0.0)
    uptime = _format_uptime(info.get("created_at", ""))
    actual_status = info.get("actual_status", "running")
    gpu_name = info.get("gpu_name", "")

    # Pad label width for alignment
    col = 10
    click.echo(f"  {'Stack:':<{col}} {info.get('stack_name', '')}")
    if gpu_name:
        click.echo(f"  {'GPU:':<{col}} {gpu_name}")
    click.echo(f"  {'Status:':<{col}} {actual_status}")
    click.echo(f"  {'URL:':<{col}} {url}")
    click.echo(f"  {'SSH:':<{col}} {ssh_cmd}")
    click.echo(f"  {'Cost:':<{col}} ${cost:.2f}/hr")
    click.echo(f"  {'Uptime:':<{col}} {uptime}")


# ─── url ──────────────────────────────────────────────────────────────────────


@cli.command()
def url() -> None:
    """Imprime la URL del servidor activo (sin decoración, ideal para scripts).

    Ejemplo:\n
        curl $(infracloud url)/generate -d '{"prompt": "a cat"}'
    """
    state = _require_state()
    click.echo(_build_url(state))


# ─── ssh ──────────────────────────────────────────────────────────────────────


@cli.command()
def ssh() -> None:
    """Abre una sesión SSH interactiva en la instancia activa."""
    state = _require_state()
    host = state["ssh_host"]
    port = str(state["ssh_port"])
    # Replace the current process with SSH so the terminal works correctly
    os.execvp("ssh", ["ssh", "-p", port, f"root@{host}"])


# ─── code ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--dir",
    "remote_dir",
    default="/workspace",
    show_default=True,
    help="Directorio remoto a abrir en VS Code.",
)
def code(remote_dir: str) -> None:
    """Abre la instancia activa en VS Code usando Remote-SSH."""
    state = _require_state()
    host = state["ssh_host"]
    port = str(state["ssh_port"])
    # VS Code Remote-SSH URI format: ssh-remote+user@host:port
    remote = f"ssh-remote+root@{host}:{port}"
    os.execvp("code", ["code", "--remote", remote, remote_dir])
