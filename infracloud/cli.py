"""
infracloud CLI — Launch and manage GPU servers on Vast.ai.

Commands:
  list         List available built-in stacks
  up <stack>   Launch a GPU server (built-in stack name or path to .yaml)
  down         Destroy the active instance
  status       Show status of the active instance
  url          Print the server URL (for use in scripts)
  ssh          Open an SSH session to the active instance
  code         Open the active instance in VS Code via Remote-SSH

Examples:

    infracloud list
    infracloud up ltx-2.3-fp8-distilled
    infracloud up ./my-stack.yaml
    curl $(infracloud url)/generate -d '{"prompt": "a cat"}'
    infracloud ssh
    infracloud down
"""

from __future__ import annotations

import os
import sys

import click


# ─── CLI Group ────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="infracloud", prog_name="infracloud")
def cli() -> None:
    """infracloud — gestión de servidores GPU en Vast.ai."""


# ─── list ─────────────────────────────────────────────────────────────────────


@cli.command(name="list")
def list_command() -> None:
    """Lista los stacks built-in disponibles."""
    from infracloud.stack import Stack

    names = Stack.list_available()
    if not names:
        click.echo("No se encontraron stacks en el directorio stacks/.")
        return

    click.echo("Stacks disponibles:\n")
    # Compute column widths for alignment
    name_width = max(len(n) for n in names)
    for name in names:
        try:
            stack = Stack.get(name)
        except Exception:
            stack = None

        if stack is None:
            click.echo(f"  {name}")
            continue

        # Format port list: single port as plain int, multiple as "p1, p2"
        ports_str = ", ".join(str(p) for p in stack.ports)
        # Use the real image name even for template_hash stacks
        image = stack.image

        click.echo(
            f"  {name:<{name_width}}    "
            f"{stack.gpu_vram_gb}GB VRAM · "
            f"{image:<16} · "
            f"port {ports_str}"
        )


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
        "Machine ID de Vast.ai a usar directamente (el número m:XXXX que aparece "
        "en la UI). Salta la búsqueda automática. Ejemplo: --offer 37509"
    ),
)
def up(stack: str, vram: int | None, disk: int | None, offer: int | None) -> None:
    """Lanza un servidor GPU y espera a que esté listo.

    STACK puede ser el nombre de un stack built-in (ej: ltx-2.3-fp8-distilled) o la ruta
    a un archivo YAML que define un stack personalizado (ej: ./mi-stack.yaml).

    Para usar una oferta específica de Vast.ai (por ejemplo elegida desde la UI):

        infracloud up ltx-2.3-fp8-distilled --offer 12345678
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
    """Destruye la instancia activa."""
    from infracloud.cloud import InfraCloud

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

    server = InfraCloud().status()
    if server is None:
        click.echo("No hay instancia activa.")
        return

    col = 10
    click.echo(f"  {'Stack:':<{col}} {server.stack_name}")
    if server.gpu_name:
        click.echo(f"  {'GPU:':<{col}} {server.gpu_name}")
    click.echo(f"  {'Status:':<{col}} {'ready' if server.is_ready else 'booting'}")
    if server.api_ports:
        click.echo(f"  {'URL:':<{col}} {server.url}")
    click.echo(f"  {'SSH:':<{col}} {server.ssh_command}")
    click.echo(f"  {'Cost:':<{col}} ${server.cost_per_hr:.2f}/hr")


# ─── url ──────────────────────────────────────────────────────────────────────


@cli.command()
def url() -> None:
    """Imprime la URL del servidor activo (sin decoración, ideal para scripts).

    Ejemplo:\n
        curl $(infracloud url)/generate -d '{"prompt": "a cat"}'
    """
    from infracloud.cloud import InfraCloud

    server = InfraCloud().status()
    if server is None:
        click.echo("✗ No hay instancia activa.", err=True)
        sys.exit(1)
    if not server.api_ports:
        click.echo("✗ La instancia no tiene puertos API aún.", err=True)
        sys.exit(1)
    click.echo(server.url)


# ─── ssh ──────────────────────────────────────────────────────────────────────


@cli.command()
def ssh() -> None:
    """Abre una sesión SSH interactiva en la instancia activa."""
    from infracloud.cloud import InfraCloud

    server = InfraCloud().status()
    if server is None:
        click.echo("✗ No hay instancia activa.", err=True)
        sys.exit(1)
    # Replace the current process with SSH so the terminal works correctly
    os.execvp("ssh", ["ssh", "-p", str(server.ssh_port), f"root@{server.ssh_host}"])


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
    from infracloud.cloud import InfraCloud

    server = InfraCloud().status()
    if server is None:
        click.echo("✗ No hay instancia activa.", err=True)
        sys.exit(1)
    # VS Code Remote-SSH URI format: ssh-remote+user@host:port
    remote = f"ssh-remote+root@{server.ssh_host}:{server.ssh_port}"
    os.execvp("code", ["code", "--remote", remote, remote_dir])
