"""
InfraCloud — Vast.ai cloud client.

Orchestrates the full lifecycle of a GPU instance on Vast.ai:
search → create → wait for running → wait for healthy → return Server handle.

Usage (Python):

    from infracloud import InfraCloud

    cloud = InfraCloud()
    server = cloud.up("ltx-video")   # blocks until server is ready
    print(server.url)                # http://ssh5.vast.ai:38291
    server.down()                    # destroy and clean up

Usage (CLI):

    infracloud up ltx-video
    infracloud url
    infracloud down
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import click
import httpx
from dotenv import load_dotenv
from vastai import VastAI

from infracloud.stack import Stack
from infracloud.state import clear_state, load_state, save_state

# Load .env if present — allows VAST_API_KEY to live in a .env file
load_dotenv()

# ─── Timeouts & polling intervals ────────────────────────────────────────────

_INSTANCE_POLL_INTERVAL = 5     # seconds between "is instance running?" polls
_INSTANCE_READY_TIMEOUT = 900   # 15 min max to go from created → running
_HEALTH_POLL_INTERVAL   = 10    # seconds between health-check requests
_HEALTH_READY_TIMEOUT   = 900   # 15 min max for the server to become healthy


# ─── Server ──────────────────────────────────────────────────────────────────

@dataclass
class Server:
    """Handle for an active GPU instance on Vast.ai.

    Returned by ``InfraCloud.up()``. Holds all the connection details needed
    to interact with the running workload.

    Attributes:
        instance_id:  Vast.ai instance ID.
        stack_name:   Name of the deployed Stack (e.g. "ltx-video").
        ssh_host:     Hostname for SSH / HTTP access (e.g. "ssh5.vast.ai").
        ssh_port:     SSH port on the host.
        api_ports:    Mapping of internal container port → external host port.
                      Keys are strings (e.g. ``"5000"``).
        cost_per_hr:  Current on-demand price in $/hr.
        gpu_name:     GPU model name (e.g. "RTX 4090").
    """

    instance_id: int
    stack_name: str
    ssh_host: str
    ssh_port: int
    public_ip: str
    api_ports: dict[str, int]
    cost_per_hr: float
    gpu_name: str

    @property
    def url(self) -> str:
        """Base URL of the first exposed port (direct public IP).

        Example: ``http://175.155.64.174:19260``
        """
        first_external = next(iter(self.api_ports.values()))
        return f"http://{self.public_ip}:{first_external}"

    @property
    def ssh_command(self) -> str:
        """Full SSH command to connect to the instance.

        Example: ``ssh -p 34567 root@ssh5.vast.ai``
        """
        return f"ssh -p {self.ssh_port} root@{self.ssh_host}"

    def down(self) -> None:
        """Destroy this instance and remove local state."""
        cloud = InfraCloud()
        cloud._vastai.destroy_instance(id=self.instance_id)
        clear_state()


# ─── InfraCloud ───────────────────────────────────────────────────────────────

class InfraCloud:
    """Orchestrates GPU instances on Vast.ai.

    Typical flow::

        cloud = InfraCloud()
        server = cloud.up("ltx-video")
        # ... do work ...
        cloud.down()

    Args:
        api_key: Vast.ai API key. If not provided, reads ``VAST_API_KEY``
                 from the environment (or from a ``.env`` file).
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("VAST_API_KEY")
        self._vastai = VastAI(api_key=resolved_key, raw=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def up(
        self,
        stack: Stack | str,
        offer_id: int | None = None,
        **overrides,
    ) -> Server:
        """Launch a GPU server and block until it's healthy.

        Args:
            stack:      A :class:`Stack` instance, or the name of a built-in
                        stack (e.g. ``"ltx-video"``).
            offer_id:   Optional Vast.ai offer ID to use directly, skipping the
                        automatic search. Useful when you've found a suitable
                        machine via the Vast.ai UI or ``search_offers`` yourself.
                        Example: ``cloud.up("ltx-video", offer_id=12345678)``
            **overrides: Optional field overrides applied to the resolved Stack
                         before launching (e.g. ``gpu_vram_gb=48``).

        Returns:
            A :class:`Server` with all connection details, ready to accept
            HTTP requests.

        Raises:
            RuntimeError: If no suitable GPU offer is found, instance creation
                          fails, or the server doesn't become healthy in time.
        """
        stack = self._resolve_stack(stack, overrides)

        # 1. Find cheapest matching GPU offer (or use the specified offer_id)
        if offer_id is not None:
            click.echo(f"🔍 Buscando oferta {offer_id}...")
            offer = self._fetch_offer_by_id(offer_id)
        else:
            offer = self._find_offer(stack)
        click.echo(
            f"✓ Encontrada: {offer['gpu_name']} · "
            f"${offer['dph_total']:.2f}/hr · "
            f"{offer['gpu_ram'] / 1024:.0f}GB VRAM"
        )

        # 2. Create instance
        click.echo("🚀 Creando instancia...")
        instance_id = self._create_instance(stack, offer)
        click.echo(f"  ID de instancia: {instance_id}")

        # 3. Poll until Vast.ai reports status = "running"
        click.echo("⏳ Esperando a que la instancia arranque...")
        instance = self._wait_for_running(instance_id)

        # 4. Extract connection details from instance metadata
        ssh_host = instance["ssh_host"]
        ssh_port = int(instance["ssh_port"])
        public_ip = instance["public_ipaddr"]   # direct access IP
        api_ports = self._extract_ports(instance, stack)
        cost_per_hr = float(instance.get("dph_total", 0.0))
        gpu_name = instance.get("gpu_name", "")

        # 5. Health poll — use public_ip for direct HTTP access (not SSH proxy)
        health_port = api_ports.get(str(stack.effective_health_port))
        if health_port is None:
            # fall back to first mapped port
            health_port = next(iter(api_ports.values()))

        click.echo(
            "⏳ Esperando a que el servidor esté listo... "
            "(esto puede tardar varios minutos)"
        )
        self._wait_for_health(public_ip, health_port, stack.health_url)

        # 6. Persist state and return handle
        state = {
            "instance_id": instance_id,
            "stack_name": stack.name,
            "ssh_host": ssh_host,
            "ssh_port": ssh_port,
            "public_ip": public_ip,
            "api_ports": api_ports,
            "cost_per_hr": cost_per_hr,
            "gpu_name": gpu_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)

        server = Server(
            instance_id=instance_id,
            stack_name=stack.name,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            public_ip=public_ip,
            api_ports=api_ports,
            cost_per_hr=cost_per_hr,
            gpu_name=gpu_name,
        )
        click.echo("✓ Servidor listo!")
        return server

    def down(self) -> None:
        """Destroy the active instance and remove local state.

        Raises:
            RuntimeError: If there is no active instance to destroy.
        """
        state = load_state()
        if state is None:
            raise RuntimeError("No hay instancia activa.")

        instance_id = state["instance_id"]
        self._vastai.destroy_instance(id=instance_id)
        clear_state()

    def status(self) -> dict | None:
        """Return the active instance's state, or None if no instance is active.

        Reads from the local state file and optionally verifies with Vast.ai
        that the instance is still alive.

        Returns:
            State dict (see :func:`infracloud.state.load_state`) or ``None``.
        """
        state = load_state()
        if state is None:
            return None

        # Try to enrich with live data from Vast.ai
        try:
            instance = self._vastai.show_instance(id=state["instance_id"])
            if isinstance(instance, dict):
                state["actual_status"] = instance.get("actual_status", "unknown")
        except Exception:
            state["actual_status"] = "unknown"

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_stack(self, stack: Stack | str, overrides: dict) -> Stack:
        """Resolve *stack* to a :class:`Stack` instance, applying any overrides."""
        if isinstance(stack, str):
            # Deferred import to avoid circular dependency (stacks → cloud)
            from infracloud.stacks import get as get_builtin  # noqa: PLC0415
            resolved = get_builtin(stack)
            if resolved is None:
                raise RuntimeError(
                    f"Stack desconocido: {stack!r}. "
                    "Usa el nombre de un stack built-in o pasa un objeto Stack."
                )
        else:
            resolved = stack

        if overrides:
            from dataclasses import replace  # noqa: PLC0415
            resolved = replace(resolved, **overrides)

        return resolved

    def _fetch_offer_by_id(self, machine_id: int) -> dict:
        """Fetch a Vast.ai offer by machine ID (the ``m:XXXX`` number in the UI).

        Note: Vast.ai distinguishes between *offer_id* (used internally to create
        instances) and *machine_id* (the physical host ID shown in the UI as
        ``m:XXXX``).  This method accepts the machine_id and returns the offer
        with the matching machine, ready to be passed to ``_create_instance``.

        Args:
            machine_id: The machine ID shown in the Vast.ai UI (e.g. 37509 for
                        ``m:37509``).  Passed via ``--offer`` on the CLI.

        Returns:
            The offer dict (same structure as returned by ``_find_offer``).

        Raises:
            RuntimeError: If no rentable offer is found for that machine ID.
        """
        offers = self._vastai.search_offers(
            query=f"machine_id = {machine_id}",
            order="dph_total",
            type="on-demand",
        )
        if not offers:
            raise RuntimeError(
                f"No se encontró ninguna oferta disponible para la máquina {machine_id} "
                f"(m:{machine_id}). Comprueba que la máquina está verificada y disponible "
                "para renta on-demand."
            )
        return offers[0]

    def _find_offer(self, stack: Stack) -> dict:
        """Search Vast.ai for the cheapest offer matching the stack's requirements.

        Note: the vastai-sdk ``parse_query`` function multiplies ``gpu_ram``
        values by 1000 (it expects GB, not MB).  ``disk_space`` is passed
        as-is in GB.  ``rentable = true`` is already included in the default
        query so we don't need to specify it.
        """
        disk_gb = stack.disk_gb

        cuda_msg = (
            f" y CUDA ≥{stack.min_cuda_ver}" if stack.min_cuda_ver else ""
        )
        click.echo(f"🔍 Buscando GPU con ≥{stack.gpu_vram_gb}GB VRAM{cuda_msg}...")

        # gpu_ram: pass in GB — the SDK multiplies by 1000 internally
        # disk_space: pass in GB directly
        # cuda_max_good: driver's max supported CUDA version (decimal, e.g. 12.9)
        query = (
            f"gpu_ram >= {stack.gpu_vram_gb} "
            f"disk_space >= {disk_gb} "
            f"reliability > 0.9 "
            f"verified = true"
        )
        if stack.min_cuda_ver is not None:
            query += f" cuda_max_good >= {stack.min_cuda_ver}"

        offers = self._vastai.search_offers(
            query=query,
            order="dph_total",   # cheapest first
            type="on-demand",
        )

        if not offers:
            raise RuntimeError(
                f"No se encontraron ofertas con ≥{stack.gpu_vram_gb}GB VRAM "
                f"y ≥{stack.disk_gb}GB de disco."
            )

        return offers[0]

    def _create_instance(self, stack: Stack, offer: dict) -> int:
        """Create an instance from *offer* and return the new instance ID."""
        # Build Docker-style env/port string
        env_parts: list[str] = []
        for port in stack.ports:
            env_parts.append(f"-p {port}:{port}/tcp")
        for k, v in stack.env.items():
            env_parts.append(f"-e {k}={v}")
        env_str = " ".join(env_parts) if env_parts else None

        result = self._vastai.create_instance(
            id=int(offer["id"]),
            # When the stack ships a template_hash (Vast.ai official images with
            # @vastai-automatic-tag), let the template resolve the image tag so
            # Vast.ai can pick the right CUDA variant for the target host.
            # In that case we pass image=None so the template takes over.
            image=None if stack.template_hash else stack.image,
            disk=float(stack.disk_gb),
            onstart_cmd=stack.onstart if stack.onstart else None,
            env=env_str,
            ssh=True,
            direct=True,
            template_hash=stack.template_hash,
        )

        # result is a Response object or dict; SDK returns r.json() when raw=True
        if isinstance(result, dict):
            data = result
        elif hasattr(result, "json"):
            data = result.json()
        else:
            raise RuntimeError(f"Respuesta inesperada de Vast.ai: {result!r}")

        # Vast.ai SDK sometimes returns success=False even when the instance
        # was created (new_contract is present). Accept both cases.
        if not data.get("success") and "new_contract" not in data:
            raise RuntimeError(f"Error al crear instancia: {data}")

        return int(data["new_contract"])

    def _wait_for_running(self, instance_id: int) -> dict:
        """Poll until the instance reaches status ``'running'``.

        Returns the instance dict once running.
        """
        deadline = time.monotonic() + _INSTANCE_READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                instance = self._vastai.show_instance(id=instance_id)
            except Exception:
                time.sleep(_INSTANCE_POLL_INTERVAL)
                continue

            if not isinstance(instance, dict):
                time.sleep(_INSTANCE_POLL_INTERVAL)
                continue

            status = instance.get("actual_status", "")
            if status == "running":
                return instance

            # Detect GPU / provisioning errors early — no need to wait 15 min
            if "error" in status.lower():
                raise RuntimeError(
                    f"La instancia {instance_id} entró en estado de error: "
                    f"'{status}'. La máquina puede estar averiada. "
                    "Prueba con otra oferta (--offer) o deja que la búsqueda "
                    "automática seleccione una diferente."
                )

            elapsed = time.monotonic() - (deadline - _INSTANCE_READY_TIMEOUT)
            click.echo(
                f"  Estado: {status or 'iniciando'} "
                f"({elapsed:.0f}s)...",
                err=True,
            )
            time.sleep(_INSTANCE_POLL_INTERVAL)

        raise RuntimeError(
            f"La instancia {instance_id} no llegó a 'running' "
            f"en {_INSTANCE_READY_TIMEOUT}s."
        )

    def _wait_for_health(
        self, host: str, port: int, path: str
    ) -> None:
        """Poll ``http://host:port{path}`` until it returns HTTP 200.

        Raises:
            RuntimeError: If the health endpoint doesn't respond 200
                          within ``_HEALTH_READY_TIMEOUT`` seconds.
        """
        url = f"http://{host}:{port}{path}"
        deadline = time.monotonic() + _HEALTH_READY_TIMEOUT

        with httpx.Client(timeout=10.0) as client:
            while time.monotonic() < deadline:
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass

                elapsed = time.monotonic() - (deadline - _HEALTH_READY_TIMEOUT)
                mins, secs = divmod(int(elapsed), 60)
                click.echo(
                    f"  Esperando health check... ({mins}m {secs:02d}s)",
                    err=True,
                )
                time.sleep(_HEALTH_POLL_INTERVAL)

        raise RuntimeError(
            f"El servidor no respondió en {url} dentro de "
            f"{_HEALTH_READY_TIMEOUT // 60} minutos."
        )

    @staticmethod
    def _extract_ports(instance: dict, stack: Stack) -> dict[str, int]:
        """Parse port mappings from the Vast.ai instance response.

        Vast.ai returns port mappings in ``instance["ports"]`` using a
        Docker-inspect-style format::

            {"5000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "38291"}]}

        Returns a dict mapping internal port (as str) → external port (int),
        e.g. ``{"5000": 38291}``.

        Falls back to the original port if the mapping cannot be found (useful
        in tests or when direct mode assigns the same port).
        """
        raw_ports: dict = instance.get("ports") or {}
        result: dict[str, int] = {}

        for internal_port in stack.ports:
            key = f"{internal_port}/tcp"
            mapping = raw_ports.get(key)
            if mapping and isinstance(mapping, list) and mapping:
                try:
                    result[str(internal_port)] = int(mapping[0]["HostPort"])
                    continue
                except (KeyError, ValueError, TypeError):
                    pass
            # fall back: assume external == internal
            result[str(internal_port)] = internal_port

        return result
