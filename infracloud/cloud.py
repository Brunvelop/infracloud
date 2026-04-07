"""
InfraCloud — Vast.ai cloud client.

Orchestrates the full lifecycle of a GPU instance on Vast.ai:
search → create → wait for running → wait for healthy → return Server handle.

Usage (Python):

    from infracloud import InfraCloud

    cloud = InfraCloud()
    server = cloud.up("ltx-2.3-fp8-distilled")   # blocks until server is ready
    print(server.url)                              # http://ssh5.vast.ai:38291
    server.down()                    # destroy and clean up

Usage (CLI):

    infracloud up ltx-2.3-fp8-distilled
    infracloud url
    infracloud down
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass

import click
import httpx
from dotenv import load_dotenv
from vastai import VastAI

from infracloud.stack import Stack

# ─── Logging: redact secrets from vastai SDK debug output ────────────────────
# The vastai SDK uses the root logger to dump full kwargs (including api_key)
# and full API responses (including tokens) at DEBUG level.
# This filter intercepts those records and replaces secret values with ***.

class _SecretRedactFilter(logging.Filter):
    """Redact API keys and tokens from log records before they are emitted.

    Targets patterns produced by the vastai SDK at DEBUG level, e.g.:
        Calling show__instances with arguments: kwargs={'api_key': "'abc123'", ...}
        └-> [{'jupyter_token': '...', 'extra_env': [['HF_TOKEN', '...'], ...], ...}]

    The SDK uses repr() on values, so secrets always appear wrapped in single quotes.
    """

    # Each pattern captures the key part (group 1) and matches the secret value.
    # Substitution keeps group 1 and replaces the value with '***'.
    _PATTERNS = [
        # 'api_key': "'<long-value>'"  — kwargs repr format from the SDK
        re.compile(r"('api_key'\s*:\s*)'[^']{8,}'"),
        # 'jupyter_token': '<long-value>'
        re.compile(r"('jupyter_token'\s*:\s*)'[^']{8,}'"),
        # HF_TOKEN inside extra_env lists: ['HF_TOKEN', 'hf_...']
        re.compile(r"('HF_TOKEN'\s*,\s*)'[^']{8,}'"),
        # Generic fallback: any key ending in token/key/secret with a long value
        re.compile(r"('[^']*(?:token|key|secret)[^']*'\s*:\s*)'[^']{16,}'", re.I),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            for pat in self._PATTERNS:
                msg = pat.sub(r"\g<1>'***'", msg)
            record.msg = msg
            record.args = None
        except Exception:
            pass
        return True


logging.getLogger().addFilter(_SecretRedactFilter())

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
        stack_name:   Name of the deployed Stack (e.g. "ltx-2.3-fp8-distilled").
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
    is_ready: bool = False

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
        """Destroy this instance."""
        cloud = InfraCloud()
        cloud._vastai.destroy_instance(id=self.instance_id)


# ─── InfraCloud ───────────────────────────────────────────────────────────────

class InfraCloud:
    """Orchestrates GPU instances on Vast.ai.

    Typical flow::

        cloud = InfraCloud()
        server = cloud.up("ltx-2.3-fp8-distilled")
        # ... do work ...
        cloud.down()

    Args:
        api_key: Vast.ai API key. If not provided, reads ``VAST_API_KEY``
                 from the environment (or from a ``.env`` file).
    """

    LABEL_PREFIX = "infracloud:"
    _ACTIVE_STATUSES = frozenset({"running", "loading", "creating", "provisioning"})

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("VAST_API_KEY")
        self._vastai = VastAI(api_key=resolved_key, raw=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        stack: Stack | str,
        offer_id: int | None = None,
        **overrides,
    ) -> int:
        """Create a GPU instance and return immediately (non-blocking).

        Does **not** wait for the instance to be running or healthy.
        The caller should poll :meth:`status` to track progress.

        Args:
            stack:      A :class:`Stack` instance, or the name of a built-in
                        stack (e.g. ``"ltx-2.3-fp8-distilled"``).
            offer_id:   Optional Vast.ai offer ID to use directly, skipping
                        the automatic search.
            **overrides: Optional field overrides applied to the resolved Stack
                         before launching (e.g. ``gpu_vram_gb=48``).

        Returns:
            The Vast.ai instance ID.

        Raises:
            RuntimeError: If no offer is found or instance creation fails.
        """
        stack = self._resolve_stack(stack, overrides)

        if offer_id is not None:
            offer = self._fetch_offer_by_id(offer_id)
        else:
            offer = self._find_offer(stack)

        instance_id = self._create_instance(stack, offer)
        return instance_id

    def up(
        self,
        stack: Stack | str,
        offer_id: int | None = None,
        **overrides,
    ) -> Server:
        """Launch a GPU server and block until it's healthy (CLI use).

        Calls :meth:`start` then polls until the instance is running and
        the health endpoint responds HTTP 200.

        Args:
            stack:      A :class:`Stack` instance, or the name of a built-in
                        stack (e.g. ``"ltx-2.3-fp8-distilled"``).
            offer_id:   Optional Vast.ai offer ID to use directly, skipping
                        the automatic search.
            **overrides: Optional field overrides applied to the resolved Stack
                         before launching (e.g. ``gpu_vram_gb=48``).

        Returns:
            A :class:`Server` with all connection details and ``is_ready=True``.

        Raises:
            RuntimeError: If no suitable GPU offer is found, instance creation
                          fails, or the server doesn't become healthy in time.
        """
        # Resolve stack here so we have it for the health URL after start()
        resolved = self._resolve_stack(stack, overrides)

        # 1. Create instance (non-blocking)
        instance_id = self.start(resolved, offer_id=offer_id)
        click.echo(f"  ID de instancia: {instance_id}")

        # 2. Poll until Vast.ai reports status = "running"
        click.echo("⏳ Esperando a que la instancia arranque...")
        instance = self._wait_for_running(instance_id)

        # 3. Build Server from live instance data
        server = self._instance_to_server(instance)

        # 4. Health poll — use public_ip for direct HTTP access (not SSH proxy)
        health_port = next(iter(server.api_ports.values()), None)
        if health_port is not None:
            click.echo(
                "⏳ Esperando a que el servidor esté listo... "
                "(esto puede tardar varios minutos)"
            )
            self._wait_for_health(server.public_ip, health_port, resolved.health_url)

        server.is_ready = True
        click.echo("✓ Servidor listo!")
        return server

    def down(self) -> None:
        """Destroy the active infracloud instance.

        Queries Vast.ai to find the instance by label, then destroys it.
        Works regardless of local state.

        Raises:
            RuntimeError: If no active infracloud instance is found.
        """
        instance = self._find_active_instance()
        if instance is None:
            raise RuntimeError("No active infracloud instance found.")
        self._vastai.destroy_instance(id=instance["id"])

    def status(self) -> Server | None:
        """Return a Server handle for the active infracloud instance, or None.

        Queries the Vast.ai API directly — no local state file needed.
        Performs a health check to determine if the server is ready.

        Returns:
            :class:`Server` with ``is_ready=True/False``, or ``None`` if no
            active instance is found.
        """
        instance = self._find_active_instance()
        if instance is None:
            return None
        server = self._instance_to_server(instance)
        server.is_ready = self._check_health(server)
        return server

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_stack(self, stack: Stack | str, overrides: dict) -> Stack:
        """Resolve *stack* to a :class:`Stack` instance, applying any overrides.

        After loading the stack, ``repo_url`` is injected from the
        ``INFRACLOUD_REPO_URL`` environment variable if the stack doesn't
        already define one.  This keeps the repo URL out of committed config
        while still allowing it to be set per-machine via a ``.env`` file.
        """
        if isinstance(stack, str):
            resolved = Stack.get(stack)
            if resolved is None:
                available = Stack.list_available()
                hint = (
                    f" Stacks disponibles: {', '.join(available)}."
                    if available
                    else " No se encontraron stacks built-in en el directorio stacks/."
                )
                raise RuntimeError(
                    f"Stack desconocido: {stack!r}.{hint}"
                )
        else:
            resolved = stack

        # Inject repo_url from env if the stack doesn't define one.
        # Required by build_onstart() for onstart_mode="uv" stacks.
        if resolved.repo_url is None:
            repo_url = os.environ.get("INFRACLOUD_REPO_URL")
            if repo_url:
                from dataclasses import replace  # noqa: PLC0415
                resolved = replace(resolved, repo_url=repo_url)

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

        # Label format: "infracloud:ltx-2.3-fp8-distilled" — lets _find_active_instance()
        # identify our instances without relying on local state.json.
        label = f"{self.LABEL_PREFIX}{stack.name}"

        result = self._vastai.create_instance(
            id=int(offer["id"]),
            # When the stack ships a template_hash (Vast.ai official images with
            # @vastai-automatic-tag), let the template resolve the image tag so
            # Vast.ai can pick the right CUDA variant for the target host.
            # In that case we pass image=None so the template takes over.
            image=None if stack.template_hash else stack.image,
            disk=float(stack.disk_gb),
            onstart_cmd=stack.build_onstart() or None,
            env=env_str,
            ssh=True,
            direct=True,
            template_hash=stack.template_hash,
            label=label,
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

            status = instance.get("actual_status") or ""
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
    def _parse_vastai_ports(ports_raw: dict) -> dict[str, int]:
        """Parse Vast.ai Docker-style port mappings into {internal: external}.

        Input format (from Vast.ai API)::

            {"5000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "38291"}], "22/tcp": [...]}

        Output format::

            {"5000": 38291}

        Excludes port 22 (SSH). Only includes TCP ports with valid mappings.
        Works without a Stack — suitable for recovery and status queries.
        """
        result: dict[str, int] = {}
        for port_key, mappings in ports_raw.items():
            if "/tcp" not in port_key:
                continue
            internal = port_key.split("/")[0]
            if internal == "22":
                continue
            if mappings and isinstance(mappings, list):
                try:
                    result[internal] = int(mappings[0]["HostPort"])
                except (KeyError, ValueError, TypeError, IndexError):
                    pass
        return result

    @staticmethod
    def _extract_ports(instance: dict, stack: Stack) -> dict[str, int]:
        """Parse port mappings from the Vast.ai instance response.

        Delegates to ``_parse_vastai_ports()`` for the raw parsing, then adds
        a fallback for stack ports that don't appear in the mapping (useful
        in tests or when direct mode assigns the same port).

        Returns a dict mapping internal port (as str) → external port (int),
        e.g. ``{"5000": 38291}``.
        """
        all_ports = InfraCloud._parse_vastai_ports(instance.get("ports") or {})
        result: dict[str, int] = {}
        for internal_port in stack.ports:
            key = str(internal_port)
            if key in all_ports:
                result[key] = all_ports[key]
            else:
                result[key] = internal_port  # fallback: assume external == internal
        return result

    def _find_active_instance(self) -> dict | None:
        """Query Vast.ai for the active infracloud instance.

        Searches all instances on the account, filters by:

        - label starts with ``LABEL_PREFIX`` (``"infracloud:"``)
        - ``actual_status`` is in ``_ACTIVE_STATUSES``

        Returns the first matching instance dict, or ``None``.
        """
        try:
            instances = self._vastai.show_instances()
            if not isinstance(instances, list):
                return None
            for inst in instances:
                if not isinstance(inst, dict):
                    continue
                label = inst.get("label") or ""
                status = inst.get("actual_status") or ""
                if label.startswith(self.LABEL_PREFIX) and status in self._ACTIVE_STATUSES:
                    return inst
            return None
        except Exception:
            return None

    def _instance_to_server(self, instance: dict, is_ready: bool = False) -> Server:
        """Convert a raw Vast.ai instance dict to a :class:`Server` dataclass."""
        label = instance.get("label") or ""
        stack_name = (
            label.removeprefix(self.LABEL_PREFIX)
            if label.startswith(self.LABEL_PREFIX)
            else "unknown"
        )

        api_ports = self._parse_vastai_ports(instance.get("ports") or {})
        ssh_port_raw = instance.get("ssh_port")

        return Server(
            instance_id=instance.get("id"),
            stack_name=stack_name,
            ssh_host=instance.get("ssh_host") or "",
            ssh_port=int(ssh_port_raw) if ssh_port_raw else 0,
            public_ip=instance.get("public_ipaddr") or "",
            api_ports=api_ports,
            cost_per_hr=float(instance.get("dph_total") or 0.0),
            gpu_name=instance.get("gpu_name") or "",
            is_ready=is_ready,
        )

    def _check_health(self, server: Server) -> bool:
        """Single health check attempt. Returns ``True`` if server responds HTTP 200."""
        if not server.api_ports:
            return False
        try:
            url = f"{server.url}/health"
            resp = httpx.get(url, timeout=5.0)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False
