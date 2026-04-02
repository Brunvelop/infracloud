"""
Stack — the deployment contract for infracloud.

A Stack describes WHAT runs on a GPU server. InfraCloud uses it to know
which Docker image to use, how much GPU/disk to request, what script to run
on startup, and how to verify the server is ready to accept requests.

Stacks can be:
  - Built-in: defined in infracloud/stacks/ (e.g. ltx_video)
  - Custom YAML: any project can define its own stack in a .yaml file
  - Programmatic: constructed inline as a Python object

Example (programmatic):

    from infracloud.stack import Stack

    my_stack = Stack(
        name="my-server",
        image="vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04",
        gpu_vram_gb=24,
        disk_gb=50,
        ports=[5000],
        onstart=\"\"\"
            #!/bin/bash
            source /venv/main/bin/activate
            pip install fastapi uvicorn
            python /workspace/serve.py &
        \"\"\",
        health_url="/health",
    )

Example (from YAML file):

    stack = Stack.from_yaml("./my-stack.yaml")

    # my-stack.yaml:
    # name: my-server
    # image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04
    # gpu_vram_gb: 24
    # disk_gb: 50
    # ports: [5000]
    # onstart: |
    #   #!/bin/bash
    #   pip install fastapi uvicorn
    #   python /workspace/serve.py &
    # health_url: /health
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stack:
    """Describes a deployable workload for a GPU server on Vast.ai.

    Attributes:
        name:         Human-readable identifier (e.g. "ltx-video", "comfyui").
        image:        Docker image to use. Prefer vastai/base-image variants as
                      they are pre-cached on Vast.ai hosts for faster startup.
        gpu_vram_gb:  Minimum GPU VRAM in GB required by the workload.
        disk_gb:      Disk space to allocate in GB (model weights + deps).
        ports:        List of ports the server listens on inside the container.
                      Vast.ai will map these to random external ports.
        onstart:      Bash script that runs automatically when the instance
                      boots. Use this to install dependencies, download models,
                      and launch your server process.
        health_url:   HTTP path to poll to determine if the server is ready.
                      infracloud will GET http://host:port{health_url} until it
                      receives a 200 response (or times out).
        health_port:  Which port from `ports` to use for the health check.
                      Defaults to the first port in `ports` if not set.
        env:          Extra environment variables to set on the instance.
    """

    name: str
    image: str
    gpu_vram_gb: int = 24
    disk_gb: int = 50
    ports: list[int] = field(default_factory=lambda: [5000])
    onstart: str = ""
    health_url: str = "/health"
    health_port: int | None = None
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Stack.name cannot be empty")
        if not self.image:
            raise ValueError("Stack.image cannot be empty")
        if not self.ports:
            raise ValueError("Stack.ports must contain at least one port")
        if self.gpu_vram_gb < 1:
            raise ValueError("Stack.gpu_vram_gb must be >= 1")
        if self.disk_gb < 1:
            raise ValueError("Stack.disk_gb must be >= 1")

    @property
    def effective_health_port(self) -> int:
        """The port used for health checks (defaults to first port in ports)."""
        return self.health_port if self.health_port is not None else self.ports[0]

    @classmethod
    def from_yaml(cls, path: str) -> Stack:
        """Load a Stack from a YAML file.

        This allows any external project to define a stack without writing
        Python — just a simple YAML file with the same field names.

        Args:
            path: Path to the YAML file (absolute or relative to cwd).

        Returns:
            A Stack instance populated from the YAML data.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If required fields are missing or invalid.
        """
        import yaml  # optional import — only needed when using YAML stacks

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at the top level in {path!r}")

        # Allow 'ports' to be a single int in YAML for convenience
        if "ports" in data and isinstance(data["ports"], int):
            data["ports"] = [data["ports"]]

        # Allow 'env' to be absent in YAML
        data.setdefault("env", {})

        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"Stack(name={self.name!r}, image={self.image!r}, "
            f"gpu_vram_gb={self.gpu_vram_gb}, ports={self.ports})"
        )
