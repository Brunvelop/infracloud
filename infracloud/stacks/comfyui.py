"""
Built-in stack: ComfyUI

Uses Vast.ai's official pre-built ComfyUI image (vastai/comfy), which includes:
  - ComfyUI pre-installed at /ComfyUI
  - CUDA + PyTorch ready
  - ComfyUI Manager for downloading models from the UI

Access once running:
  http://<URL>:8188   -> ComfyUI web interface (open in browser)

Models are NOT downloaded automatically — use the ComfyUI Manager in the UI
or add model download commands to a custom onstart script.

Default models directory inside the container: /ComfyUI/models/

Example custom stack with model pre-download:

    from infracloud.stack import Stack
    from infracloud.stacks.comfyui import comfyui

    from dataclasses import replace

    my_comfyui = replace(comfyui, onstart=comfyui.onstart + '''
    # Download SDXL base model
    wget -q -O /ComfyUI/models/checkpoints/sdxl_base.safetensors \\
      https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/...
    ''')
"""

from infracloud.stack import Stack
from infracloud.stacks import register

# ---- onstart script ----------------------------------------------------------
#
# The vastai/comfy image has ComfyUI pre-installed. This script simply ensures
# it starts on the right interface and port.
#
# ComfyUI auto-start behavior varies by Vast.ai template version, so we always
# start it explicitly to be safe.

_ONSTART = r"""#!/bin/bash
set -e

echo "[infracloud] Starting ComfyUI..."

# Kill any existing ComfyUI process (in case the image auto-started one)
pkill -f "main.py" 2>/dev/null || true
sleep 2

# Start ComfyUI listening on all interfaces
cd /ComfyUI
nohup python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    > /var/log/comfyui.log 2>&1 &

echo "[infracloud] ComfyUI started. Logs at /var/log/comfyui.log"
echo "[infracloud] Models directory: /ComfyUI/models/"
"""

# ---- Stack definition --------------------------------------------------------

comfyui = Stack(
    name="comfyui",
    # Official Vast.ai ComfyUI image — pre-built, no install needed
    image="vastai/comfy:latest",
    gpu_vram_gb=32,    # 32 GB minimum — required for LTX-Video and comfortable for SDXL/Flux
    disk_gb=100,       # Models can be large: SDXL ~7GB, Flux ~24GB, etc.
    ports=[8188],
    onstart=_ONSTART,
    health_url="/queue",   # ComfyUI returns {"queue_running": [], "queue_pending": []}
)

register(comfyui)
