# infracloud

Lanza servidores GPU en la nube con un comando.

`infracloud` automatiza el ciclo de vida completo de instancias GPU en [Vast.ai](https://vast.ai): busca la oferta más barata, crea la instancia, espera a que el servidor esté listo y te devuelve la URL. Todo con un solo comando.

---

## Quick Start

```bash
# 1. Instalar
pip install .
# o con uv:
uv pip install -e .

# 2. Configurar API key de Vast.ai
#    Obtén tu key en: https://cloud.vast.ai/api/
export VAST_API_KEY=tu_api_key_aqui
# o cópiala en un archivo .env (ver .env.example)

# 3. Lanzar servidor de generación de vídeo
infracloud up ltx-video

# Salida esperada:
# 🔍 Buscando GPU con ≥48GB VRAM y CUDA ≥12.7...
# ✓ Encontrada: RTX A6000 · $0.65/hr · 48GB VRAM
# 🚀 Creando instancia...
# ⏳ Esperando a que la instancia arranque...
# ⏳ Esperando a que el servidor esté listo... (esto puede tardar varios minutos)
# ✓ Servidor listo!
#
#   URL:  http://175.155.64.174:19528
#   SSH:  ssh -p 18140 root@ssh3.vast.ai
#   Cost: $0.65/hr

# 4. Generar un vídeo
curl $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walking on the moon"}' \
  -o video.mp4

# 5. Apagar (y dejar de pagar)
infracloud down
```

---

## CLI Reference

| Comando | Descripción |
|---|---|
| `infracloud up <stack>` | Lanza un servidor GPU. `stack` puede ser un nombre built-in o una ruta a `.yaml`. |
| `infracloud down` | Destruye la instancia activa y limpia el estado local. |
| `infracloud status` | Muestra stack, GPU, URL, SSH, coste y uptime de la instancia activa. |
| `infracloud url` | Imprime solo la URL (sin decoración, ideal para scripts). |
| `infracloud ssh` | Abre una sesión SSH interactiva en la instancia. |
| `infracloud code` | Abre VS Code con Remote-SSH conectado a la instancia. |

### Opciones de `infracloud up`

```
infracloud up ltx-video --vram 48    # override: buscar GPU con ≥48GB VRAM
infracloud up ltx-video --disk 100   # override: solicitar 100GB de disco
infracloud up ./mi-stack.yaml        # stack personalizado desde archivo YAML
```

### `infracloud url` en scripts

```bash
# El comando solo imprime la URL, sin emojis ni texto extra:
curl $(infracloud url)/generate -d '{"prompt": "a sunset"}'

# Asignar la URL a una variable:
SERVER=$(infracloud url)
curl $SERVER/health
```

---

## Uso desde Python

```python
from infracloud import InfraCloud

cloud = InfraCloud()  # lee VAST_API_KEY del entorno

# Lanzar servidor (bloquea hasta que está listo)
server = cloud.up("ltx-video")

print(server.url)          # http://ssh5.vast.ai:38291
print(server.ssh_command)  # ssh -p 34567 root@ssh5.vast.ai
print(server.cost_per_hr)  # 0.35

# Hacer requests al servidor
import httpx
response = httpx.post(
    f"{server.url}/generate",
    json={"prompt": "a cat walking on the moon"},
    timeout=120,
)
with open("video.mp4", "wb") as f:
    f.write(response.content)

# Destruir la instancia cuando termines
server.down()
```

### Overrides al lanzar

```python
# Pedir más VRAM para resoluciones más altas
server = cloud.up("ltx-video", gpu_vram_gb=48, disk_gb=100)
```

---

## Crear un nuevo stack

La forma más robusta de definir un stack es crear un directorio dentro de `stacks/` con un `pyproject.toml` que incluya `[tool.infracloud]`. Las versiones de todas las dependencias quedan pinadas en un `uv.lock` commiteado.

```
stacks/
  mi-stack/
    pyproject.toml   ← deps + [tool.infracloud]
    serve.py         ← entrypoint (FastAPI u otro servidor)
    uv.lock          ← lockfile generado con `uv lock` (se commitea)
```

### 1. Crear el directorio y `pyproject.toml`

```toml
# stacks/mi-stack/pyproject.toml
[project]
name = "mi-stack"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.115.12",
    "uvicorn[standard]==0.34.3",
    # añade tus deps aquí
]

[tool.infracloud]
image = "vastai/pytorch"
template_hash = "b84ca276fa572e949cd7ff43ae5fe855"  # auto-selecciona tag CUDA
gpu_vram_gb = 24
disk_gb = 50
ports = [5000]
entrypoint = "serve.py"
health_url = "/health"
min_cuda_ver = 12.4

[tool.infracloud.env]
HF_TOKEN = "${HF_TOKEN}"   # se resuelve desde el entorno local en runtime
```

### 2. Crear el entrypoint

```python
# stacks/mi-stack/serve.py
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

### 3. Generar el lockfile

```bash
cd stacks/mi-stack
uv lock
git add uv.lock
```

### 4. Lanzar

```bash
infracloud up mi-stack
```

Infracloud descubre automáticamente el stack escaneando `stacks/mi-stack/pyproject.toml`. No hay que registrar nada.

### Campos de `[tool.infracloud]`

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `image` | `str` | — | Docker image a usar |
| `gpu_vram_gb` | `int` | `24` | VRAM mínima requerida (GB) |
| `disk_gb` | `int` | `50` | Disco a asignar (GB) |
| `ports` | `list[int]` | `[5000]` | Puertos expuestos en el contenedor |
| `entrypoint` | `str` | — | Script Python a ejecutar (ej: `"serve.py"`) |
| `health_url` | `str` | `"/health"` | Path HTTP para el health check |
| `health_port` | `int` | — | Puerto para health check (default: primer puerto) |
| `min_cuda_ver` | `float` | — | Versión mínima de CUDA requerida (ej: `12.7`) |
| `template_hash` | `str` | — | Hash de template Vast.ai para imágenes oficiales (`vastai/pytorch`, etc.) |
| `onstart_mode` | `str` | `"uv"` | `"uv"` para el flujo estándar, `"custom"` para usar `onstart.sh` |
| `repo_url` | `str` | — | URL del repo a clonar (override de `INFRACLOUD_REPO_URL`) |

### Stack personalizado desde YAML (método alternativo)

Para proyectos que no quieren usar el sistema de directorios:

```yaml
# mi-stack.yaml
name: my-server
image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04
gpu_vram_gb: 24
disk_gb: 50
ports:
  - 5000
onstart: |
  #!/bin/bash
  set -e
  source /venv/main/bin/activate
  pip install fastapi uvicorn
  python /workspace/serve.py
health_url: /health
```

```bash
infracloud up ./mi-stack.yaml
```

```python
from infracloud import InfraCloud
from infracloud.stack import Stack

server = InfraCloud().up(Stack.from_yaml("./mi-stack.yaml"))
```

---

## Stacks built-in

| Nombre | Modelo | VRAM | Puerto | Imagen base |
|---|---|---|---|---|
| `ltx-video` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — 22B | 48 GB | 5000 | `vastai/pytorch` |
| `comfyui` | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | 32 GB | 8188 | `vastai/comfy` |

### `ltx-video` — Generación de vídeo + audio

Modelo distilled de 22B parámetros con FP8 quantization. Genera vídeo con audio sincronizado.

```bash
infracloud up ltx-video

# Generar un vídeo (una vez el servidor esté listo)
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walking on the moon, cinematic"}' \
  -o video.mp4

# Parámetros disponibles:
# {
#   "prompt": "...",          (requerido)
#   "duration": 3.0,          (segundos; default 3.0)
#   "width": 1536,
#   "height": 1024,
#   "enhance_prompt": true,   (expande el prompt con Gemma 12B)
#   "seed": null              (int para reproducibilidad)
# }
```

> Ver la [guía completa](docs/ltx-video.md) para más detalles, recetas y resolución de problemas.

### `comfyui` — Interfaz web de generación

```bash
infracloud up comfyui

# Abrir la interfaz en el navegador
open $(infracloud url)
```

---

## Costes típicos en Vast.ai

Precios orientativos (varían según disponibilidad). Consulta [cloud.vast.ai](https://cloud.vast.ai) para precios actuales.

| GPU | VRAM | Precio típico | Uso recomendado |
|---|---|---|---|
| RTX 3090 | 24 GB | ~$0.20–0.30/hr | ltx-video, modelos SD |
| RTX 4090 | 24 GB | ~$0.30–0.50/hr | ltx-video, inferencia rápida |
| RTX A6000 | 48 GB | ~$0.50–0.80/hr | Modelos grandes, mayor resolución |
| A100 SXM | 80 GB | ~$1.50–2.50/hr | Entrenamiento, batch inference |
| H100 SXM | 80 GB | ~$2.50–4.00/hr | Máximo rendimiento |

> **Tip:** Usa `infracloud down` en cuanto termines. Las instancias se facturan por hora.

---

## Arquitectura

```
Desarrollador                        Vast.ai                      Instancia GPU
     │                                  │                               │
     │  infracloud up ltx-video         │                               │
     │─────────────────────────────►    │                               │
     │  1. search_offers(vram≥48GB)     │                               │
     │  2. create_instance(             │                               │
     │     image, onstart, ports)       │──── crea instancia ─────────►│
     │                                  │                               │ onstart:
     │  3. poll: status == "running"?   │                               │  uv install
     │  4. poll: GET /health → 200?     │                               │  git clone repo
     │◄──────────── ✓ ready ────────────│◄─────────────────────────────│  uv sync --frozen
     │                                  │                               │  descarga modelo
     │  URL: http://host:38291          │                               │  lanza FastAPI
     │                                  │                               │
     │  POST /generate {prompt: "..."}  │                               │
     │─────────────────────────────────────────────────────────────────►│
     │◄─────────────────── video.mp4 ──────────────────────────────────│
     │                                  │                               │
     │  infracloud down                 │                               │
     │─────────────────────────────►    │──── destroy ────────────────►│
     │  ✓ destruida                     │                               │
```

---

## Instalación

### Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip
- Cuenta en [Vast.ai](https://vast.ai) con crédito cargado

### Desde el repositorio

```bash
git clone https://github.com/tu-usuario/infracloud
cd infracloud
uv sync              # instala dependencias y crea .venv
uv run infracloud --help
```

### Como dependencia en tu proyecto

```bash
uv add git+https://github.com/tu-usuario/infracloud
# o:
pip install git+https://github.com/tu-usuario/infracloud
```

### Variables de entorno

Copia `.env.example` a `.env` y rellena los valores:

```bash
cp .env.example .env
```

| Variable | Descripción |
|---|---|
| `VAST_API_KEY` | API key de Vast.ai (obligatorio). Obtén la tuya en [cloud.vast.ai/api/](https://cloud.vast.ai/api/). |
| `INFRACLOUD_REPO_URL` | URL del repositorio infracloud a clonar en la instancia remota. Necesaria para stacks en modo `"uv"` (ej: `https://github.com/tu-usuario/infracloud.git`). |
| `HF_TOKEN` | Token de Hugging Face para modelos gated (ej: Gemma). Obtén el tuyo en [hf.co/settings/tokens](https://huggingface.co/settings/tokens). |

O expórtalas directamente:

```bash
export VAST_API_KEY=tu_api_key
export INFRACLOUD_REPO_URL=https://github.com/tu-usuario/infracloud.git
export HF_TOKEN=hf_xxx  # solo si usas modelos gated
```

---

## Licencia

MIT
