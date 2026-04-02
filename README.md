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
# 🔍 Buscando GPU con ≥24GB VRAM...
# ✓ Encontrada: RTX 4090 · $0.35/hr · 24GB VRAM
# 🚀 Creando instancia...
# ⏳ Esperando a que la instancia arranque...
# ⏳ Esperando a que el servidor esté listo... (esto puede tardar varios minutos)
# ✓ Servidor listo!
#
#   URL:  http://ssh5.vast.ai:38291
#   SSH:  ssh -p 34567 root@ssh5.vast.ai
#   Cost: $0.35/hr

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

## Custom Stacks

Cualquier proyecto puede definir su propio stack sin depender de Python,
usando un archivo YAML con los mismos campos que el dataclass `Stack`.

```yaml
# mi-comfyui.yaml
name: my-comfyui
image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04
gpu_vram_gb: 24
disk_gb: 50
ports:
  - 8188
onstart: |
  #!/bin/bash
  set -e
  source /venv/main/bin/activate
  pip install --quiet comfyui
  cd /workspace
  python -m comfyui --port 8188 --listen 0.0.0.0
health_url: /
```

```bash
infracloud up ./mi-comfyui.yaml
```

```python
from infracloud import InfraCloud
from infracloud.stack import Stack

server = InfraCloud().up(Stack.from_yaml("./mi-comfyui.yaml"))
```

### Campos del Stack

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `name` | `str` | — | Identificador del stack |
| `image` | `str` | — | Docker image |
| `gpu_vram_gb` | `int` | `24` | VRAM mínimo requerido (GB) |
| `disk_gb` | `int` | `50` | Disco a asignar (GB) |
| `ports` | `list[int]` | `[5000]` | Puertos expuestos en el contenedor |
| `onstart` | `str` | `""` | Script bash que se ejecuta al arrancar |
| `health_url` | `str` | `"/health"` | Path HTTP para el health check |
| `health_port` | `int\|None` | `None` | Puerto para health check (default: primer puerto) |
| `env` | `dict` | `{}` | Variables de entorno adicionales |

---

## Stacks built-in

| Nombre | Modelo | VRAM | Puerto | Endpoints |
|---|---|---|---|---|
| `ltx-video` | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | 24 GB | 5000 | `GET /health`, `POST /generate` |

### `ltx-video` — Generación de vídeo

```bash
# Generar con parámetros por defecto
curl $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walking on the moon"}' \
  -o video.mp4

# Parámetros disponibles:
# {
#   "prompt": "...",                   (requerido)
#   "negative_prompt": "...",          (default: ruido, blur, etc.)
#   "num_frames": 97,                  (debe ser 8k+1: 25, 49, 97...)
#   "width": 768,
#   "height": 512,
#   "num_inference_steps": 40,
#   "guidance_scale": 7.5,
#   "seed": null                       (int para reproducibilidad)
# }
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
     │  1. search_offers(vram≥24GB)     │                               │
     │  2. create_instance(            │                               │
     │     image, onstart, ports)       │──── crea instancia ─────────► │
     │                                  │                               │ onstart:
     │  3. poll: status == "running"?   │                               │  pip install ...
     │  4. poll: GET /health → 200?     │                               │  descarga modelo
     │◄──────────── ✓ ready ────────────│◄──────────────────────────── │  lanza FastAPI
     │                                  │                               │
     │  URL: http://host:38291          │                               │
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

Copia `.env.example` a `.env` y rellena tu API key:

```bash
cp .env.example .env
# edita .env y añade: VAST_API_KEY=tu_api_key
```

O expórtala directamente:

```bash
export VAST_API_KEY=tu_api_key
```

---

## Licencia

MIT
