# LTX-2.3 con infracloud

Guía completa para lanzar un servidor de generación de vídeo + audio en la nube (Vast.ai) y descargar los vídeos en local.

El servidor expone una API HTTP mínima: el vídeo generado se devuelve directamente como bytes en la respuesta, sin almacenamiento en el cloud. Todo queda en tu máquina.

**Modelo:** [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — 22B parámetros, genera vídeo + audio sincronizado. Usa la versión distilled (8 pasos), FP8 quantization, spatial upsampler ×2 y Gemma 3 12B como text encoder.

---

## Requisitos previos

- Python 3.11+ e `infracloud` instalado (`uv pip install -e .`)
- Cuenta en [Vast.ai](https://vast.ai) con crédito cargado
- Token de [Hugging Face](https://huggingface.co/settings/tokens) con acceso a [gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) (requiere aceptar términos del modelo)

Copia `.env.example` y rellena las tres variables:

```bash
cp .env.example .env
# edita .env:
# VAST_API_KEY=tu_api_key_de_vastai
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# INFRACLOUD_REPO_URL=https://github.com/tu-usuario/infracloud.git
```

> **`INFRACLOUD_REPO_URL`** es necesario para que el onstart script clone el repo en la instancia remota.
> **`HF_TOKEN`** es necesario para descargar Gemma 3 12B desde Hugging Face (modelo con acceso restringido).

---

## 1. Lanzar el servidor

```bash
infracloud up ltx-video
```

Esto hace automáticamente:
1. Busca la GPU más barata con ≥48 GB VRAM y CUDA ≥12.8 en Vast.ai
2. Crea la instancia con imagen Docker pinada (`vastai/pytorch:2.7.1-cuda-12.8.1-py311-24.04-2026-03-26`) para reproducibilidad total
3. Instala [uv](https://docs.astral.sh/uv/) en la instancia
4. Clona el repositorio infracloud en `/workspace/infracloud`
5. Ejecuta `uv sync --frozen` en `stacks/ltx-video/` — instala **todas las dependencias pinadas** del `uv.lock` (xformers, flashpack, ltx-core, ltx-pipelines y el resto) directamente en el venv pre-instalado con torch+CUDA
6. Descarga los checkpoints desde Hugging Face:
   - Modelo distilled 22B (`ltx-2.3-22b-distilled.safetensors`, ~50 GB)
   - Spatial upsampler ×2 (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`)
   - Gemma 3 12B text encoder (`google/gemma-3-12b-it-qat-q4_0-unquantized`, ~25 GB)
7. Lanza un servidor FastAPI en el puerto 5000
8. Espera hasta que el servidor responda correctamente al health check

**Tiempo típico: 30–45 minutos** (la mayor parte es la descarga de los modelos ~75 GB).

Salida esperada:

```
🔍 Buscando GPU con ≥48GB VRAM y CUDA ≥12.8...
✓ Encontrada: RTX 5880Ada · $0.34/hr · 48GB VRAM
🚀 Creando instancia...
  ID de instancia: 34145024
⏳ Esperando a que la instancia arranque...
⏳ Esperando a que el servidor esté listo... (esto puede tardar varios minutos)
✓ Servidor listo!

  URL:  http://175.155.64.145:19633
  SSH:  ssh -p 25024 root@ssh2.vast.ai
  Cost: $0.34/hr
```

---

## 2. Generar vídeos en local

Una vez el servidor está listo, usa `curl` para generar un vídeo y descargarlo directamente a tu máquina. El endpoint devuelve el MP4 (con audio sincronizado) en el cuerpo de la respuesta.

### Ejemplo básico

```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walking on the moon, cinematic"}' \
  -o video.mp4
```

### Con todos los parámetros

```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat walking on the moon, cinematic, 4k",
    "duration": 5.0,
    "width": 1536,
    "height": 1024,
    "enhance_prompt": true,
    "seed": 42
  }' \
  -o video.mp4
```

### Referencia de parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `prompt` | `string` | **requerido** | Descripción del vídeo a generar |
| `duration` | `float` | `3.0` | Duración en segundos (el frame count se calcula automáticamente como `8k+1`) |
| `width` | `int` | `1536` | Ancho en píxeles (se redondea al múltiplo de 32 más cercano) |
| `height` | `int` | `1024` | Alto en píxeles (se redondea al múltiplo de 32 más cercano) |
| `enhance_prompt` | `bool` | `true` | Expande el prompt usando Gemma antes de la generación |
| `seed` | `int\|null` | `null` | Semilla para reproducibilidad. Aleatorio si `null` |

> **Nota:** LTX-2.3 usa el modelo distilled (8 pasos, CFG=1), por lo que no hay parámetros de `num_inference_steps` ni `guidance_scale`.

### Recetas rápidas

**Prueba rápida (~1 min, 768×512, 3s):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a dog running on the beach", "duration": 3.0, "width": 768, "height": 512}' \
  -o prueba.mp4
```

**Calidad estándar (~4–6 min, 1536×1024, 3s):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a sunset over the ocean, golden hour, cinematic", "duration": 3.0, "seed": 1234}' \
  -o sunset.mp4
```

**Duración larga (~8–15 min, 768×512, 20s):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "abstract dreamscape, surreal geometric shapes morphing into silhouettes, deep indigo and crimson, hypnotic",
    "duration": 20.0,
    "width": 768,
    "height": 512,
    "enhance_prompt": true
  }' \
  --max-time 900 \
  -o largo.mp4
```

**Máxima calidad (~10–15 min, 1536×1024, 5s):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a futuristic city at night, neon lights, rain, blade runner style", "duration": 5.0}' \
  --max-time 900 \
  -o ciudad.mp4
```

### Generar varios vídeos en serie

```bash
#!/bin/bash
URL=$(infracloud url)

prompts=(
  "a horse galloping through a forest"
  "a rocket launching into space"
  "waves crashing on rocks, slow motion"
)

for i in "${!prompts[@]}"; do
  echo "Generando vídeo $((i+1))..."
  curl -s -X POST "$URL/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${prompts[$i]}\", \"duration\": 3.0, \"seed\": $i}" \
    -o "video_${i}.mp4"
  echo "  Guardado: video_${i}.mp4"
done
```

---

## 3. Generar vídeos desde Python

```python
import httpx
import subprocess

# Obtener la URL del servidor activo
url = subprocess.check_output(["infracloud", "url"]).decode().strip()

with httpx.Client(timeout=900) as client:  # timeout generoso: modelo grande
    response = client.post(
        f"{url}/generate",
        json={
            "prompt": "a cat walking on the moon, cinematic, 4k",
            "duration": 3.0,
            "width": 1536,
            "height": 1024,
            "enhance_prompt": True,
            "seed": 42,
        },
    )
    response.raise_for_status()

with open("video.mp4", "wb") as f:
    f.write(response.content)

print(f"Vídeo guardado: {len(response.content) / 1024:.0f} KB")
```

---

## 4. Verificar que el servidor está listo

```bash
# Health check manual
curl $(infracloud url)/health

# Respuesta cuando está listo:
# {"status": "ready", "model": "ltx-2.3"}

# Respuesta cuando aún carga el modelo:
# {"status": "loading", "model": "ltx-2.3"}  (HTTP 503)
```

---

## 5. Ver estado y URL

```bash
infracloud status   # muestra GPU, coste, URL, uptime
infracloud url      # solo la URL (útil en scripts)
```

---

## 6. Apagar el servidor

```bash
infracloud down
```

> ⚠️ **Importante:** Las instancias de Vast.ai se facturan por hora mientras están activas. Ejecuta `infracloud down` cuando termines para dejar de pagar.

---

## Tiempos y costes de referencia

Medidos en despliegues reales con `duration=3.0, width=768, height=512`:

| GPU | VRAM | Precio típico | Tiempo arranque* | Tiempo por vídeo (3s, 768×512) |
|---|---|---|---|---|
| RTX 5880 Ada | 48 GB | ~$0.34/hr | ~35–45 min | ~53s |
| RTX A6000 | 48 GB | ~$0.50–0.80/hr | ~30–40 min | ~3–5 min |
| RTX 6000 Ada | 48 GB | ~$0.80–1.20/hr | ~30–40 min | ~2–4 min |
| A100 SXM | 80 GB | ~$1.50–2.50/hr | ~30–40 min | ~1–2 min |
| H100 PCIe | 80 GB | ~$2.00–3.50/hr | ~30–40 min | ~1 min |

> \* El tiempo de arranque incluye la descarga de ~75 GB de modelos (LTX-2.3 + Gemma). Depende del ancho de banda de la instancia.
> Los precios varían según disponibilidad. Consulta [cloud.vast.ai](https://cloud.vast.ai).

---

## Resolución de problemas

### El health check expira antes de que el servidor esté listo

`infracloud up` espera 15 minutos en el health check. La descarga de modelos (~75 GB) puede tardar 35–45 minutos dependiendo del ancho de banda de la instancia. Si el comando falla con timeout:

1. **El servidor puede seguir arrancando.** Comprueba si ya está listo:
   ```bash
   # Obtén la IP y puerto de la instancia desde cloud.vast.ai → tu instancia → "Connect"
   curl http://<IP>:<PUERTO>/health
   ```
2. Si responde `{"status":"ready"}`, ya puedes usarlo directamente con `curl`
3. Para futuros despliegues, elige instancias con buen ancho de banda en la UI de Vast.ai

### Error `libcudart.so.13: cannot open shared object file`

Este error ocurría con `torchaudio==2.11.0` (CUDA 13) en hosts con CUDA 12.x. Está corregido en el lock file actual (`torchaudio==2.7.0`). Si ves este error, asegúrate de tener el código actualizado con `git pull` antes de relanzar.

### El vídeo generado está vacío o hay un error 503
- El servidor aún puede estar cargando el modelo. Espera y vuelve a intentarlo
- Verifica el estado: `curl $(infracloud url)/health`

### No hay GPU disponible con los requisitos
```bash
# Prueba con una GPU de 40 GB (rendimiento reducido, puede haber OOM)
infracloud up ltx-video --vram 40
```

### Error de VRAM durante la generación
LTX-2.3 con FP8 + spatial upsampler a 1536×1024 necesita ~45–48 GB.
Si hay OOM, reduce la resolución en la petición:

```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "...", "width": 768, "height": 512}' \
  -o video.mp4
```

### Depurar en la instancia remota

Conecta por SSH y monitoriza el proceso de arranque directamente:

```bash
# Conectar a la instancia
infracloud ssh

# Una vez dentro, ver el estado del servidor
ps aux | grep python
curl localhost:5000/health

# Ver si los modelos están descargados
ls -lh /root/.cache/huggingface/hub/

# Lanzar el servidor manualmente si falló
source /venv/main/bin/activate
cd /workspace/infracloud/stacks/ltx-video
python serve.py
```
