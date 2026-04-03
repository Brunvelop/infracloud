# LTX-Video con infracloud

Guía completa para lanzar un servidor de generación de vídeo en la nube (Vast.ai) y descargar los vídeos en local.

El servidor expone una API HTTP mínima: el vídeo generado se devuelve directamente como bytes en la respuesta, sin almacenamiento en el cloud. Todo queda en tu máquina.

---

## Requisitos previos

- Python 3.11+ e `infracloud` instalado (`uv pip install -e .`)
- Cuenta en [Vast.ai](https://vast.ai) con crédito cargado
- API key de Vast.ai → guárdala en `.env`:

```bash
cp .env.example .env
# edita .env y añade:
# VAST_API_KEY=tu_api_key
```

O expórtala directamente:

```bash
export VAST_API_KEY=tu_api_key
```

---

## 1. Lanzar el servidor

```bash
infracloud up ltx-video
```

Esto hace automáticamente:
1. Busca la GPU más barata con ≥32GB VRAM y CUDA ≥12.9 en Vast.ai
2. Crea la instancia y espera a que arranque
3. Instala las dependencias Python en el servidor
4. Descarga el modelo `Lightricks/LTX-Video` (~15GB desde HuggingFace)
5. Lanza un servidor FastAPI en el puerto 5000
6. Espera hasta que el servidor responda correctamente al health check

**Tiempo típico: 7–10 minutos** (la mayor parte es la descarga del modelo).

Salida esperada:

```
🔍 Buscando GPU con ≥32GB VRAM y CUDA ≥12.9...
✓ Encontrada: RTX 4080S · $0.20/hr · 32GB VRAM
🚀 Creando instancia...
  ID de instancia: 34018141
⏳ Esperando a que la instancia arranque...
⏳ Esperando a que el servidor esté listo... (esto puede tardar varios minutos)
✓ Servidor listo!

  URL:  http://175.155.64.174:19528
  SSH:  ssh -p 18140 root@ssh3.vast.ai
  Cost: $0.20/hr
```

---

## 2. Generar vídeos en local

Una vez el servidor está listo, usa `curl` para generar un vídeo y descargarlo directamente a tu máquina. El endpoint devuelve el MP4 en el cuerpo de la respuesta.

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
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 49,
    "width": 768,
    "height": 512,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  }' \
  -o video.mp4
```

### Referencia de parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `prompt` | `string` | **requerido** | Descripción del vídeo a generar |
| `negative_prompt` | `string` | ruido, blur, etc. | Qué evitar en la generación |
| `num_frames` | `int` | `97` | Nº de frames. Debe ser `8k+1`: **25, 49, 97, 193**… |
| `width` | `int` | `768` | Ancho en píxeles |
| `height` | `int` | `512` | Alto en píxeles |
| `num_inference_steps` | `int` | `40` | Pasos de difusión. Más = mejor calidad, más lento |
| `guidance_scale` | `float` | `7.5` | Adherencia al prompt. Rango típico: 3.5–7.5 |
| `seed` | `int\|null` | `null` | Semilla para reproducibilidad |

### Recetas rápidas

**Prueba rápida (~1 min, calidad básica):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a dog running on the beach", "num_frames": 25, "num_inference_steps": 20}' \
  -o prueba.mp4
```

**Calidad estándar (~2–3 min):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a sunset over the ocean, golden hour, cinematic", "num_frames": 49, "num_inference_steps": 30, "seed": 1234}' \
  -o sunset.mp4
```

**Máxima calidad (~5–8 min):**
```bash
curl -X POST $(infracloud url)/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a futuristic city at night, neon lights, rain, blade runner style", "num_frames": 97, "num_inference_steps": 50, "guidance_scale": 7.5}' \
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
    -d "{\"prompt\": \"${prompts[$i]}\", \"num_frames\": 49, \"num_inference_steps\": 30, \"seed\": $i}" \
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

with httpx.Client(timeout=300) as client:
    response = client.post(
        f"{url}/generate",
        json={
            "prompt": "a cat walking on the moon, cinematic, 4k",
            "num_frames": 49,
            "num_inference_steps": 30,
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
# {"status": "ready", "model": "ltx-video"}

# Respuesta cuando aún carga el modelo:
# {"status": "loading", "model": "ltx-video"}  (HTTP 503)
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

| GPU | VRAM | Precio típico | Tiempo arranque | Tiempo por vídeo (49 frames, 30 steps) |
|---|---|---|---|---|
| RTX 4080S | 32 GB | ~$0.17–0.20/hr | ~7 min | ~2 min |
| RTX 4090 | 24 GB | ~$0.35–0.50/hr | ~7 min | ~1.5 min |
| RTX A6000 | 48 GB | ~$0.50–0.80/hr | ~7 min | ~2 min |

> Los precios varían según disponibilidad. Consulta [cloud.vast.ai](https://cloud.vast.ai).

---

## Resolución de problemas

### El servidor no arranca en 15 minutos
El timeout de health check es de 15 minutos. Si se supera:
1. La instancia puede seguir corriendo — verifica en [cloud.vast.ai](https://cloud.vast.ai)
2. Destruye la instancia manualmente o con `infracloud down`
3. Relanza con `infracloud up ltx-video`

### El vídeo generado está vacío o tiene errores
- Verifica que el servidor está listo: `curl $(infracloud url)/health`
- Comprueba que `num_frames` sea `8k+1` (25, 49, 97, 193…)
- Aumenta `num_inference_steps` para mejor calidad

### No hay GPU disponible con los requisitos
```bash
# Reducir requisito de VRAM (mínimo para LTX-Video: 16GB con bfloat16)
infracloud up ltx-video --vram 24
```
