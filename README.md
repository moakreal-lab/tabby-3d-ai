# Tabby 3D AI 1.0

This project is a fully working Python pipeline that turns a single image into a depth map, textured point cloud, and 3D mesh. Upload an image in the browser and download the generated assets.

## Features
- Depth estimation with MiDaS (no training required)
- Textured point cloud generation with Open3D
- Mesh reconstruction via Poisson surface reconstruction with color transfer
- Simple FastAPI-powered website

## Setup

### Windows (PowerShell or Command Prompt)
```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> If you see “Defaulting to user installation,” it's not an error — it just means Python doesn’t have write access to the global site-packages. The install still works.

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Then open `http://localhost:8000`.

## Notes
- The first request will download the MiDaS model weights.
- CPU works fine; GPU will be used automatically if available. The default model targets low VRAM (~2 GB).
- The pipeline auto-resizes large images and downsamples the point cloud to keep processing fast on low-spec hardware.
