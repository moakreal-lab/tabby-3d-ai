# Tabby 3D AI 1.0

This project is a fully working Python pipeline that turns a single image into a depth map, textured point cloud, and 3D mesh. Run it locally on any machine (CPU or GPU) with a simple CLI or via a local API.

## Features
- Depth estimation with MiDaS (no training required)
- Textured point cloud generation with Open3D
- Mesh reconstruction via Poisson surface reconstruction with color transfer
- FastAPI endpoint for local automation

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

## Run (CLI)

```bash
python main.py --image path/to/photo.jpg
```

Optional: choose a different output directory.

```bash
python main.py --image path/to/photo.jpg --output-dir outputs
```

## Run (API)

```bash
python main.py --serve
```

Then POST an image to `http://localhost:8000/api/process`. The response includes `estimated_seconds` and `elapsed_seconds`.

## Notes
- The first request will download the MiDaS model weights.
- CPU works fine; GPU will be used automatically if available. The default model targets low VRAM (~2 GB).
- The pipeline auto-resizes large images and downsamples the point cloud to keep processing fast on low-spec hardware.
