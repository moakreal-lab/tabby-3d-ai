# Image to 3D Pipeline

This project is a fully working Python pipeline that turns a single image into a depth map, point cloud, and 3D mesh. Upload an image in the browser and download the generated assets.

## Features
- Depth estimation with MiDaS (no training required)
- Point cloud generation with Open3D
- Mesh reconstruction via Poisson surface reconstruction
- Simple FastAPI-powered website

## Setup

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
- CPU works fine; GPU will be used automatically if available.
