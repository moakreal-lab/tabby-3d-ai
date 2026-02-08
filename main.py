import io
import os
import uuid
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Tabby 3D AI 1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "MiDaS_small"
MAX_INPUT_SIZE = 640
POINT_STRIDE = 4
VOXEL_DOWNSAMPLE_SIZE = 0.01
POISSON_DEPTH = 6


def load_midas_model():
    model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)
    model.to(DEVICE)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return model, transform


MODEL, TRANSFORM = load_midas_model()


def run_depth_estimation(image: Image.Image) -> np.ndarray:
    width, height = image.size
    scale = min(MAX_INPUT_SIZE / max(width, height), 1.0)
    if scale < 1.0:
        resized = image.resize(
            (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
        )
    else:
        resized = image
    input_batch = TRANSFORM(resized).to(DEVICE)
    with torch.no_grad():
        prediction = MODEL(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=resized.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    if resized.size != image.size:
        depth = np.array(
            Image.fromarray(depth).resize(image.size, Image.Resampling.BICUBIC)
        )
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def depth_to_point_cloud(depth: np.ndarray) -> o3d.geometry.PointCloud:
    depth_image = o3d.geometry.Image((depth * 1000).astype(np.uint16))
    height, width = depth.shape
    fx = fy = max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        intrinsics,
        depth_scale=1000.0,
        depth_trunc=3.0,
        stride=POINT_STRIDE,
    )
    pcd.remove_non_finite_points()
    if VOXEL_DOWNSAMPLE_SIZE > 0:
        pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE_SIZE)
    pcd.estimate_normals()
    return pcd


def colorize_point_cloud(pcd: o3d.geometry.PointCloud, image: Image.Image) -> None:
    colors = np.asarray(image).astype(np.float32) / 255.0
    height, width, _ = colors.shape
    points = np.asarray(pcd.points)
    if points.size == 0:
        return
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range == 0 or y_range == 0:
        return
    u = (points[:, 0] - x_min) / x_range
    v = (points[:, 1] - y_min) / y_range
    u = np.clip((u * (width - 1)).round().astype(int), 0, width - 1)
    v = np.clip(((1 - v) * (height - 1)).round().astype(int), 0, height - 1)
    pcd.colors = o3d.utility.Vector3dVector(colors[v, u])


def point_cloud_to_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    pcd.orient_normals_consistent_tangent_plane(10)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=POISSON_DEPTH,
    )
    mesh.compute_vertex_normals()
    return mesh


def transfer_colors_to_mesh(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud
) -> None:
    if len(pcd.points) == 0:
        return
    if len(pcd.colors) == 0:
        return
    mesh.vertex_colors = pcd.colors


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    depth = run_depth_estimation(image)
    depth_image = Image.fromarray((depth * 255).astype(np.uint8))
    depth_path = job_dir / "depth.png"
    depth_image.save(depth_path)

    pcd = depth_to_point_cloud(depth)
    colorize_point_cloud(pcd, image)
    pcd_path = job_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    mesh = point_cloud_to_mesh(pcd)
    transfer_colors_to_mesh(mesh, pcd)
    mesh_path = job_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    mesh_obj_path = job_dir / "mesh.obj"
    o3d.io.write_triangle_mesh(str(mesh_obj_path), mesh, write_triangle_uvs=True)

    return {
        "job_id": job_id,
        "depth_url": f"/outputs/{job_id}/depth.png",
        "point_cloud_url": f"/outputs/{job_id}/point_cloud.ply",
        "mesh_url": f"/outputs/{job_id}/mesh.ply",
        "mesh_obj_url": f"/outputs/{job_id}/mesh.obj",
    }


@app.get("/outputs/{job_id}/{filename}")
async def download_output(job_id: str, filename: str):
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
