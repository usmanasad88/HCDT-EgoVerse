from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any
from rich import print


def _run(cmd: list[str], cwd: Path | None = None):
    print(f"[cyan]$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _check_colmap_available():
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP not found on PATH. Please install and ensure 'colmap' is available.")


def run_colmap(image_dir: str | os.PathLike, output_dir: str | os.PathLike) -> Path:
    """
    Run COLMAP SfM pipeline and produce transforms.json for neural rendering.

    Steps:
      - feature_extractor
      - exhaustive_matcher
      - mapper (sparse reconstruction)
      - model_converter to text
      - parse to transforms.json (one camera per image with 4x4 pose)
    """
    _check_colmap_available()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 1) Feature extraction
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ])

    # 2) Matching
    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ])

    # 3) Mapping
    _run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ])

    # Choose first model
    model_dir = next((p for p in sparse_dir.iterdir() if p.is_dir()), None)
    if model_dir is None:
        raise RuntimeError("COLMAP mapping produced no model")

    # 4) Convert to text for easier parsing
    text_dir = output_dir / "sparse_text"
    text_dir.mkdir(exist_ok=True)
    _run([
        "colmap", "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(text_dir),
        "--output_type", "TXT",
    ])

    # 5) Parse to transforms.json
    transforms = colmap_text_to_transforms(text_dir, image_dir)
    out_json = output_dir / "transforms.json"
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"[green]Wrote poses to {out_json}")
    # Also place a copy next to images/ for 3DGS source path consumption
    source_root = image_dir.parent
    try:
        with open(source_root / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)
        print(f"[green]Copied poses to {source_root / 'transforms.json'}")
    except Exception as e:
        print(f"[yellow]Could not copy transforms.json to {source_root}: {e}")
    return out_json


def colmap_text_to_transforms(text_model_dir: str | os.PathLike, image_dir: str | os.PathLike) -> Dict[str, Any]:
    """
    Parse COLMAP TXT model (cameras.txt, images.txt, points3D.txt) into a transforms.json-like dict.

    Output format (NeRF-like):
    {
      "camera_angle_x": float,
      "frames": [
        {"file_path": "images/frame_00000.png", "transform_matrix": [[...],[...],[...],[...]]},
        ...
      ]
    }
    """
    text_model_dir = Path(text_model_dir)
    image_dir = Path(image_dir)
    cameras_txt = text_model_dir / "cameras.txt"
    images_txt = text_model_dir / "images.txt"

    # Read camera intrinsics from first camera
    fx = fy = None
    width = height = None
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            # FORMAT: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            if model in ("PINHOLE", "SIMPLE_PINHOLE"):
                if model == "PINHOLE":
                    fx, fy, cx, cy = params[:4]
                else:
                    fx = fy = params[0]
                    cx, cy = params[1:3]
            elif model in ("OPENCV", "SIMPLE_RADIAL", "FULL_OPENCV"):
                fx, fy, cx, cy = params[:4]
            else:
                # Fallback reasonable default
                fx = fy = params[0]
                cx, cy = params[1:3]
            break

    import math
    assert fx is not None and fy is not None
    if width and fx:
        camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))
    else:
        camera_angle_x = 0.7

    frames = []
    with open(images_txt, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue
        parts = line.strip().split()
        # FORMAT:
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = " ".join(parts[9:])  # support names with spaces

        c2w = qvec_t_to_pose(qw, qx, qy, qz, tx, ty, tz)
        frames.append({
            "file_path": str(Path("images") / name),
            "transform_matrix": c2w,
        })
        # Skip next line with 2D-3D correspondences
        i += 2
    

    return {"camera_angle_x": camera_angle_x, "frames": frames}


def qvec_t_to_pose(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float):
    """Convert COLMAP world-to-camera (q,t) into camera-to-world 4x4 list of lists.

    COLMAP stores image orientation as a quaternion qvec that rotates world to camera,
    and translation t such that x_cam = R * x_world + t.
    We need camera-to-world: R_c2w = R.T, t_c2w = -R.T @ t.
    """
    import numpy as np

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    R = qvec2rotmat(q)
    t = np.array([tx, ty, tz], dtype=np.float64)
    R_c2w = R.T
    t_c2w = -R.T @ t
    M = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3, 3] = t_c2w
    return M.tolist()


def qvec2rotmat(qvec):
    import numpy as np
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
