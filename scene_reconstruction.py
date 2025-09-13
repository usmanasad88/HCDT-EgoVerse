from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from rich import print


def _run(cmd: list[str], cwd: Path | None = None):
    print(f"[cyan]$ {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, check=True)


def train_3dgs(colmap_project_dir: str | os.PathLike, image_dir: str | os.PathLike, output_model_path: str | os.PathLike):
    """
    Launch 3D Gaussian Splatting training using the official repo.

    Requirements:
      - Environment var DGS_REPO must point to repo root containing train.py
      - COLMAP outputs (sparse model + transforms.json) available in colmap_project_dir
    """
    dgs_repo = os.environ.get("DGS_REPO", None)
    if not dgs_repo:
        raise RuntimeError("DGS_REPO env var not set. Point it to the gaussian-splatting repo root.")

    dgs_repo = Path(dgs_repo)
    train_py = dgs_repo / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Could not find train.py at {train_py}")

    image_dir = Path(image_dir)
    colmap_project_dir = Path(colmap_project_dir)
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Typical 3DGS invocation; adjust args as needed for your repo/version.
    # Many repos accept --source_path with images + transforms.json.
    source_path = image_dir.parent  # where images/ and transforms.json reside
    ckpt_dir = output_model_path.parent

    cmd = [
        "python", str(train_py),
        "-s", str(source_path),  # source path expects images/ + transforms.json
        "--eval", "False",
        "--iterations", "30000",
        "--save_iterations", "30000",
        "--checkpoint_folder", str(ckpt_dir),
    ]

    _run(cmd, cwd=dgs_repo)

    # Heuristic: many repos write point_cloud/point_cloud.ply or output/*.ply
    # Try common locations and copy/link to final_model.ply
    candidates = [
        ckpt_dir / "point_cloud" / "point_cloud.ply",
        ckpt_dir / "point_cloud.ply",
        ckpt_dir / "output" / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists():
            shutil.copy2(c, output_model_path)
            print(f"[green]Saved Gaussian model to {output_model_path}")
            return

    raise FileNotFoundError("Could not locate trained .ply output from 3DGS. Check repo's output paths.")
