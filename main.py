#!/usr/bin/env python3
"""
HCDT-EgoVerse: Semantic, human-aware digital twin from EgoExo4D videos

Setup prerequisites (one-time):
1) Python deps:
   pip install -r requirements.txt

2) External tools:
   - COLMAP: Install system-wide and ensure `colmap` is on PATH.
     Ubuntu: sudo apt install colmap (or build from source for CUDA).
   - 3D Gaussian Splatting (3DGS) official repo cloned locally.
     Export env var DGS_REPO pointing to repo root (contains train.py).
     Example: export DGS_REPO=~/Repos/gaussian-splatting

3) SMPL-X model files:
   - Download SMPL-X model (male/female/neutral) from https://smpl-x.is.tue.mpg.de/
   - Set env var SMPLX_MODEL_DIR to the folder containing model .pkl files.

Run the pipeline:
   python main.py \
     --ego_video /path/to/ego.mp4 \
     --exo_video /path/to/exo.mp4 \
     --workdir /path/to/output_dir

Notes:
 - This script orchestrates modules: preprocess -> colmap -> 3dgs -> analysis -> visualization.
 - COLMAP and 3DGS training can be compute intensive; consider sub-sampling frames via --fps.
 - For SMPL-X estimation you may plug in your preferred estimator; a lightweight placeholder is provided.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

from rich import print
from rich.panel import Panel

import preprocess
import pose_estimation
import scene_reconstruction
import analysis
import visualizer


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Create a semantic, human-aware digital twin from EgoExo4D videos")
    parser.add_argument(
        "--ego_video",
        type=str,
        required=False,
        default="/home/mani/Central/Cooking1/FairCooking/fair_cooking_05_2/aria02_214-1.mp4",
        help="Path to egocentric video file (ego.mp4)",
    )
    parser.add_argument(
        "--exo_video",
        type=str,
        required=False,
        default="/home/mani/Central/Cooking1/FairCooking/fair_cooking_05_2/cam03.mp4",
        help="Path to exocentric video file (exo.mp4)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=False,
        default=None,
        help="Working directory for outputs (optional; auto if omitted)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Optional FPS for frame extraction (downsample)")
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP if poses already exist")
    parser.add_argument("--skip_3dgs", action="store_true", help="Skip 3DGS training if model exists")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--smplx_gender", type=str, default="neutral", choices=["neutral", "male", "female"], help="SMPL-X gender to load")
    args = parser.parse_args()

    # Resolve working directory: if not provided, create under ./outputs/<ego_stem> (unique if exists)
    if args.workdir:
        workdir = Path(args.workdir).absolute()
    else:
        ego_stem = Path(args.ego_video).stem if args.ego_video else "run"
        outputs_root = Path.cwd() / "outputs"
        ensure_dir(outputs_root)
        candidate = outputs_root / ego_stem
        if candidate.exists():
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate = outputs_root / f"{ego_stem}_{ts}"
        workdir = candidate
    ensure_dir(workdir)

    print(Panel.fit(f"[bold cyan]HCDT-EgoVerse Pipeline[/]\nWorkdir: [green]{workdir}[/]"))

    # 1) Preprocess: extract frames from ego (and optionally exo)
    images_dir = workdir / "images_ego"
    ensure_dir(images_dir)
    preprocess.extract_frames(args.ego_video, images_dir, fps=args.fps)

    # 2) Camera pose estimation via COLMAP
    colmap_dir = workdir / "colmap_ego"
    ensure_dir(colmap_dir)
    if not args.skip_colmap:
        transforms_json = pose_estimation.run_colmap(images_dir, colmap_dir)
    else:
        transforms_json = colmap_dir / "transforms.json"
        if not transforms_json.exists():
            raise FileNotFoundError("--skip_colmap set but transforms.json not found")

    # 3) Train 3DGS
    dgs_model_path = workdir / "gaussians" / "final_model.ply"
    if not args.skip_3dgs:
        scene_reconstruction.train_3dgs(colmap_project_dir=colmap_dir, image_dir=images_dir, output_model_path=dgs_model_path)
    else:
        if not dgs_model_path.exists():
            raise FileNotFoundError("--skip_3dgs set but final_model.ply not found")

    # 4) Analysis: objects + SMPL-X
    detections = analysis.detect_objects(images_dir, model_name=args.yolo_model)
    smplx_poses = analysis.estimate_human_pose(images_dir, gender=args.smplx_gender)

    # Persist intermediate JSONs for reproducibility
    with open(workdir / "detections.json", "w") as f:
        json.dump(detections, f)
    with open(workdir / "smplx_poses.json", "w") as f:
        json.dump(smplx_poses, f)

    # Load camera poses from transforms.json
    with open(transforms_json, "r") as f:
        camera_poses: Dict[str, Any] = json.load(f)

    # 5) Visualize fused digital twin
    visualizer.visualize_digital_twin(
        dgs_model_path=dgs_model_path,
        object_detections=detections,
        smplx_poses=smplx_poses,
        camera_poses=camera_poses,
    )


if __name__ == "__main__":
    main()
