from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
from rich import print


def detect_objects(image_dir: str | os.PathLike, model_name: str = "yolov8n.pt") -> Dict[str, Any]:
    """
    Run YOLOv8 inference on images and return detections per frame.

    Returns dict: {frame_name: [{"bbox": [x1,y1,x2,y2], "cls": int, "conf": float, "label": str}, ...]}
    """
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    model = YOLO(model_name)
    detections: Dict[str, Any] = {}
    images = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in image_dir.glob(ext)])
    for img_path in images:
        results = model.predict(str(img_path), verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy.squeeze(0).tolist()
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                label = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)
                dets.append({"bbox": xyxy, "cls": cls_id, "conf": conf, "label": label})
        detections[img_path.name] = dets
    print(f"[green]YOLO detections computed for {len(images)} frames")
    return detections


def estimate_human_pose(image_dir: str | os.PathLike, gender: str = "neutral") -> Dict[str, Any]:
    """
    Placeholder SMPL-X estimator.

    In production, plug in an ExPose/VIBE/PIXIE-like regressor to obtain pose, shape, and translation.
    Here we create a simple, slowly moving global translation with neutral pose.

    Returns dict: {frame_name: {"betas": [...], "global_orient": [3], "body_pose": [63], "transl": [3]}}
    """
    # Load SMPL-X body model to verify availability, but we won't run optimization here.
    try:
        import smplx
        _ = smplx.create(model_path=os.environ.get("SMPLX_MODEL_DIR", ""), model_type="smplx", gender=gender)
    except Exception as e:
        print(f"[yellow]SMPL-X model load failed or path not set (SMPLX_MODEL_DIR). Proceeding with dummy params. {e}")

    image_dir = Path(image_dir)
    frames = sorted([p.name for p in image_dir.glob("*.png")])
    out: Dict[str, Any] = {}
    for i, name in enumerate(frames):
        # Simple walk forward in Z
        transl = [0.0, 0.0, float(i) * 0.01]
        out[name] = {
            "betas": [0.0] * 10,
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0] * 63,
            "left_hand_pose": [0.0] * 45,
            "right_hand_pose": [0.0] * 45,
            "jaw_pose": [0.0] * 3,
            "leye_pose": [0.0] * 3,
            "reye_pose": [0.0] * 3,
            "expression": [0.0] * 50,
            "transl": transl,
        }
    print(f"[green]SMPL-X parameters prepared for {len(frames)} frames (placeholder)")
    return out
