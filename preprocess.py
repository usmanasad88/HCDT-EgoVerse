from __future__ import annotations

import cv2
import os
from pathlib import Path
from typing import Optional
from rich import print


def extract_frames(video_path: str | os.PathLike, output_dir: str | os.PathLike, fps: Optional[float] = None) -> Path:
    """
    Extract frames from video into output_dir/images as PNGs.

    - If fps is None, uses all frames.
    - Creates a directory layout compatible with COLMAP (images in a single folder).

    Returns the images directory path.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps is None or fps <= 0:
        frame_skip = 1
    else:
        frame_skip = max(1, int(round(src_fps / fps)))

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:
            out_path = images_dir / f"frame_{saved:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[green]Extracted {saved} frames to {images_dir}")
    return images_dir
