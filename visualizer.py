from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Dict, Any, List

import numpy as np
import open3d as o3d
from rich import print


def _load_point_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        print(f"[yellow]Warning: point cloud {ply_path} is empty")
    return pcd


def _bbox3d_from_2d(xyxy: List[float], depth: float, K: np.ndarray, c2w: np.ndarray) -> o3d.geometry.AxisAlignedBoundingBox:
    """Project a 2D bbox with an approximate depth into 3D axis-aligned bbox in world coordinates.
    This is a crude approach: we back-project 4 corners at a single depth to get a thin box.
    """
    x1, y1, x2, y2 = xyxy
    corners_px = np.array([
        [x1, y1, 1.0],
        [x2, y1, 1.0],
        [x2, y2, 1.0],
        [x1, y2, 1.0],
    ]).T  # 3x4
    Kinv = np.linalg.inv(K)
    rays = Kinv @ corners_px  # 3x4 camera rays
    pts_cam = rays * depth  # assume z=depth along ray
    # Homogenize and transform to world
    pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
    pts_world_h = c2w @ pts_cam_h
    pts_world = pts_world_h[:3, :].T
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts_world))
    return aabb


def _smplx_mesh_from_params(params: Dict[str, Any], gender: str = "neutral") -> o3d.geometry.TriangleMesh:
    """Create an SMPL-X mesh from provided params using smplx; if not available, return a sphere placeholder."""
    try:
        import smplx
        import torch
        model_path = Path(os.environ.get("SMPLX_MODEL_DIR", ""))
        body_model = smplx.create(model_path=str(model_path), model_type="smplx", gender=gender, use_pca=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        body_model = body_model.to(device)

        with torch.no_grad():
            out = body_model(
                betas=torch.tensor(params.get("betas", [0.0]*10), dtype=torch.float32, device=device).unsqueeze(0),
                global_orient=torch.tensor(params.get("global_orient", [0.0,0.0,0.0]), dtype=torch.float32, device=device).unsqueeze(0),
                body_pose=torch.tensor(params.get("body_pose", [0.0]*63), dtype=torch.float32, device=device).unsqueeze(0),
                left_hand_pose=torch.tensor(params.get("left_hand_pose", [0.0]*45), dtype=torch.float32, device=device).unsqueeze(0),
                right_hand_pose=torch.tensor(params.get("right_hand_pose", [0.0]*45), dtype=torch.float32, device=device).unsqueeze(0),
                jaw_pose=torch.tensor(params.get("jaw_pose", [0.0]*3), dtype=torch.float32, device=device).unsqueeze(0),
                leye_pose=torch.tensor(params.get("leye_pose", [0.0]*3), dtype=torch.float32, device=device).unsqueeze(0),
                reye_pose=torch.tensor(params.get("reye_pose", [0.0]*3), dtype=torch.float32, device=device).unsqueeze(0),
                expression=torch.tensor(params.get("expression", [0.0]*50), dtype=torch.float32, device=device).unsqueeze(0),
                transl=torch.tensor(params.get("transl", [0.0,0.0,0.0]), dtype=torch.float32, device=device).unsqueeze(0),
            )
        vertices = out.vertices[0].detach().cpu().numpy()
        faces = body_model.faces
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.6, 0.4])
        return mesh
    except Exception as e:
        # Fallback placeholder
        print(f"[yellow]SMPL-X mesh generation failed, using sphere placeholder: {e}")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.6, 0.4])
        return mesh


def visualize_digital_twin(dgs_model_path: str | Path,
                           object_detections: Dict[str, Any],
                           smplx_poses: Dict[str, Any],
                           camera_poses: Dict[str, Any]):
    """
    Visualize the fused scene using Open3D.
    - dgs_model_path: path to Gaussian point cloud (.ply)
    - object_detections: dict from detect_objects
    - smplx_poses: dict from estimate_human_pose
    - camera_poses: transforms.json dict from COLMAP
    """
    dgs_model_path = Path(dgs_model_path)
    pcd = _load_point_cloud(dgs_model_path)

    # Build a very rough camera intrinsics from camera_angle_x and image size if available
    cam_angle_x = float(camera_poses.get("camera_angle_x", 0.7))
    frames = camera_poses.get("frames", [])
    if not frames:
        print("[red]No camera frames found in poses; cannot visualize")
        return

    # Try to get image size from any detection key by reading one image lazily
    import cv2
    any_img_name = next((k for k in object_detections.keys()), None) or Path(frames[0]["file_path"]).name
    # Assumes images are under images/ relative path where detections were created
    img_path_guess = None
    # Attempt to locate image file by searching known folders
    for hint in ["images_ego/images", "images", "."]:
        p = Path(hint) / Path(any_img_name)
        if p.exists():
            img_path_guess = p
            break
    width = height = 640
    if img_path_guess is not None and Path(img_path_guess).exists():
        im = cv2.imread(str(img_path_guess))
        if im is not None:
            height, width = im.shape[:2]
    fx = width / (2.0 * np.tan(cam_angle_x / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Select a reference frame for visualizing boxes and avatar
    ref = frames[len(frames)//2]
    ref_name = Path(ref["file_path"]).name
    c2w = np.array(ref["transform_matrix"], dtype=np.float64)

    # Build geometries list
    geoms: List[o3d.geometry.Geometry] = [pcd]

    # Draw 3D boxes for detections at a heuristic depth (e.g., 2.0m)
    depth_guess = 2.0
    for det in object_detections.get(ref_name, []):
        aabb = _bbox3d_from_2d(det["bbox"], depth_guess, K, c2w)
        line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        colors = [[1, 0, 0] for _ in range(len(line_set.lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(line_set)

    # Add SMPL-X mesh for the same frame
    mesh = _smplx_mesh_from_params(smplx_poses.get(ref_name, {}))
    geoms.append(mesh)

    o3d.visualization.draw_geometries(geoms)
