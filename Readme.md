HCDT-EgoVerse: Semantic, Human-Aware Digital Twin Pipeline

Overview
This repository provides a modular Python pipeline to create a semantic, human-aware digital twin from synchronized egocentric and exocentric videos (EgoExo4D). It reconstructs a 3D environment using 3D Gaussian Splatting, detects objects, estimates human pose (SMPL-X), and fuses everything into an interactive 3D scene.

Prerequisites
- OS: Linux recommended
- Python: 3.9+
- GPU: CUDA GPU recommended for COLMAP/3DGS/Torch acceleration
- External tools:
	- COLMAP installed and on PATH
	- 3D Gaussian Splatting official repo cloned; set env var DGS_REPO to its path
	- SMPL-X model files downloaded; set env var SMPLX_MODEL_DIR to the directory with .pkl files

Install via Conda (recommended)
```bash
conda create -y -n hcdt-egoverse python=3.10
conda activate hcdt-egoverse
pip install -r requirements.txt
```

Install Python dependencies (pip, alternative)
```bash
pip install -r requirements.txt
```

Run the pipeline
```bash
python main.py \
	--ego_video /path/to/ego.mp4 \
	--exo_video /path/to/exo.mp4 \
	--workdir /path/to/workdir \
	--fps 5
```

Outputs
- workdir/images_ego/images: extracted frames
- workdir/colmap_ego: COLMAP database, sparse model, transforms.json
- workdir/gaussians/final_model.ply: trained 3DGS point cloud
- workdir/detections.json: YOLOv8 detections
- workdir/smplx_poses.json: SMPL-X params (placeholder)

Notes
- 3DGS training is a wrapper; ensure train.py and repo options match your clone.
- SMPL-X estimation is a placeholder; plug in your preferred regressor for real parameters.
