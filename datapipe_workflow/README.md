# DataPipeline Workflow Code Export

This directory contains the code needed to run the integrated DataPipeline workflow for the current `caption` and `hawor` pipelines. Model weights, MANO assets, test videos, outputs, scratch folders, build artifacts, and Python caches are intentionally excluded.

## Structure

```text
annotate_video.py       one video in, action labels and hand/camera export out
env_profiles/           runtime Python and GPU profile
pipelines/caption/      action and scene annotation
pipelines/hawor/        HaWoR hand and camera reconstruction
scripts/                install and asset download scripts
assets/assets.yaml      model/asset manifest
examples/               videos the viewer discovers
```

## Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

For the full HaWoR GPU runtime, also run:

```bash
bash scripts/install_hawor_env.sh
```

That script handles CUDA-specific packages, DROID-SLAM extension build, PyTorch3D CUDA rebuild, `mmcv`, and `chumpy`.

## Download Model Assets

```bash
bash scripts/download_assets.sh
```

The script downloads public HuggingFace/Google Drive assets listed in `assets/assets.yaml` and checks MANO files. MANO files must be downloaded manually from the official MANO website after accepting the license:

```text
pipelines/hawor/_DATA/data/mano/MANO_RIGHT.pkl
pipelines/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

## Run

From this directory, pass one video. API settings come from `pipelines/caption/api_config.json`. HaWoR uses the Python and GPU in `env_profiles/default.yaml`.

```bash
python annotate_video.py /path/to/video.mp4 --output examples/my_clip
```

The result is `input/<video>.mp4`, `output/ego_action_annotation.json`, `output/ego_process/ego_hands_reconstruction/hands.npz`, plus `output/keypoints.npz` and `output/mesh.bin` for the viewer.
