# Full pipeline (copy-paste): egocentric video → HaWoR → HumanEgo policy

Verified end-to-end on `scilab_sample` (no Project Aria). Each block below is meant to
be **pasted as-is** into a shell. Paths are absolute and match this machine.

```
video ─▶ [HaWoR annotate] ─▶ SLAM/*.npz + cam_space/*  (camera + hands)
      ─▶ [adapter ingest]  ─▶ HumanEgo session (per-frame rgb/cam/hands/phases/slam)
      ─▶ [HumanEgo downstream: DINO+SAM2 → CoTracker → triangulation → LaMa → datasetgen]
      ─▶ training_data.json ─▶ [train] ─▶ runs/scilab/HumanEgo/
```

Pipeline stages and what they need:

| Stage | Model | Needs |
|-------|-------|-------|
| adapter ingest | — | numpy + cv2 |
| dinosam | Grounding DINO + SAM2 | GPU + local weights |
| kptsselector | — | — |
| cotracker | CoTracker3 | GPU + local weights |
| camtriangulator | — (PCA pose) | camera parallax |
| lama | LaMa-ONNX | GPU + local weights |
| visualkpts / datasetgen | — | — |
| training | — | GPU |

---

## 0. One-time setup

### 0a. Shared shell variables (paste once per shell)

```bash
conda activate humanego
export HE=/mnt/data/liuyu/project/HumanEgo
export ADP=/mnt/data/liuyu/project/DataPipeline/pipelines/hawor/integrations/humanego
export SS=/mnt/data/liuyu/project/scilab/scilab_sample      # HaWoR-annotated clips
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1              # use local weights only
export CUDA_VISIBLE_DEVICES=0
```

### 0b. Download the 4 model weights to local folders (needs network; run once)

```bash
hf download IDEA-Research/grounding-dino-tiny  --local-dir $HE/models/grounding-dino-tiny
hf download facebook/sam2-hiera-tiny  sam2_hiera_tiny.pt  --local-dir $HE/models/sam2-hiera-tiny
hf download facebook/cotracker3       scaled_offline.pth  --local-dir $HE/models/cotracker3
hf download Carve/LaMa-ONNX           lama_fp32.onnx      --local-dir $HE/models/lama-onnx
```

`cfg/preprocess/tasks/scilab.yaml` already points at exactly these paths; the loaders
(`DINOSAM.py`, `CoTrackerOffline.py`, `Lama.py`) use a local path when present and only
fall back to HuggingFace otherwise. Orient-Anything is **not** needed (scilab uses
`pose_method: pca1`).

---

## 1. (Optional) HaWoR annotation — skip if you already have HaWoR outputs

`scilab_sample` is already HaWoR-annotated, so **skip this** and go to Step 2. To annotate
your **own** video, arrange a session folder `<name>/<name>.MP4` + `<name>/<name>/` and run:

```bash
conda activate hawor   # separate env: py3.10 / torch1.13 / cu117 + DROID-SLAM + weights
cd /mnt/data/liuyu/project/DataPipeline/pipelines/hawor
python scripts/hawor_video_processor.py \
    --video_path /path/to/<name>/<name>.MP4 \
    --output_dir /path/to/<name>/<name> \
    --gpu_ids 0 --vis_mode off --overwrite_output
conda activate humanego   # switch back for Steps 2+
```

---

## 2. Ingest + preprocess ≥2 recordings into the HumanEgo dataset

Training holds out recording `000` for eval, so process at least two. This loop maps the
first two `scilab_sample` clips to `mps_scilab_000_vrs` / `mps_scilab_001_vrs` and runs the
whole HumanEgo downstream (~6 min/recording on one GPU). `--max_frames 150` keeps it quick;
drop it to process the full clip.

```bash
cd $ADP
SESSIONS=(DJI_20260409180326_0364_D_A012 DJI_20260409182349_0366_D_A012)
i=0
for s in "${SESSIONS[@]}"; do
  rec=$(printf "%03d" $i)
  echo "=== recording $rec  <-  $s ==="
  python run_humanego_from_hawor.py \
    --hawor_session "$SS/$s" \
    --out_dir       "$HE/data/scilab/aria/mps_scilab_${rec}_vrs" \
    --humanego_root "$HE" \
    --task scilab --through datasetgen --max_frames 150 --use_mano
  i=$((i+1))
done
```

> `--use_mano` exports the true thumb/index-tip **midpoint** (and a thumb-index-distance
> grasp) instead of the MANO wrist — needs `torch` + `_DATA/data/mano/MANO_RIGHT.pkl`
> (symlink it from `$HE/models/pretrained_models/MANO_RIGHT.pkl`). It improves the hand
> control point and is required for the grasp-latch to have a chance of firing (see
> EVALUATION.md / INTEGRATION_REPORT.txt). Drop it to use the dependency-free wrist proxy.

Check it produced per-frame targets:

```bash
for r in 000 001; do
  echo -n "rec $r: "; ls $HE/data/scilab/aria/mps_scilab_${r}_vrs/preprocess/all_data/*/training_data.json | wc -l
done
```

> **Run faster / more data:** the GPU stages are independent per recording — process clips
> on different GPUs in parallel by setting `CUDA_VISIBLE_DEVICES=k` in separate shells.
> Validate the seam without GPU/weights first with `--through indices`.

---

## 3. Train

`cfg/training/scilab/HumanEgo.yaml` is preconfigured (dual-hand, `data_sources: {aria: N}`,
image `rgb_WoArm_WArmObjKpts.png`). Recording `000` is held out for eval.

```bash
cd $HE
python -m training.FlowMatchingTrainer --task scilab --use_cfg --job HumanEgo
```

Quick smoke test (1 epoch) to confirm the dataloader before a long run:

```bash
cd $HE
python -m training.FlowMatchingTrainer --task scilab --use_cfg --job HumanEgo \
    --epochs 1 --batch_size 4 --num_workers 2
```

Outputs (checkpoints, eval renders) go to `runs/scilab/HumanEgo/`.

---

## 4. (Optional) Inference on a real robot

```bash
cd $HE
SKIP_HARDWARE=0 bash setup.sh                      # RealSense + Trossen drivers
python inference/run_inference.py cfg/inference/example_dualarm.yaml
```

Implement `Camera` / `RobotArm` / `Perception` for your rig — see `inference/README.md`.

---

## What this integration changed in HumanEgo

| File | Change |
|------|--------|
| `preprocess/DatasetGen.py` | registered `hawor` in the two hand-method maps |
| `preprocess/DINOSAM.py` | accept local `sam2_checkpoint_path` (fallback to HF) |
| `preprocess/CoTrackerOffline.py` | accept local `cotracker_checkpoint_path` (fallback to HF) |
| `preprocess/Lama.py` | accept local `lama_model_path` (fallback to HF) |
| `cfg/preprocess/tasks/scilab.yaml` | object prompts + local model paths + `pca1` pose |
| `cfg/training/scilab/HumanEgo.yaml` | dual-hand training config |

The adapter (`hawor_to_humanego.py`) writes, per HaWoR frame: `rgb.png`,
`aria_cam_rgb.json` (metric `c2w`+`K`), `aria_hands.json` + `hawor_hands.json`,
`aria_phases.json` (mode 4 ⇒ `is_finished` tail), `aria_slam.json`, plus top-level
`aria_cam_rgb_config.json` / `aria_phases_results.json`. The runner then skips the Aria
extraction stage and drives `indices → … → datasetgen`.

## Known limitations / tunables (carried from the HaWoR front-end)

- **Hand frame = MANO wrist**, not the true thumb/index midpoint (no MANO layer used).
  Because the wrist sits >0.20 m from object centroids, DatasetGen's grasp **latch never
  fires**, so objects stay at their static triangulated pose (`is_dynamic=False`). To get
  dynamic object poses: run MANO in the `hawor` env for a true midpoint, and/or raise the
  latch distance in `DatasetGen` (the `min_dist < 0.20` check).
- **Grasp** is a sequence-relative finger-flexion proxy (`--grasp_rel`).
- **Triangulation** needs camera parallax; egocentric clips with a near-static head can
  give noisy object depth. Inspect `preprocess/object_centric.png` /
  `preprocess/dinosam_mask_obj*.png` and tune the object prompts in `scilab.yaml`.
- **`--max_frames 150`** is for speed; drop it for full clips. Full clip ≈ 1193 frames
  ≈ ~20 min/recording through dinosam.

## Verified run (this machine)

ingest → indices → dinosam (2.9 min) → kptsselector → cotracker (27 s) → camtriangulator
(2 s, obj1/obj2 6D poses) → lama (2 min) → visualkpts (1 min) → datasetgen → **150
training_data.json/recording**; 2 recordings processed; `FlowMatchingTrainer` built 300
samples, trained 1 epoch, rendered eval video — all exit 0.
