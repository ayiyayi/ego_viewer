# HaWoR → HumanEgo integration

Use **HaWoR** as the geometric front-end for **HumanEgo** on **non-Aria** egocentric
video. Where HumanEgo normally gets its camera trajectory and per-frame hand poses
from Project Aria MPS, this adapter supplies both from a HaWoR-annotated clip, so the
rest of HumanEgo's Aria preprocessing pipeline runs unchanged — no Aria glasses, no MPS.

```
HaWoR session ──(hawor_to_humanego.py)──▶ HumanEgo session
   SLAM/*.npz   →  per-frame aria_cam_rgb.json (c2w + K)        ┐
   cam_space/*  →  per-frame aria_hands.json (+ hawor_hands.json)│ = what the Aria
   (captions)   →  aria_phases_results.json (whole clip = manip) ┘   stage would write
                                                          │
        skip preprocess_aria (needs VRS/MPS)              ▼
   indices ▶ dinosam ▶ kptsselector ▶ cotracker ▶ camtriangulator
           ▶ lama ▶ visualkpts ▶ datasetgen ▶ training_data.json
```

## Why not just "enable an alternative hand method"?

HumanEgo's WiLoR / HaMeR / MediaPipe hand methods are **not** standalone: they still
take camera poses from Aria (`WiLoRHands` is constructed with an `AriaCam` and lifts
detections to world via Aria's per-frame `c2w`). With no Aria there is no camera
trajectory, no world frame, and no object triangulation baseline. HaWoR replaces that
**whole geometric backbone** (its masked DROID-SLAM gives a metric-scaled camera
trajectory) *and* the hands (per-frame MANO), which is exactly the pair the Aria stage
produces — so it substitutes the front-end wholesale instead of just the hand detector.

## Files

| File | Role |
|------|------|
| `hawor_to_humanego.py` | Ingest adapter: one HaWoR session → one HumanEgo session directory. Pure `numpy` for geometry; `cv2` only for frame extraction. |
| `run_humanego_from_hawor.py` | Runner: ingest, then drive HumanEgo's Aria-path stages (skipping `preprocess_aria`) up to a chosen stage. |

HumanEgo-side changes made by this integration:
* `preprocess/DatasetGen.py` — registered `"hawor"` in `HAND_METHOD_JSON_MAP`
  (`hawor_hands.json`) and `HAND_METHOD_ENTITY_KEY` (`hands_hawor`).
* `cfg/preprocess/tasks/scilab.yaml` — task config for the wet-lab clips (object
  prompts + `hand_tracking_methods: ["aria_mps", "hawor"]`).

## Quick start

Run in HumanEgo's conda env (`humanego`: has `numpy`, `cv2`, `torch`+CUDA):

```bash
SESS=/mnt/data/liuyu/project/scilab/scilab_sample/DJI_20260409180326_0364_D_A012
OUT=/mnt/data/liuyu/project/HumanEgo/data/scilab/aria/mps_scilab_000_vrs

# A) adapter only — produces the HumanEgo session layout
python hawor_to_humanego.py --session_dir "$SESS" --out_dir "$OUT"

# B) ingest + run the HumanEgo downstream (object stages need GPU + weights, see below)
python run_humanego_from_hawor.py \
    --hawor_session "$SESS" --out_dir "$OUT" \
    --humanego_root /mnt/data/liuyu/project/HumanEgo \
    --task scilab --through datasetgen

# validate the seam without GPU/weights:
python run_humanego_from_hawor.py ... --task scilab --through indices --max_frames 220
```

Then train exactly as HumanEgo documents (the session lives under `data/scilab/aria/`):

```bash
python -m training.FlowMatchingTrainer --task scilab --use_cfg --job HumanEgo
```

## What the adapter produces (HumanEgo session layout)

```
<out>/preprocess/
    aria_cam_rgb_config.json
    aria_phases_results.json            # whole clip = one manipulation window
    all_data/<idx:05d>/
        rgb.png                         # frame, sampled at HaWoR's 30 fps grid
        aria_cam_rgb.json               # c2w (metric) + K + w/h/fps
        aria_hands.json                 # default "hands" source for DatasetGen
        hawor_hands.json                # provenance copy → entity "hands_hawor"
```

## Key conventions & approximations (read before trusting the numbers)

* **Cameras** are decoded exactly like HaWoR's `load_slam_cam`: quaternion order
  `[x,y,z,w]`, translation × `scale` (metric). The `R_x = diag(1,-1,-1)` flip in
  HaWoR's `demo.py` is **not** applied — it is only for aitviewer rendering. Camera and
  hands are lifted by the *same* raw `c2w`, so they share one consistent world frame.
  Verified: frame 0 ≈ identity; wrists reproject onto the real hands in the RGB.
* **Frame sampling**: HaWoR extracts with ffmpeg `-vf fps=30`, so HaWoR index `i` ↔
  native frame `round(i * native_fps / 30)`. The adapter reproduces this so each
  `rgb.png` aligns with its pose. (Sample clip: 2688×1512 @ 50 fps native → 1193 frames.)
* **Intrinsics**: `K = [[focal,0,cx],[0,focal,cy],[0,0,1]]` from the npz `img_focal` /
  `img_center`; `w,h = round(2·cx), round(2·cy)`. Poses were solved under this `focal`,
  so it is kept as-is for reprojection consistency.
* **Hand frame** = MANO **root (wrist)** pose (`init_trans`, `init_root_orient`). The
  true HumanEgo thumb/index "midpoint" frame needs the MANO layer (model + torch); we
  fill `midpoint_*` and `wrist_*` identically with the wrist pose — a consistent control
  point DatasetGen consumes unchanged. **Upgrade path:** run MANO in the `hawor` env to
  emit the real midpoint and fingertip keypoints.
* **Grasp** is a proxy: per-frame mean finger-joint flexion (geodesic angle of the 15
  `init_hand_pose` rotations), thresholded **relative to the clip's own range**
  (`--grasp_rel`, default 0.5) because MANO's pose is relative to a curled template.
  For a principled grasp, run MANO and use thumb-index tip distance (cf. AriaHands
  `GRASP_THRESHOLD = 0.105 m`).
* **Velocities** are written as zeros; recompute downstream if training consumes them.
* **Confidence** = 1.0 (cam_space carries no per-frame validity).

## Requirements to run the full pipeline (object branch)

The `dinosam → … → datasetgen` stages are HumanEgo's existing GPU pipeline and need:
1. **GPU + model weights** — Grounding DINO, SAM 2, CoTracker, LaMa. Pre-fetch with
   `PREDOWNLOAD=1 bash setup.sh` (they download from HuggingFace on first run otherwise).
2. **Object prompts** — open-vocabulary text per object in the task config
   (`cfg/preprocess/tasks/<task>.yaml`, `DINOSAM.dinosam_prompt`). `obj1` becomes the
   world anchor, so pick the most stable object in view. A `scilab.yaml` is provided.

## Validation status (on `scilab_sample/DJI_20260409180326_0364_D_A012`)

* ✅ Camera decode — frame 0 ≈ identity; hand reprojection lands on the real hands.
* ✅ Frame extraction — 30 fps grid, 2688×1512.
* ✅ Hand world poses — orthonormal `R` (det 1); both hands; grasp varies over the clip.
* ✅ Seam — HumanEgo `preprocess_indices` consumes the synthesized files (manip=full clip,
  object-centric window built, reference frame selected).
* ✅ Hand contract — HumanEgo `DatasetGen._build_hands_entry_from_method` parses both
  `aria_hands.json` (→ `hands`) and `hawor_hands.json` (→ `hands_hawor`) into valid
  `T_hand_to_world`.
* ⏳ `dinosam → datasetgen` — code path correct; not executed here (needs GPU + weights +
  network for model download). Run in a provisioned `humanego` env.
