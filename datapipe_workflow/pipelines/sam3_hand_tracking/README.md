# SAM3 hand tracking

Ported from `HandActionData/scripts/ego4d_sam3_hawor/` at local commit
`e8633bab053a6daea3e4e761010cb86db04ce493`. The source checkout is unchanged.
`sam3_point.py` retains the point-prompt / propagation / mask-to-tight-box
implementation. `sam_anchor.py` and `sam_repropagate.py` contain only its needed
helpers; DataHub, OSS, scheduling, production environments and worker loops are
not imported. Model code is supplied through a separate SAM3 checkout.

## Flow

1. Detect at confidence 0.2 when SAM3 is enabled. Associate both hands jointly
   using center distance, IoU and detector side as a soft hint. Each candidate
   can serve at most one side. Missing frames emit no boxes.
2. Use the ported `box_below` prompt: box corners and five negative points below
   the box. Select detector seeds above **0.70** in 2-second windows. This is a
   detector seed threshold, not the SAM mask threshold (0.15).
3. Propagate forward/backward with `single_anchor`. Use separate sessions per
   side, so a clip without simultaneous confident left/right detections does
   not fail the upstream joint-initialization assertion. Process at most 600
   source frames per session. Actual source FPS is passed; no resampling.
4. For `annotate_video.py --hands wilor`, `resident_worker.py` loads SAM3 once
   for all 600-frame jobs across the video's camera chunks. Sessions remain
   separate per hand. The worker exits before SLAM so its GPU memory is released.
5. `wilor/camera_pipeline.py` unions left/right SAM masks, resizes and crops
   them to DROID's image geometry, and stores boolean masks. Foreground is
   excluded from both SLAM and Metric3D scale estimation. This path does not
   run HaWoR hand estimation, mesh-mask rendering, or the hand infiller.
6. WiLoR consumes the merged `sam_boxes.npz` and camera trajectory, followed
   by MANO v3 temporal processing (now the default). Existing jump rejection
   stays unchanged. Eligible gaps up to 0.3 seconds use MANO rotation SLERP
   and linear wrist/shape interpolation, followed by bounded wrist smoothing
   (at most 1.5 cm and 5 pixels). Camera breaks and endpoint motion/pose gates
   prevent unsafe interpolation. Finger poses on observed frames are retained.
   MANO parameters are captured during WiLoR inference, without a second pass.
   `output/raw_wilor/` preserves pre-filter geometry and MANO parameters;
   `processed_mano.npz` records accepted/interpolated masks and parameters,
   and `temporal_report.json` records counts and smoothing metrics.
   Viewer `keypoints.npz` / `mesh.bin` schemas are unchanged.

`--hands hawor` retains the full HaWoR path. The WiLoR camera path retains the
existing 100-second camera chunking and merge policy; it does not fix camera
discontinuities between chunks. SAM masks differ from rendered MANO masks, so
camera estimates can change and should be visually checked.

WiLoR outputs retain `camera_pipeline.log`, `stage_timings.json`,
`sam3_jobs.timings.json`, `hand_pipeline_timings.json`, and `camera_slam.npz`.
Work directories are retained under `$EGO_VIEWER_WORK_ROOT/pipeline_cache/` for
resume and reuse, including extracted frames and masks. They consume disk space;
remove an inactive cache directory manually when it is no longer needed.
`hand_pipeline_timings.json` records current-call elapsed time and cache-hit flags.
`stage_timings.json` retains the camera stages' timings; a cache hit is not a fresh
inference benchmark.
Direct calls to `wilor/export_viewer.py` need `--temporal-v3`; this preserves the
raw-export interface for callers that apply their own postprocessor. The normal
`annotate_video.py --hands wilor` entry point always enables v3.

Cold-start side identity still depends on detector labels. Association is a
heuristic; long absences, crossings, and SAM session boundaries can still need
improvement. This port is not an accuracy claim for FineBio.

## Enable

All three variables are required together (use actual installed paths):

```bash
export EGO_VIEWER_SAM3_CHECKPOINT=/path/to/sam3.pt
export EGO_VIEWER_SAM3_PYTHON=/path/to/separate-sam3-env/bin/python
export EGO_VIEWER_SAM3_SRC=/path/to/sam3-checkout
```

Do not install SAM3 or upgrade torch in `hawor` or `wilor`. The adapter rejects
these two interpreters for SAM3. It passes numeric NPZ detection data between
processes, not pickled objects. If no SAM3 variables are set, temporal detector
association works without SAM3. A partial/invalid configuration fails explicitly.

```bash
cd /data-hyp/ego_viewer/datapipe_workflow
/data/heyuping/ego_viewer/envs/hawor/bin/python annotate_video.py \
  /data-hyp/ego_viewer/hand_test_videos/P22_02_02.mp4 \
  --hands wilor --hands-only \
  --output /data/heyuping/ego_viewer/demo/P22_02_02_sam3_result
```

`--hands hawor` uses the same SAM3 boxes with the HaWoR pose model. Use a **new**
output directory. Scratch is allocated under `/data/heyuping/ego_viewer/demo`
(`EGO_VIEWER_WORK_ROOT` can override it); `/data-hyp` scratch is rejected.
Successful top-level runs copy viewer files and the small SAM box archive,
then remove scratch. Failures leave scratch/logs for diagnosis. The low-level
WiLoR exporter also accepts `--hand-boxes` or explicit `--sam3-checkpoint`,
`--sam3-python`, `--sam3-src` with `--frames` on the scratch disk.

`sam_boxes.npz` is an internal sidecar: `boxes[T,2,4]` absolute xyxy pixels,
`valid[T,2]`, `width`, `height`, `side_order='left,right'`. The viewer ignores it.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=datapipe_workflow \
/data/heyuping/ego_viewer/envs/hawor/bin/python \
  datapipe_workflow/tests/test_sam_hand_tracking.py -v
```

Tests cover ambiguous labels, confidence swaps, gaps/reacquisition, duplicate
assignment, mask tight boxes, missing masks, subprocess chunk indices, archive
validation, and viewer output compatibility. They use a fake SAM predictor;
real GPU inference and FineBio accuracy still require SAM3 weights/environment.

## Local runtime configuration (2026-09-24)

Load the concrete paths without changing the default environment:

```bash
source /data-hyp/ego_viewer/datapipe_workflow/env_profiles/sam3.env
```

| Setting | Path |
|---|---|
| SAM3 Python | `/data/heyuping/ego_viewer/envs/sam3/bin/python` |
| Official source | `/data/heyuping/ego_viewer/sam3` |
| Checkpoint destination | `/data/heyuping/ego_viewer/assets/weights/sam3/sam3.pt` |
| Download/build cache | `/data/heyuping/ego_viewer/cache/uv` |

Source commit: `2345a4ad109ac29c569da749c91d84f10dc08c40`, cloned from
[facebookresearch/sam3](https://github.com/facebookresearch/sam3).
Python 3.12 uses an independent venv with system site packages disabled.
Torch 2.10.0 / torchvision 0.25.0 use the official PyTorch CUDA 12.8 index.
`env_profiles/sam3-constraints.txt` constrains the runtime; setuptools is pinned
because this source imports `pkg_resources`.

The initial official checkpoint download was denied by Hugging Face because
this account lacks access to [facebook/sam3](https://huggingface.co/facebook/sam3).
The configured checkpoint path is a destination, not evidence that weights exist.
After approval, retry with the already authenticated CLI (no token in scripts):

```bash
bash -ic 'proxy_on >/dev/null; HF_ENDPOINT=https://huggingface.co hf download facebook/sam3 sam3.pt --local-dir /data/heyuping/ego_viewer/assets/weights/sam3'
```

Check imports, CUDA kernels, and checkpoint presence:

```bash
source /data-hyp/ego_viewer/datapipe_workflow/env_profiles/sam3.env
"$EGO_VIEWER_SAM3_PYTHON" /data-hyp/ego_viewer/datapipe_workflow/scripts/check_sam3_environment.py --require-checkpoint
```

This check does not run the model or download weights. Use a short video smoke
test once the checkpoint is present before attempting a full FineBio run.


## Track validation and local recovery

The WiLoR camera pipeline now passes `--validate-tracks`. It reuses the already
computed per-frame detector results; it does not run YOLO again for each failure.
Before exporting masks or crop boxes, both sides are checked independently:

- Detector confidence must be at least 0.70 to confirm SAM. Lower-confidence
  face false positives were observed in packaging, so 0.5 is insufficient.
- Require detector-box coverage >=0.45, IoU >=0.15 and SAM/detector box area
  ratio in [0.15, 3.5]. These are initial heuristic thresholds, not calibrated
  accuracy guarantees.
- Frames without a reliable detection can only survive between two confirmed
  frames at most 0.5 seconds apart, with no detection conflict or abrupt box
  jump. Area changes above 3x or large center jumps break this bridge. A jump
  with direct high-confidence detector agreement remains allowed.
- Each failed 0.5-second block can receive one fresh SAM session seeded by its
  strongest >=0.70 detection. Propagation stays inside that short block. The
  recovered masks undergo the same checks; earlier independently verified
  frames are retained. The initial pass still uses 600-frame sessions.
- Unconfirmed masks are zeroed before SLAM, and their crop boxes are removed
  before WiLoR. `sam_boxes.npz` includes `interpolation_blocked` (frames, 2),
  preventing v3 from filling across these failures. Visibility may decrease
  when the detector is uncertain; detection/SAM agreement cannot guarantee
  correct identity or reject a high-confidence false positive.

`tracking_validation.json` is retained in the final output, including per-frame
reasons and recovery counts. Existing annotations are not modified automatically;
a new run is necessary to update both hand estimates and the camera masks.


## Performance and reuse

`annotate_video.py --hands wilor` automatically reuses completed camera/SAM,
raw WiLoR/MANO, and temporal-output stages. Receipts check input file metadata,
configuration, relevant implementation source and output size/mtime. Model
weights use file metadata rather than expensive checkpoint hashing. Outputs
from older runs without receipts are not silently trusted. Camera work retains
split/detection receipts and per-SAM-job/per-SLAM-chunk receipts for resume.
Never edit intermediate files inside a managed cache.

- `--wilor-batch-size 16` (default): batch crops across frames, prefetch one CPU
  batch, preserve each crop's frame/side association. Use `2` to compare numerical
  behavior or reduce VRAM. Floating-point results need not be bit-identical.
- `--force-hands`: create a fresh cache and rerun hand/camera stages.
- `--postprocess-only`: use `output/raw_wilor/{keypoints,mano}.npz` on CPU. This
  also works for older v3 exports that saved raw MANO, without any cache receipt.
  It skips captions, detection, SAM, SLAM and the WiLoR neural network.

Example, from `datapipe_workflow`:

```bash
python annotate_video.py examples/ego_hand_04_packaging/input/ego_hand_04_packaging.mp4 \
  --output examples/ego_hand_04_packaging --postprocess-only
```

SAM workers reuse the backend's exact preprocessed tensor for the second hand
within a job; local recovery sessions remain independent. Compact mode stores
intermediate masks as lossless packed bits with fast zlib compression and writes
SLAM-resolution union masks directly. It avoids full-resolution framewise PNG
exports/reloads. Resize/crop, frame rate, detector thresholds and validation
rules are unchanged. Standalone SAM runs retain the PNG interface unless
`--compact-masks` is passed.
