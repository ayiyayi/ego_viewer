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
4. HaWoR consumes the SAM mask tight boxes, then saves `sam_boxes.npz`. The
   segmented runner merges this archive in source-frame order. WiLoR reuses it
   without running detection or SAM again. Missing masks remain missing.
5. Existing viewer `keypoints.npz` / `mesh.bin` schemas and cameras are unchanged.

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
