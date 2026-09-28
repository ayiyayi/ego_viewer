"""Run WiLoR on one video and write the viewer hand archive.

The upstream demo renders a jpg and, with --save_mesh, one obj per detection.
Those files are not what the viewer reads. This script keeps only:

  output/keypoints.npz
  output/mesh.bin

Hand pose comes from WiLoR, in the camera frame. The camera trajectory comes
from a HaWoR SLAM npz (masked DROID-SLAM, metric scale from Metric3D). Each
camera-frame point is lifted with that frame's pose:

  p_world = R_c2w @ p_cam + t_c2w

WiLoR's own focal constant is not used. The translation that places the hand
in front of the camera is computed with the SLAM img_focal, and that same
focal is written for the ego overlay.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
from pipelines.hand_association import ContinuityAssociator

WILOR_ROOT = Path("/data/heyuping/ego_viewer/WiLoR")
FALLBACK_DETECTOR = Path("/data/heyuping/ego_viewer/assets/weights/external/detector.pt")
HAWOR_EXPORT = Path(
    "/data-hyp/ego_viewer/datapipe_workflow/pipelines/hawor/scripts/export_openaoe_hands.py"
)


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length):
    """Full-image camera translation. Local copy so pyrender is not imported."""
    import torch

    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def align_pred_cam(pred_cam, right):
    """Put the crop-camera x shift back into the full image.

    Left-hand crops are mirrored before WiLoR. pred_cam[:, 1] is that mirrored
    x shift. The official demo multiplies it by (2*right-1) before
    cam_crop_to_full. Joints and vertices are mirrored separately.
    """
    cam = pred_cam.clone()
    sign = (2 * right.float() - 1).view(-1)
    cam[:, 1] = cam[:, 1] * sign
    return cam


def probe_video(video: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise SystemExit(f"video has no frames: {video}")
    if fps <= 0:
        fps = 30.0
    return width, height, fps, n


def select_hands(result, associator=None, image_shape=None):
    """One box per side, with temporal continuity when an associator is given."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        if associator is not None:
            associator.update({}, image_shape)
        return None, None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    sides = boxes.cls.detach().cpu().numpy().astype(np.int32)
    scores = boxes.conf.detach().cpu().numpy()
    candidates = {0: [], 1: []}
    for box, side, score in zip(xyxy, sides, scores):
        if side not in (0, 1):
            continue
        b = box.astype(np.float32)
        candidates[int(side)].append({
            "box": b,
            "score": float(score),
            "area": float(max(0., b[2] - b[0]) * max(0., b[3] - b[1])),
        })
    if associator is not None:
        chosen = associator.update(candidates, image_shape)
        chosen = {s: (float(v["score"]), v["box"], float(s)) for s, v in chosen.items() if v is not None}
    else:
        chosen = {
            side: max(items, key=lambda x: x["score"])
            for side, items in candidates.items() if items
        }
        chosen = {s: (float(v["score"]), v["box"], float(s)) for s, v in chosen.items()}
    if not chosen:
        return None, None
    order = sorted(chosen)
    return (
        np.stack([chosen[side][1] for side in order]),
        np.asarray([chosen[side][2] for side in order], dtype=np.float32),
    )


def load_slam(path: Path):
    """HaWoR camera track in the same convention as the viewer export."""
    spec = importlib.util.spec_from_file_location("export_openaoe_hands", HAWOR_EXPORT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {HAWOR_EXPORT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r_c2w, t_c2w, r_w2c, t_w2c, focal = module.load_camera(path)
    t_c2w = np.asarray(t_c2w, dtype=np.float32)
    t_w2c = np.asarray(t_w2c, dtype=np.float32)
    if t_c2w.ndim == 3:
        t_c2w = t_c2w[..., 0]
    if t_w2c.ndim == 3:
        t_w2c = t_w2c[..., 0]
    if focal <= 1.0:
        raise SystemExit(f"SLAM img_focal is {focal} in {path}")
    return (
        np.asarray(r_c2w, dtype=np.float32),
        t_c2w,
        np.asarray(r_w2c, dtype=np.float32),
        t_w2c,
        float(focal),
    )


def camera_to_world(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Camera-frame points (..., 3) to world, with R_c2w and t_c2w."""
    return np.einsum("ij,...j->...i", rotation, points) + translation


def list_frames(frames_dir: Path) -> list[Path]:
    paths = [path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(paths, key=lambda path: int(path.stem) if path.stem.isdigit() else path.name)


def write_viewer(
    out_dir: Path,
    joints,
    verts,
    valid,
    faces,
    focal,
    width,
    height,
    fps,
    r_w2c,
    t_w2c,
    r_c2w,
    t_c2w,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    keypoints = out_dir / "keypoints.npz"
    mesh = out_dir / "mesh.bin"
    cam_z = np.ascontiguousarray(r_c2w[:, :, 2], dtype=np.float32)
    np.savez_compressed(
        keypoints,
        joints_world=joints,
        pred_valid=valid,
        R_w2c=np.ascontiguousarray(r_w2c, dtype=np.float32),
        t_w2c=np.ascontiguousarray(t_w2c, dtype=np.float32),
        cam_pos=np.ascontiguousarray(t_c2w, dtype=np.float32),
        cam_z=cam_z,
        cam_R=np.ascontiguousarray(r_c2w, dtype=np.float32),
        focal=np.float32(focal),
        width=np.int32(width),
        height=np.int32(height),
        fps=np.float32(fps),
    )
    packed = np.ascontiguousarray(verts.astype(np.float16))
    faces = np.ascontiguousarray(faces.astype(np.int32))
    n_frames = int(joints.shape[1])
    with mesh.open("wb") as handle:
        handle.write(struct.pack("<III", n_frames, packed.shape[2], faces.shape[0]))
        handle.write(faces.tobytes())
        handle.write(packed.tobytes())
    return keypoints, mesh


def iter_frames(video: Path, frames_dir: Path | None, limit: int):
    if frames_dir is not None:
        paths = list_frames(frames_dir)[:limit]
        if not paths:
            raise SystemExit(f"no frames in {frames_dir}")
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise SystemExit(f"cannot read {path}")
            yield frame
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    seen = 0
    while seen < limit:
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        yield frame
    cap.release()


def export_video(
    video: Path,
    out_dir: Path,
    checkpoint: Path,
    detector_path: Path,
    rescale_factor: float,
    slam_path: Path,
    frames_dir: Path | None,
    sam3_checkpoint: Path | None = None,
    sam3_python: Path | None = None,
    sam3_src: Path | None = None,
    hand_boxes: Path | None = None,
) -> tuple[Path, Path]:
    import torch
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor

    os.chdir(WILOR_ROOT)
    width, height, fps, _ = probe_video(video)
    r_c2w, t_c2w, r_w2c, t_w2c, focal = load_slam(slam_path)
    n = int(r_c2w.shape[0])
    print(f"slam frames={n} focal={focal:.2f}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = None if hand_boxes is not None else YOLO(str(detector_path))
    sam_boxes = None
    if hand_boxes is not None:
        from pipelines.sam3_hand_tracking.refine import load_box_archive
        stored, mask = load_box_archive(hand_boxes, n, width, height)
        sam_boxes = [{side: stored[t, side] if mask[t, side] else None for side in (0, 1)} for t in range(n)]
    if sam3_checkpoint is not None:
        if hand_boxes is not None or sam3_python is None or sam3_src is None:
            raise ValueError('SAM3 needs explicit --sam3-python and --sam3-src; do not combine with --hand-boxes')
        if frames_dir is None:
            raise SystemExit("--sam3-checkpoint requires --frames so SAM3 can read an aligned frame directory")
        from pipelines.sam3_hand_tracking.refine import refine_boxes_with_sam3
        raw = []
        pre_assoc = ContinuityAssociator()
        frame_paths = list_frames(frames_dir)[:n]
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise SystemExit(f"cannot read {frame_path}")
            result = detector(frame, conf=0.2, verbose=False)[0]
            boxes0 = result.boxes
            by_side = {0: [], 1: []}
            if boxes0 is not None:
                xyxy0 = boxes0.xyxy.detach().cpu().numpy()
                cls0 = boxes0.cls.detach().cpu().numpy().astype(np.int32)
                conf0 = boxes0.conf.detach().cpu().numpy()
                for b, s, c in zip(xyxy0, cls0, conf0):
                    if int(s) in by_side:
                        by_side[int(s)].append({"box": b.astype(np.float32), "score": float(c)})
            raw.append(pre_assoc.update(by_side, frame.shape))
        detector.to("cpu")
        torch.cuda.empty_cache()
        sam_boxes = refine_boxes_with_sam3(
            frames_dir, raw, frames_dir.parent / "_sam3_refine",
            sam3_checkpoint, sam3_python, sam3_src, fps=fps,
        )

    # Release the detector before loading the hand model after SAM refinement.
    if sam_boxes is not None:
        del detector
        detector = None
        torch.cuda.empty_cache()
    model, model_cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(WILOR_ROOT / "pretrained_models" / "model_config.yaml"))
    model = model.to(device).eval()
    faces = np.asarray(model.mano.faces, dtype=np.int32)
    n_verts = int(model.mano.v_template.shape[0])

    joints = np.zeros((2, n, 21, 3), dtype=np.float32)
    verts = np.zeros((2, n, n_verts, 3), dtype=np.float32)
    valid = np.zeros((2, n), dtype=np.float32)

    frame_i = 0
    associator = ContinuityAssociator()
    for frame in iter_frames(video, frames_dir, n):
        boxes, right = None, None
        if sam_boxes is None:
            boxes, right = select_hands(detector(frame, conf=0.3, verbose=False)[0],
                                        associator=associator, image_shape=frame.shape)
        else:
            if frame_i >= len(sam_boxes):
                raise ValueError('SAM3 timeline shorter than decoded frames')
            refined = sam_boxes[frame_i]
            rows = [(side, refined.get(side)) for side in (0, 1) if refined.get(side) is not None]
            if rows:
                boxes = np.stack([np.asarray(box, dtype=np.float32) for _, box in rows])
                right = np.asarray([float(side) for side, _ in rows], dtype=np.float32)
        if boxes is not None:
            dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=rescale_factor)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            for batch in loader:
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                with torch.inference_mode():
                    out = model(batch)
                img_size = batch["img_size"].float()
                cam = align_pred_cam(out["pred_cam"], batch["right"])
                cam_t = cam_crop_to_full(
                    cam,
                    batch["box_center"].float(),
                    batch["box_size"].float(),
                    img_size,
                    focal,
                )
                pred_j = out["pred_keypoints_3d"][:, :21].detach()
                pred_v = out["pred_vertices"].detach()
                sign = (2 * batch["right"].float() - 1).view(-1, 1)
                pred_j = pred_j.clone()
                pred_v = pred_v.clone()
                pred_j[:, :, 0] *= sign
                pred_v[:, :, 0] *= sign
                pred_j = pred_j + cam_t[:, None, :]
                pred_v = pred_v + cam_t[:, None, :]
                sides = batch["right"].detach().cpu().numpy().astype(np.int32)
                pred_j = pred_j.cpu().numpy()
                pred_v = pred_v.cpu().numpy()
                rotation = r_c2w[frame_i]
                translation = t_c2w[frame_i]
                for row, side in enumerate(sides):
                    if side not in (0, 1):
                        continue
                    joints[side, frame_i] = camera_to_world(pred_j[row], rotation, translation)
                    verts[side, frame_i] = camera_to_world(pred_v[row], rotation, translation)
                    valid[side, frame_i] = 1.0
        frame_i += 1
        if frame_i % 30 == 0:
            print(f"frame {frame_i}", flush=True)
    if frame_i == 0:
        raise SystemExit(f"video has no frames: {video}")
    if frame_i != n:
        print(f"frames read {frame_i}, slam {n}; writing the overlap", flush=True)
    joints = joints[:, :frame_i]
    verts = verts[:, :frame_i]
    valid = valid[:, :frame_i]
    keypoints, mesh = write_viewer(
        out_dir,
        joints,
        verts,
        valid,
        faces,
        focal,
        width,
        height,
        fps,
        r_w2c[:frame_i],
        t_w2c[:frame_i],
        r_c2w[:frame_i],
        t_c2w[:frame_i],
    )
    print(
        f"wrote {keypoints} {keypoints.stat().st_size} bytes, "
        f"{mesh} {mesh.stat().st_size} bytes, frames={frame_i} "
        f"valid_left={valid[0].mean():.3f} valid_right={valid[1].mean():.3f}",
        flush=True,
    )
    return keypoints, mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a WiLoR video as viewer keypoints and mesh.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt")
    parser.add_argument("--detector", type=Path, default=WILOR_ROOT / "pretrained_models" / "detector.pt")
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--slam", type=Path, required=True, help="HaWoR SLAM npz with traj, scale, and img_focal.")
    parser.add_argument("--frames", type=Path, default=None, help="Jpg sequence aligned with the SLAM track.")
    parser.add_argument("--sam3-checkpoint", type=Path, default=None, help="Optional SAM3 checkpoint for tight-box refinement.")
    parser.add_argument("--sam3-python", type=Path, default=None, help="Python interpreter in the separate SAM3 environment.")
    parser.add_argument("--sam3-src", type=Path, default=None, help="SAM3 source checkout used by the separate interpreter.")
    parser.add_argument("--hand-boxes", type=Path, help="Aligned SAM3 box archive from HaWoR, left/right order.")
    args = parser.parse_args()

    if str(WILOR_ROOT) not in sys.path:
        sys.path.insert(0, str(WILOR_ROOT))
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"WiLoR checkpoint not found: {checkpoint}")
    detector = args.detector.expanduser().resolve()
    if not detector.is_file():
        detector = FALLBACK_DETECTOR
    if not detector.is_file():
        raise SystemExit(f"hand detector not found: {args.detector}")
    slam = args.slam.expanduser().resolve()
    if not slam.is_file():
        raise SystemExit(f"SLAM track not found: {slam}")
    frames = args.frames.expanduser().resolve() if args.frames is not None else None
    if frames is not None and not frames.is_dir():
        raise SystemExit(f"frame folder not found: {frames}")
    export_video(
        video,
        args.output_dir.expanduser().resolve(),
        checkpoint,
        detector,
        args.rescale_factor,
        slam,
        frames,
        sam3_checkpoint=args.sam3_checkpoint.expanduser().resolve() if args.sam3_checkpoint else None,
        sam3_python=args.sam3_python.expanduser().resolve() if args.sam3_python else None,
        sam3_src=args.sam3_src.expanduser().resolve() if args.sam3_src else None,
        hand_boxes=args.hand_boxes.expanduser().resolve() if args.hand_boxes else None,
    )


if __name__ == "__main__":
    main()
