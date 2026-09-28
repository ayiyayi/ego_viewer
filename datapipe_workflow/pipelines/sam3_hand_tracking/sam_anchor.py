"""Minimal helpers ported from HandActionData sam_anchor.py."""
import cv2
from collections import defaultdict
import numpy as np
import torch
SIDE_NAMES = {0: "left", 1: "right"}

def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def load_detection(path):
    with np.load(path, allow_pickle=False) as archive:
        detection = {key: archive[key].copy() for key in archive.files}
    return {
        "frame_idx": as_numpy(detection["frame_idx"]).astype(np.int64),
        "track_id": as_numpy(detection["track_id"]).astype(np.int64),
        "boxes_xyxy": as_numpy(detection["boxes_xyxy"]).astype(np.float32),
        "scores": as_numpy(detection["scores"]).astype(np.float32),
        "handedness": as_numpy(detection["handedness"]).astype(np.int64),
    }

def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom

def xywh_rel_to_xyxy_abs(box_xywh, width, height):
    x, y, w, h = [float(v) for v in box_xywh]
    return np.array([x * width, y * height, (x + w) * width, (y + h) * height], dtype=np.float32)

def xyxy_abs_to_xywh_rel(box_xyxy, width, height):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return np.array([x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height], dtype=np.float32)

def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1.0, ys.max() + 1.0], dtype=np.float32)

def sorted_frames(segment_dir):
    frames = sorted((segment_dir / "frames").glob("*.jpg"))
    assert len(frames) > 0, f"no jpg frames in {segment_dir / 'frames'}"
    return frames

def detection_index(det):
    by_frame = defaultdict(list)
    for frame, tid, box, score, side in zip(det["frame_idx"], det["track_id"], det["boxes_xyxy"], det["scores"], det["handedness"]):
        by_frame[int(frame)].append({"track_id": int(tid), "box_xyxy": box.astype(np.float32), "score": float(score), "side": int(side)})
    return by_frame

def normalize_outputs(outputs):
    obj_ids = as_numpy(outputs["out_obj_ids"]).astype(np.int64)
    boxes = as_numpy(outputs["out_boxes_xywh"]).astype(np.float32)
    masks = as_numpy(outputs["out_binary_masks"])
    probs = as_numpy(outputs.get("out_probs", np.ones((len(obj_ids),), dtype=np.float32))).astype(np.float32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.astype(bool)
    return obj_ids, boxes, masks, probs

def build_predictor(checkpoint_path, default_output_prob_thresh, sam_version="sam3"):
    if sam_version == "sam3":
        from sam3.model_builder import build_sam3_video_predictor

        kwargs = dict(compile=False, async_loading_frames=False)
        if checkpoint_path:
            kwargs["checkpoint_path"] = checkpoint_path
        predictor = build_sam3_video_predictor(**kwargs)
        predictor.default_output_prob_thresh = float(default_output_prob_thresh)
        return predictor
    if sam_version == "sam3.1":
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        kwargs = dict(use_fa3=False, compile=False, warm_up=False, async_loading_frames=False, default_output_prob_thresh=float(default_output_prob_thresh))
        if checkpoint_path:
            kwargs["checkpoint_path"] = checkpoint_path
        return build_sam3_multiplex_video_predictor(**kwargs)
    raise AssertionError(f"unknown SAM version: {sam_version}")

