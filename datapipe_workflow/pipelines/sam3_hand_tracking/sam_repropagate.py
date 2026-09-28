"""Mask geometry helpers ported from HandActionData sam_repropagate.py."""
import numpy as np
from .sam_anchor import (SIDE_NAMES, build_predictor, detection_index, load_detection, mask_to_box, normalize_outputs, sorted_frames, xywh_rel_to_xyxy_abs, xyxy_abs_to_xywh_rel, xyxy_iou)

def side_probs_from_label_score(det):
    # Current detections have one score, so use it as a temporary side proxy.
    score = float(det["score"])
    if int(det["side"]) == 0:
        return score, 1.0 - score
    return 1.0 - score, score

def side_margin(det):
    prob_left, prob_right = side_probs_from_label_score(det)
    return abs(prob_left - prob_right)

def box_area_xyxy(box_xyxy):
    x1, y1, x2, y2 = [float(x) for x in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def output_box_xyxy(box_rel, mask, width, height):
    box = xywh_rel_to_xyxy_abs(box_rel, width, height)
    if int(mask.sum()) > 0 and ((box[2] - box[0]) <= 1 or (box[3] - box[1]) <= 1):
        box = mask_to_box(mask)
    return box.astype(np.float32)

