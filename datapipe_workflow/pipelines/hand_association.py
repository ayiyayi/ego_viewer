"""Two-hand image-space association; detector side labels are hints after initialization."""
from dataclasses import dataclass
from itertools import product
import numpy as np


def _iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    inter = np.maximum(0, hi - lo).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - inter
    return float(inter / max(float(union), 1e-6))


@dataclass
class SideTrack:
    box: object = None
    missing: int = 0


class ContinuityAssociator:
    """Assign candidates jointly so one detection cannot represent both hands.

    Cold starts use the detector label. Established tracks may survive a class
    flip, but reject distant boxes. A rejected/missing frame stays missing; no
    interpolated or stale boxes are emitted. Identity after a long absence must
    be reinitialized and is not guaranteed by this image-space heuristic.
    """
    def __init__(self, sides=(0, 1), max_missing=12, distance_gate=0.18):
        self.tracks = {int(s): SideTrack() for s in sides}
        self.max_missing = int(max_missing)
        self.distance_gate = float(distance_gate)

    def update(self, candidates_by_side, image_shape):
        candidates = []
        for label, items in candidates_by_side.items():
            if label not in self.tracks:
                continue
            for item in items:
                box = np.asarray(item['box'], dtype=np.float32)
                if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                    continue
                candidates.append((label, dict(item, box=box.copy())))
        diag = max(1., float(np.hypot(*image_shape[:2])))
        choices = []
        for side, track in self.tracks.items():
            opts = [(None, 0.8)]
            for idx, (label, item) in enumerate(candidates):
                score = float(item.get('score', 0.))
                if track.box is None:
                    if label != side:
                        continue
                    cost = 0.5 - 0.2 * score
                else:
                    box = item['box']
                    dist = float(np.linalg.norm((box[:2] + box[2:] - track.box[:2] - track.box[2:]) * .5)) / diag
                    if dist > self.distance_gate:
                        continue
                    cost = dist + .25 * (1 - _iou(track.box, box)) + .08 * (label != side) - .02 * score
                opts.append((idx, cost))
            choices.append(opts)
        best, best_cost = None, float('inf')
        for assignment in product(*choices):
            used = [idx for idx, _ in assignment if idx is not None]
            if len(used) != len(set(used)):
                continue
            # Suppress overlapping duplicate detections even when their labels differ.
            if any(_iou(candidates[a][1]['box'], candidates[b][1]['box']) > .8
                   for i, a in enumerate(used) for b in used[i+1:]):
                continue
            cost = sum(c for _, c in assignment)
            if cost < best_cost:
                best, best_cost = assignment, cost
        result = {}
        for (side, track), (idx, _) in zip(self.tracks.items(), best):
            if idx is None:
                track.missing += 1
                if track.missing >= self.max_missing:
                    track.box = None
                result[side] = None
            else:
                item = candidates[idx][1]
                track.box = item['box'].copy()
                track.missing = 0
                result[side] = item
        return result
