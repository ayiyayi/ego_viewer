"""Conservative detector checks for SAM tracks; no model dependencies.

Detection is already available on every frame. Missing detection is not proof
of failure, but unsupported propagation is bounded and never crosses a jump.
"""
import numpy as np


def area(box):
    b = np.asarray(box, dtype=float)
    return float(np.prod(np.maximum(b[2:] - b[:2], 0)))


def compatible(mask_box, detector_box):
    a, b = np.asarray(mask_box, float), np.asarray(detector_box, float)
    aa, ab = area(a), area(b)
    if min(aa, ab) <= 0 or not np.isfinite(np.r_[a, b]).all():
        return False
    inter = area(np.r_[np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])])
    return .15 <= aa / ab <= 3.5 and inter / ab >= .45 and inter / (aa + ab - inter) >= .15


def jumped(a, b):
    if a is None or b is None:
        return True
    aa, ab = area(a), area(b)
    if min(aa, ab) <= 0:
        return True
    delta = np.linalg.norm((np.asarray(a)[:2] + np.asarray(a)[2:] - np.asarray(b)[:2] - np.asarray(b)[2:]) / 2)
    return max(aa, ab) / min(aa, ab) > 3 or delta > max(np.sqrt(aa), np.sqrt(ab))


def trusted_detections(det_by_frame, n, side, threshold=.7):
    result = []
    for t in range(n):
        options = [r for r in det_by_frame.get(t, []) if int(r['side']) == side
                   and float(r['score']) >= threshold and area(r['box_xyxy']) > 0
                   and np.isfinite(r['box_xyxy']).all()]
        result.append(max(options, key=lambda r: float(r['score'])) if options else None)
    return result


def assess(boxes, detections, fps):
    """Direct checks at every detection; bridge <=0.5s only between verified frames."""
    n = len(boxes)
    direct = np.array([box is not None and det is not None and compatible(box, det['box_xyxy'])
                       for box, det in zip(boxes, detections)], bool)
    conflict = np.array([det is not None and not direct[t] for t, det in enumerate(detections)], bool)
    keep = direct.copy()
    anchors = np.flatnonzero(direct)
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a > round(.5 * fps) or conflict[a+1:b].any():
            continue
        if any(jumped(boxes[t-1], boxes[t]) for t in range(a+1, b+1)):
            continue
        keep[a+1:b] = True
    reasons = ['verified' if direct[t] else 'bounded_bridge' if keep[t]
               else 'detector_conflict' if conflict[t] else 'no_local_confirmation' for t in range(n)]
    return keep, reasons


def validate_and_recover(selected, det_by_frame, n, fps, recover):
    """Retry each failed half-second block at most once; validate retries too.

    recover(side, start, stop, anchor, detection) returns global-frame records.
    Failed records are discarded; no box-only fallback can reach SLAM/WiLoR.
    """
    result = dict(selected)
    blocked = np.ones((n, 2), bool)
    reports = []
    for side in (0, 1):
        detections = trusted_detections(det_by_frame, n, side)
        def boxes():
            return [result.get((t, side), {}).get('box_xyxy') for t in range(n)]
        before, _ = assess(boxes(), detections, fps)
        retries = 0
        recovery_anchors = []
        window = max(1, round(.5 * fps))
        for start in range(0, n, window):
            stop = min(n, start + window)
            if before[start:stop].all():
                continue
            candidates = [t for t in range(start, stop) if not before[t] and detections[t] is not None
                          and float(detections[t]['score']) >= .7]
            if not candidates:
                continue
            anchor = max(candidates, key=lambda t: float(detections[t]['score']))
            repaired = recover(side, start, stop, anchor, detections[anchor])
            retries += 1
            recovery_anchors.append(dict(frame=int(anchor), score=float(detections[anchor]["score"])))
            for t, rec in repaired.items():
                if start <= t < stop and not before[t] and rec is not None:
                    result[t, side] = rec
        keep, reasons = assess(boxes(), detections, fps)
        blocked[:, side] = ~keep
        for t in np.flatnonzero(~keep):
            result.pop((int(t), side), None)
        reports.append(dict(side=side, verified_before=int(before.sum()), kept=int(keep.sum()),
                            recovered=int((keep & ~before).sum()), blocked=int((~keep).sum()),
                            recovery_sessions=retries, recovery_anchors=recovery_anchors, reasons=reasons))
    return result, blocked, reports
