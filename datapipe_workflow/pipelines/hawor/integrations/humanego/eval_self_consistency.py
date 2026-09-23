#!/usr/bin/env python3
"""B1: no-ground-truth self-consistency eval for a HaWoR->HumanEgo session.

Measures whether the HaWoR-derived geometry is internally consistent, using signals
that were produced independently of each other:

  * Hand reprojection vs arm mask: project the per-frame hand world point (from HaWoR,
    via that frame's camera c2w) into the image and test it against the SAM2 arm mask
    (produced by DINOSAM, which never saw the hand pose). Agreement => the HaWoR camera
    trajectory and hand poses are mutually consistent.
  * Object reprojection vs object mask: project each object's triangulated 3D point cloud
    (static in world) into every frame and test against that frame's SAM2 object mask.
    Agreement across the moving camera => camera trajectory + triangulated 3D are consistent.
  * Trajectory metric sanity: hand speed (m/s) from world positions + fps; should be
    human-plausible (no teleport jumps).
  * Metric scale: object point-cloud extent (m) and hand-to-camera distance (m).

These are necessary (not sufficient) conditions for good annotation: low reprojection
error + plausible metric trajectories => the front-end is geometrically sound. It cannot
detect a globally wrong-but-consistent frame; for that use the GT comparison (B2).

Usage:
    python eval_self_consistency.py --session <HumanEgo session dir> [--out report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np


def _load(frame_dir):
    cam = json.load(open(os.path.join(frame_dir, "aria_cam_rgb.json")))
    K = np.array(cam["k"], float)
    c2w = np.array(cam["c2w"], float)
    return K, c2w, cam["w"], cam["h"]


def _project(K, w2c, pts_w):
    """Project (N,3) world points to (N,2) pixels; returns px and in-front-of-cam mask."""
    pc = (w2c[:3, :3] @ pts_w.T + w2c[:3, 3:4]).T          # (N,3) camera frame
    z = pc[:, 2]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z > 0.05          # require >5cm in front of camera (drop degenerate proj)


def _dist_to_mask(uv, mask):
    """For each pixel uv, distance (px) to nearest mask>0 pixel; and inside-mask flag."""
    h, w = mask.shape
    inside, dists = [], []
    # distance transform of the BACKGROUND gives, at each bg pixel, dist to nearest fg.
    dt = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    for u, v in uv:
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            ins = mask[vi, ui] > 0
            inside.append(ins)
            dists.append(0.0 if ins else float(dt[vi, ui]))
    return np.array(inside, bool), np.array(dists, float)


def evaluate(session: str) -> dict:
    pre = os.path.join(session, "preprocess")
    frame_dirs = sorted(glob.glob(os.path.join(pre, "all_data", "*")))
    frame_dirs = [d for d in frame_dirs if os.path.exists(os.path.join(d, "rgb.png"))]

    tri = json.load(open(os.path.join(pre, "camtriangulator_results.json")))
    obj_pts = {k: np.array(v["points_3d_world"], float)
               for k, v in tri["objects"].items()}

    hand_inside, hand_dist = [], []
    obj_inside = {k: [] for k in obj_pts}
    obj_dist = {k: [] for k in obj_pts}
    hand_world_seq = {"left": [], "right": []}        # (idx, pos)
    cam_pos_seq = []
    diag_px = None

    for d in frame_dirs:
        idx = int(os.path.basename(d))
        K, c2w, W, H = _load(d)
        if diag_px is None:
            diag_px = float(np.hypot(W, H))
        w2c = np.linalg.inv(c2w)
        cam_pos_seq.append((idx, c2w[:3, 3].copy()))

        # --- hands vs arm mask ---
        hands = json.load(open(os.path.join(d, "aria_hands.json")))
        arm_p = os.path.join(d, "mask_arm.png")
        arm = cv2.imread(arm_p, 0) if os.path.exists(arm_p) else None
        for side_key, side in [("hand_l", "left"), ("hand_r", "right")]:
            h = hands.get(side_key)
            if not h or h.get("midpoint_translation_opt_world") is None:
                continue
            p = np.array(h["midpoint_translation_opt_world"], float)[None, :]
            hand_world_seq[side].append((idx, p[0]))
            if arm is not None:
                uv, front = _project(K, w2c, p)
                if front[0]:
                    ins, dd = _dist_to_mask(uv[front], arm)
                    if len(ins):
                        hand_inside.append(bool(ins[0]))
                        hand_dist.append(float(dd[0]))

        # --- objects vs object mask ---
        for k, pts in obj_pts.items():
            mp = os.path.join(d, f"mask_{k}.png")
            m = cv2.imread(mp, 0) if os.path.exists(mp) else None
            if m is None:
                continue
            uv, front = _project(K, w2c, pts)
            if front.sum() < 1:
                continue
            ins, dd = _dist_to_mask(uv[front], m)
            if len(ins):
                obj_inside[k].append(float(ins.mean()))
                obj_dist[k].append(float(np.median(dd)))

    # --- trajectory metric speed (m/s) ---
    fps = json.load(open(os.path.join(frame_dirs[0], "aria_cam_rgb.json")))["fps"]
    speed = {}
    for side, seq in hand_world_seq.items():
        if len(seq) < 2:
            continue
        seq = sorted(seq)
        idxs = np.array([s[0] for s in seq])
        pos = np.array([s[1] for s in seq])
        dt = np.clip(np.diff(idxs), 1, None) / fps
        v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
        speed[side] = {"median_m_s": float(np.median(v)),
                       "p95_m_s": float(np.percentile(v, 95)),
                       "max_m_s": float(v.max())}

    # --- metric scale ---
    scale = {}
    for k, pts in obj_pts.items():
        ext = pts.max(0) - pts.min(0)
        scale[k] = {"extent_m": [round(float(x), 4) for x in ext],
                    "diag_m": round(float(np.linalg.norm(ext)), 4)}
    cam_pos = np.array([p for _, p in cam_pos_seq])
    cam_path_len = float(np.linalg.norm(np.diff(cam_pos, axis=0), axis=1).sum())

    def _stat(a):
        a = np.array(a, float)
        a = a[np.isfinite(a)]
        return None if a.size == 0 else {
            "n": int(a.size), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
        }

    report = {
        "session": os.path.basename(session.rstrip("/")),
        "n_frames": len(frame_dirs),
        "image_diag_px": round(diag_px, 1),
        "hand_reproj_vs_arm_mask": {
            "frac_inside_mask": (round(float(np.mean(hand_inside)), 3)
                                 if hand_inside else None),
            "px_dist_to_mask": _stat(hand_dist),
            "px_dist_pct_of_diag": (round(100 * np.median(hand_dist) / diag_px, 2)
                                    if hand_dist else None),
        },
        "object_reproj_vs_obj_mask": {
            k: {"frac_points_inside_mask": _stat(obj_inside[k]),
                "median_px_dist_to_mask": _stat(obj_dist[k])}
            for k in obj_pts
        },
        "hand_speed": speed,
        "metric_scale": {"objects": scale, "camera_path_len_m": round(cam_path_len, 4)},
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="HumanEgo session dir (has preprocess/)")
    ap.add_argument("--out", default=None, help="Write JSON report here")
    args = ap.parse_args()
    rep = evaluate(args.session)
    print(json.dumps(rep, indent=2))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
