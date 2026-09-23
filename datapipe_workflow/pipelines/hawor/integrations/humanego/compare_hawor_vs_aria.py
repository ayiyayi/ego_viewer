#!/usr/bin/env python3
"""B2: quantify HaWoR's front-end against Aria MPS ground truth (serve_bread).

Compares a HaWoR annotation of a clip to the Aria MPS ground truth recorded for the
SAME clip:

  * Camera/device trajectory: HaWoR masked-DROID-SLAM vs MPS closed-loop trajectory.
    The two live in different world frames at different scales, so we estimate a single
    Sim(3) (scale + rotation + translation) that best maps HaWoR -> Aria from the camera
    centers (Umeyama), apply it, and report ATE (absolute trajectory error, m) and the
    recovered scale.
  * Hand: HaWoR wrist (world) vs MPS wrist (world = T_world_device @ wrist_in_device),
    after the SAME Sim(3). Reports position error (cm) and orientation error (deg).

Temporal association: HaWoR frame i (extracted at `--hawor_fps`, default 30) maps to a
time t = i / hawor_fps; the nearest Aria timestamp is matched. If you extracted RGB with
a per-frame timestamp map (frames_ts.json from extract_vrs_rgb.py), pass it via
--frames_ts for exact association.

Prerequisite (not runnable in the humanego env — needs the hawor env):
    1. python extract_vrs_rgb.py --vrs <serve_bread>/sample.vrs --out /tmp/sb_rgb --fps 30
    2. (hawor env) python scripts/hawor_video_processor.py \
           --video_path /tmp/sb_rgb/video.mp4 --output_dir /tmp/sb_hawor/sb --gpu_ids 0
    3. python compare_hawor_vs_aria.py \
           --aria_session <serve_bread> --hawor_dir /tmp/sb_hawor/sb [--frames_ts /tmp/sb_rgb/frames_ts.json]

Self-test (no HaWoR needed): --selftest feeds the Aria GT in as the "HaWoR" side and
checks the metrics are ~0, proving the alignment + metric code is correct.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Quaternion / Sim(3) helpers
# --------------------------------------------------------------------------- #
def quat_xyzw_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def geodesic_deg(Ra, Rb):
    R = Ra.transpose(0, 2, 1) @ Rb
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def umeyama_sim3(src, dst):
    """Least-squares Sim(3): find s,R,t with dst ~ s*R@src + t. src,dst: (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


# --------------------------------------------------------------------------- #
# Aria MPS ground-truth loaders
# --------------------------------------------------------------------------- #
def load_aria_camera(session):
    """closed_loop_trajectory.csv -> (ts_us[N], T_world_device[N,4,4])."""
    p = os.path.join(session, "slam", "closed_loop_trajectory.csv")
    ts, Ts = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            ts.append(int(row["tracking_timestamp_us"]))
            t = np.array([float(row[f"t{a}_world_device"]) for a in "xyz"])
            q = [float(row[f"q{a}_world_device"]) for a in ["x", "y", "z", "w"]]
            T = np.eye(4); T[:3, :3] = quat_xyzw_to_R(q); T[:3, 3] = t
            Ts.append(T)
    return np.array(ts), np.array(Ts)


def load_aria_hand_world(session, ts_cam, T_world_device):
    """hand_tracking_results.csv wrist (device frame) -> world via matched T_world_device.

    Returns {side: (ts[N], pos_world[N,3], R_world[N,3,3])}.
    """
    p = os.path.join(session, "hand_tracking", "hand_tracking_results.csv")
    out = {"left": ([], [], []), "right": ([], [], [])}
    with open(p) as f:
        for row in csv.DictReader(f):
            tus = int(row["tracking_timestamp_us"])
            j = int(np.argmin(np.abs(ts_cam - tus)))
            Twd = T_world_device[j]
            for side in ("left", "right"):
                qw = row.get(f"qw_{side}_device_wrist", "")
                if qw == "" or float(qw) == 0:
                    continue
                t = np.array([float(row[f"t{a}_{side}_device_wrist"]) for a in "xyz"])
                q = [float(row[f"q{a}_{side}_device_wrist"]) for a in ["x", "y", "z", "w"]]
                Tdw = np.eye(4); Tdw[:3, :3] = quat_xyzw_to_R(q); Tdw[:3, 3] = t
                Twrist = Twd @ Tdw
                out[side][0].append(tus)
                out[side][1].append(Twrist[:3, 3])
                out[side][2].append(Twrist[:3, :3])
    return {s: (np.array(a), np.array(b), np.array(c)) for s, (a, b, c) in out.items()
            if a}


# --------------------------------------------------------------------------- #
# HaWoR loaders (reuse the adapter's decoders)
# --------------------------------------------------------------------------- #
def load_hawor(hawor_dir, hawor_fps, ts_cam):
    """HaWoR SLAM npz + cam_space -> camera centers and wrist poses, time-matched to Aria.

    Returns (cam_t_world[N,3], hand{side:(pos[N,3],R[N,3,3])}, matched_aria_idx[N]).
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "hawor_to_humanego", os.path.join(here, "hawor_to_humanego.py"))
    A = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = A
    spec.loader.exec_module(A)

    inner = hawor_dir
    npz = sorted(glob.glob(os.path.join(inner, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not npz:
        inner = os.path.join(hawor_dir, os.path.basename(hawor_dir.rstrip("/")))
        npz = sorted(glob.glob(os.path.join(inner, "SLAM", "hawor_slam_w_scale_*.npz")))
    cam = A.decode_slam_cameras(npz[0])
    cam_space = os.path.join(inner, "cam_space")
    hands = {}
    for sub, side in A.CAM_SPACE_SIDE.items():
        cand = sorted(glob.glob(os.path.join(cam_space, sub, "*.json")))
        hands[side] = A.decode_cam_space_hand(cand[0]) if cand else None

    n = cam.n_frames
    t0 = ts_cam.min()
    aria_idx = np.array([int(np.argmin(np.abs(ts_cam - (t0 + i * 1e6 / hawor_fps))))
                         for i in range(n)])
    cam_t = cam.c2w[:, :3, 3]
    hand_out = {}
    for side, tr in hands.items():
        if tr is None:
            continue
        pos, R = [], []
        for i in range(n):
            Rw, tw = A.lift_pose_to_world(cam.c2w[i], tr.R_cam[i], tr.trans_cam[i])
            pos.append(tw); R.append(Rw)
        hand_out[side] = (np.array(pos), np.array(R))
    return cam_t, hand_out, aria_idx


# --------------------------------------------------------------------------- #
def compare(aria_session, hawor_dir, hawor_fps, frames_ts=None, selftest=False):
    ts_cam, Twd = load_aria_camera(aria_session)
    aria_cam_t = Twd[:, :3, 3]
    aria_hand = load_aria_hand_world(aria_session, ts_cam, Twd)

    if selftest:
        # Feed Aria GT in as the "HaWoR" side, sampled on the camera timeline (as the
        # real HaWoR path is: hand frame i shares frame i's time with camera i). Errors
        # must be ~0, validating the Sim(3) + cross-modal (cam-ts -> hand-ts) matching.
        haw_cam_t = aria_cam_t.copy()
        aria_idx = np.arange(len(ts_cam))
        haw_hand = {}
        for side, (a_ts, a_pos, a_R) in aria_hand.items():
            gi = np.array([int(np.argmin(np.abs(a_ts - tc))) for tc in ts_cam])
            haw_hand[side] = (a_pos[gi].copy(), a_R[gi].copy())
    else:
        haw_cam_t, haw_hand, aria_idx = load_hawor(hawor_dir, hawor_fps, ts_cam)

    # Sim(3) from camera centers (HaWoR -> Aria), using time-matched pairs.
    src = haw_cam_t
    dst = aria_cam_t[aria_idx]
    s, R, t = umeyama_sim3(src, dst)
    haw_cam_agn = (s * (R @ src.T).T + t)
    ate = np.linalg.norm(haw_cam_agn - dst, axis=1)

    report = {
        "aria_session": os.path.basename(aria_session.rstrip("/")),
        "selftest": selftest,
        "n_hawor_frames": int(len(src)),
        "sim3_scale": float(s),
        "camera_ATE_m": {"rmse": float(np.sqrt((ate ** 2).mean())),
                         "median": float(np.median(ate)),
                         "max": float(ate.max())},
        "hands": {},
    }
    for side, (hp, hR) in haw_hand.items():
        if side not in aria_hand:
            continue
        a_ts, a_pos, a_R = aria_hand[side]
        # match each hawor frame's aria index into the hand timeline
        gi = np.array([int(np.argmin(np.abs(a_ts - ts_cam[aria_idx[i]])))
                       for i in range(len(hp))])
        hp_agn = (s * (R @ hp.T).T + t)
        pos_err = np.linalg.norm(hp_agn - a_pos[gi], axis=1)
        hR_agn = R @ hR
        rot_err = geodesic_deg(hR_agn, a_R[gi])
        report["hands"][side] = {
            "n": int(len(hp)),
            "pos_err_cm": {"median": float(np.median(pos_err) * 100),
                           "rmse": float(np.sqrt((pos_err ** 2).mean()) * 100)},
            "rot_err_deg": {"median": float(np.median(rot_err)),
                            "rmse": float(np.sqrt((rot_err ** 2).mean()))},
        }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aria_session", required=True, help="serve_bread mps_*_vrs dir (has slam/, hand_tracking/)")
    ap.add_argument("--hawor_dir", default=None, help="HaWoR output dir (SLAM/, cam_space/)")
    ap.add_argument("--hawor_fps", type=float, default=30.0)
    ap.add_argument("--frames_ts", default=None, help="frames_ts.json from extract_vrs_rgb.py (exact assoc)")
    ap.add_argument("--selftest", action="store_true", help="GT-vs-GT sanity check (errors ~0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = compare(args.aria_session, args.hawor_dir, args.hawor_fps,
                  args.frames_ts, args.selftest)
    print(json.dumps(rep, indent=2))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
