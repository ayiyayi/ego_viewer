#!/usr/bin/env python3
"""HaWoR -> HumanEgo ingest adapter.

Turns a HaWoR-annotated egocentric clip (the ``scilab_sample`` layout) into a
HumanEgo *session* directory whose ``preprocess/`` folder looks exactly like the
output of HumanEgo's ``aria`` stage -- so the rest of the HumanEgo Aria pipeline
(``indices -> dinosam -> kptsselector -> cotracker -> camtriangulator -> lama ->
visualkpts -> datasetgen``) can run **without Project Aria / MPS**.

Why this exists
---------------
HumanEgo's geometric front-end (camera trajectory + per-frame hand poses) is
normally produced by Aria MPS. Its alternative hand methods (WiLoR / HaMeR /
MediaPipe) still piggy-back on Aria for camera poses and the world frame, so they
are **not** usable on non-Aria footage. HaWoR, by contrast, reconstructs *both*
the camera trajectory (its masked DROID-SLAM, metric-scaled by Metric3D) **and**
per-frame MANO hand poses from a single egocentric video. That is exactly the two
things the ``aria`` stage produces, so HaWoR can stand in for it wholesale.

The downstream HumanEgo stages are file-driven: they read ``rgb.png`` plus the
per-frame ``aria_cam_rgb.json`` (camera ``c2w`` + ``k``) and ``aria_hands.json``
(``midpoint_*_opt_world`` / ``confidence`` / ``grasp_state``) from each
``all_data/<idx>/`` folder, and the top-level ``aria_phases_results.json``. This
adapter writes precisely those files.

HaWoR input layout (one session)::

    <session>/
        <name>.MP4                         # source video (50 fps native here)
        <name>.json                        # caption segments (optional, for phases)
        <name>/
            SLAM/hawor_slam_w_scale_0_<N>.npz   # traj (N,7) + scale + img_focal/center
            cam_space/0/0_<N-1>.json            # LEFT  hand MANO params (camera frame)
            cam_space/1/0_<N-1>.json            # RIGHT hand MANO params (camera frame)

HumanEgo output layout (one session)::

    <out>/
        preprocess/
            aria_cam_rgb_config.json
            aria_phases_results.json
            all_data/<idx:05d>/
                rgb.png
                aria_cam_rgb.json
                aria_hands.json            # primary -> training_data "hands"
                hawor_hands.json           # provenance copy -> "hands_hawor"

Coordinate conventions / approximations (read before trusting the numbers)
-------------------------------------------------------------------------
* Camera poses are decoded exactly like HaWoR's own ``load_slam_cam``: the stored
  quaternion order is ``[x, y, z, w]`` and translation is multiplied by ``scale``.
  We do NOT apply the ``R_x = diag(1,-1,-1)`` flip from ``demo.py`` -- that flip is
  only for aitviewer rendering. Camera and hands are lifted by the *same* raw
  ``c2w``, so they live in one consistent HaWoR world frame.
* The hand "frame" we export is the **MANO root (wrist) pose**: translation =
  ``init_trans``, rotation = ``init_root_orient``. Computing HumanEgo's true
  thumb/index "midpoint" frame would require running the MANO layer (model files +
  torch). We instead populate ``midpoint_*`` and ``wrist_*`` identically with the
  wrist pose -- a consistent control point that DatasetGen consumes unchanged.
* ``grasp_state`` is a proxy from finger-joint flexion (mean geodesic angle of the
  15 ``init_hand_pose`` joint rotations), thresholded. ``confidence`` is 1.0
  (cam_space carries no per-frame validity).
* Velocities are written as zeros; recompute downstream if training needs them.

These approximations are intentional and isolated in clearly named helpers so they
can be upgraded later (e.g. by running MANO in the hawor env to get true midpoints).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # frame extraction needs cv2; the math path does not
    cv2 = None


# Older HaWoR runs resampled with ffmpeg ``fps=30``. detect_track_video now
# keeps every source frame, so pass the source fps when ingesting those runs.
HAWOR_EXTRACT_FPS = 30.0

# cam_space subdir -> hand side (matches demo.py hand2idx = {"right":1,"left":0})
CAM_SPACE_SIDE = {"0": "left", "1": "right"}


# --------------------------------------------------------------------------- #
# Geometry helpers (pure numpy; no torch / MANO dependency)
# --------------------------------------------------------------------------- #
def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` -> 3x3 rotation matrix.

    Mirrors HaWoR's ``load_slam_cam`` which feeds ``q[[3,0,1,2]]`` (i.e. w,x,y,z)
    into pytorch3d's ``quaternion_to_matrix``; this is the same rotation.
    """
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rotation_geodesic_angle(R: np.ndarray) -> float:
    """Geodesic angle (radians) of a rotation matrix: arccos((tr(R)-1)/2)."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


@dataclass
class CameraTrack:
    c2w: np.ndarray          # (N, 4, 4) camera-to-world, metric-scaled
    K: np.ndarray            # (3, 3) intrinsics
    width: int
    height: int
    n_frames: int


def decode_slam_cameras(npz_path: str) -> CameraTrack:
    """Decode HaWoR SLAM npz into per-frame metric ``c2w`` + intrinsics."""
    d = dict(np.load(npz_path, allow_pickle=True))
    traj = d["traj"]                       # (N, 7): [tx,ty,tz, qx,qy,qz,qw]
    scale = float(d["scale"])
    focal = float(d["img_focal"])
    cx, cy = [float(v) for v in d["img_center"]]
    n = traj.shape[0]

    c2w = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    for i in range(n):
        t = traj[i, :3] * scale
        R = quat_xyzw_to_R(traj[i, 3:])
        c2w[i, :3, :3] = R
        c2w[i, :3, 3] = t

    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    # img_center is the principal point = (W/2, H/2) for HaWoR's centered model.
    width, height = int(round(2 * cx)), int(round(2 * cy))
    return CameraTrack(c2w=c2w, K=K, width=width, height=height, n_frames=n)


@dataclass
class HandTrack:
    trans_cam: np.ndarray    # (N, 3) wrist/root translation in camera frame
    R_cam: np.ndarray        # (N, 3, 3) wrist/root orientation in camera frame
    grasp: np.ndarray        # (N,) binary grasp proxy
    valid: np.ndarray        # (N,) bool


def decode_cam_space_hand(json_path: str, grasp_rel: float = 0.5,
                          grasp_abs: Optional[float] = None) -> HandTrack:
    """Decode one cam_space hand JSON (camera-frame MANO params).

    Grasp proxy: per-frame mean geodesic angle of the 15 ``init_hand_pose`` joint
    rotations (finger flexion). Because MANO's pose is relative to a partly-curled
    template, the *absolute* flexion is a weak discriminator (a relaxed hand already
    sits near ~0.8 rad), so by default we threshold **relative to this sequence's own
    range**: ``thr = p10 + grasp_rel * (p90 - p10)``. Pass ``grasp_abs`` to force a
    fixed absolute threshold instead. For a principled grasp, run MANO in the hawor
    env and use thumb-index tip distance (cf. AriaHands GRASP_THRESHOLD = 0.105 m).
    """
    d = json.load(open(json_path))
    trans = np.asarray(d["init_trans"], dtype=np.float64)[0]        # (N,3)
    root = np.asarray(d["init_root_orient"], dtype=np.float64)[0]   # (N,3,3)
    pose = np.asarray(d["init_hand_pose"], dtype=np.float64)[0]     # (N,15,3,3)
    n = trans.shape[0]

    flex = np.array([
        float(np.mean([rotation_geodesic_angle(pose[i, j]) for j in range(pose.shape[1])]))
        for i in range(n)
    ])
    if grasp_abs is not None:
        thr = grasp_abs
    else:
        lo, hi = np.percentile(flex, 10), np.percentile(flex, 90)
        thr = lo + grasp_rel * (hi - lo)
    grasp = (flex > thr).astype(np.float64)

    # A frame is "valid" unless the pose is degenerate (all-zero translation/rot).
    valid = ~(np.all(trans == 0, axis=1) & np.all(root.reshape(n, -1) == 0, axis=1))
    return HandTrack(trans_cam=trans, R_cam=root, grasp=grasp, valid=valid)


# --------------------------------------------------------------------------- #
# MANO-based decoder (optional): true thumb/index-tip "midpoint" control point.
# The pure-numpy decoder above exports the MANO *root (wrist)*, which sits ~13-20 cm
# from the actual grasp/contact point -- too far for DatasetGen's 0.20 m grasp latch,
# so objects never become dynamic. Running the MANO layer recovers the 21 hand joints
# (incl. fingertips); we use the thumb-index-tip midpoint as the control point and the
# tip distance as a principled grasp signal (cf. AriaHands GRASP_THRESHOLD = 0.105 m).
# --------------------------------------------------------------------------- #
# OpenPose-21 hand joint indices (smplx MANO + HaMeR joint_map):
#   0 wrist | 1-4 thumb(CMC,MCP,IP,TIP) | 5-8 index(MCP,PIP,DIP,TIP) | ...
J_WRIST, J_THUMB_MCP, J_THUMB_TIP, J_INDEX_MCP, J_INDEX_TIP = 0, 2, 4, 5, 8


def load_mano_layer(hawor_root: str, device: str = "cuda"):
    """Build HaWoR's right-hand MANO layer. Needs torch + _DATA/data/mano/MANO_RIGHT.pkl."""
    import sys
    import torch
    if hawor_root not in sys.path:
        sys.path.insert(0, hawor_root)
    from lib.models.mano_wrapper import MANO
    cfg = {"data_dir": os.path.join(hawor_root, "_DATA/data/"),
           "model_path": os.path.join(hawor_root, "_DATA/data/mano"),
           "gender": "neutral", "num_hand_joints": 15, "create_body_pose": False}
    mano = MANO(**cfg).to(device).eval()
    return mano, torch, device


def _orthonormal_frame(x, y):
    """Right-handed frame with X≈x, Y≈y orthogonalized, Z=X×Y. (N,3) inputs."""
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    y = y - (np.sum(y * x, axis=1, keepdims=True)) * x
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)            # (N,3,3) columns = axes


def decode_cam_space_hand_mano(json_path, mano_bundle, side, grasp_dist_m=0.105):
    """Decode cam_space via the MANO layer -> thumb/index-tip midpoint control point.

    side: 'right' uses the right MANO model directly; 'left' uses the SAME right model
    (no mirror) -- its transl places the hand at the left hand's true camera location, so
    the tip midpoint is correct even though the mesh chirality is a right hand. (Exact left
    geometry would need MANO_LEFT.pkl; for a control point + grasp it is unnecessary.)
    """
    mano, torch, device = mano_bundle
    d = json.load(open(json_path))
    go = np.asarray(d["init_root_orient"], dtype=np.float32)[0]     # (N,3,3)
    hp = np.asarray(d["init_hand_pose"], dtype=np.float32)[0]       # (N,15,3,3)
    be = np.asarray(d["init_betas"], dtype=np.float32)[0]           # (N,10)
    tr = np.asarray(d["init_trans"], dtype=np.float32)[0]           # (N,3)
    n = tr.shape[0]

    with torch.no_grad():
        out = mano(global_orient=torch.from_numpy(go).view(n, 1, 3, 3).to(device),
                   hand_pose=torch.from_numpy(hp).view(n, 15, 3, 3).to(device),
                   betas=torch.from_numpy(be).view(n, 10).to(device),
                   transl=torch.from_numpy(tr).view(n, 3).to(device),
                   pose2rot=False)
        J = out.joints.detach().cpu().numpy()                       # (N,21,3)
    # NOTE: left uses the right model directly. Chirality is wrong (right-hand mesh) but
    # transl places it at the LEFT hand's true location, so the thumb/index-tip midpoint is
    # right. Do NOT mirror x -- that flips the hand across the optical axis to the wrong side.

    thumb_tip, index_tip = J[:, J_THUMB_TIP], J[:, J_INDEX_TIP]
    midpoint = 0.5 * (thumb_tip + index_tip)
    R = _orthonormal_frame(J[:, J_INDEX_MCP] - J[:, J_THUMB_MCP],   # X: across the pinch
                           midpoint - J[:, J_WRIST])                # Y: palm-forward
    grasp = (np.linalg.norm(thumb_tip - index_tip, axis=1) < grasp_dist_m).astype(np.float64)
    valid = ~(np.all(tr == 0, axis=1) & np.all(go.reshape(n, -1) == 0, axis=1))
    return HandTrack(trans_cam=midpoint.astype(np.float64),
                     R_cam=R.astype(np.float64), grasp=grasp, valid=valid)


def lift_pose_to_world(c2w: np.ndarray, R_cam: np.ndarray, t_cam: np.ndarray):
    """Lift a camera-frame pose (R_cam, t_cam) into world via c2w. Returns (R_w, t_w)."""
    Rcw = c2w[:3, :3]
    tcw = c2w[:3, 3]
    R_w = Rcw @ R_cam
    t_w = Rcw @ t_cam + tcw
    return R_w, t_w


# --------------------------------------------------------------------------- #
# HumanEgo per-frame JSON builders (schema matches AriaCamTypes / AriaHandsTypes)
# --------------------------------------------------------------------------- #
def _T(R: np.ndarray, t: np.ndarray) -> list:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T.tolist()


def build_cam_json(idx: int, ts: int, cam: CameraTrack, fps: float) -> dict:
    return {
        "idx": idx,
        "ts": ts,
        "fov": 0.0,
        "h": cam.height,
        "w": cam.width,
        "k": cam.K.tolist(),
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "c2w": cam.c2w[idx].tolist(),
        "c2d": np.eye(4).tolist(),
        "d2w": cam.c2w[idx].tolist(),
        "rgb_path": os.path.join("preprocess", "all_data", f"{idx:05d}", "rgb.png"),
        "fps": fps,
    }


def _pack_hand(R_w: np.ndarray, t_w: np.ndarray, grasp: float, conf: float) -> dict:
    """Build one hand-side dict in the full AriaHands schema.

    The wrist pose doubles as the 'midpoint' control frame (see module docstring).
    DatasetGen reads only midpoint_translation_opt_world / midpoint_orientation_opt_world
    (flat 9, row-major) / confidence / grasp_state; the rest are filled for
    completeness so existing consumers/visualizers don't trip on missing keys.
    """
    pose = _T(R_w, t_w)
    orient_flat = R_w.flatten().tolist()
    pos = t_w.tolist()
    z3 = [0.0, 0.0, 0.0]
    return {
        "d2c": None,
        "c2w": None,
        "confidence": float(conf),
        "grasp_state": int(grasp),
        "wrist_pose": pose,
        "palm_pose": pose,
        "kpts_3d": None,
        "kpts_2d": None,
        "joint_angles": {},
        "wrist_pose_raw_world": pose,
        "wrist_pose_opt_world": pose,
        "wrist_lin_vel_raw_world": z3,
        "wrist_ang_vel_raw_world": z3,
        "wrist_lin_vel_opt_world": z3,
        "wrist_ang_vel_opt_world": z3,
        "index_translation_raw_world": None,
        "index_translation_opt_world": None,
        "thumb_translation_raw_world": None,
        "thumb_translation_opt_world": None,
        "midpoint_pose_raw_world": pose,
        "midpoint_pose_opt_world": pose,
        "midpoint_translation_raw_world": pos,
        "midpoint_orientation_raw_world": orient_flat,
        "midpoint_translation_opt_world": pos,
        "midpoint_orientation_opt_world": orient_flat,
        "midpoint_lin_vel_raw_world": z3,
        "midpoint_ang_vel_raw_world": z3,
        "midpoint_lin_vel_opt_world": z3,
        "midpoint_ang_vel_opt_world": z3,
        "distance_midpoint2wrist_raw_world": 0.0,
        "distance_midpoint2wrist_opt_world": 0.0,
    }


def build_hands_json(idx: int, ts: int, cam: CameraTrack,
                     hands: dict, conf: float) -> dict:
    """hands: {side -> HandTrack}. Returns per-frame aria_hands.json dict."""
    out = {"idx": idx, "ts": ts, "hand_r": None, "hand_l": None}
    for side, track in hands.items():
        if track is None or not track.valid[idx]:
            continue
        R_w, t_w = lift_pose_to_world(cam.c2w[idx], track.R_cam[idx], track.trans_cam[idx])
        out["hand_r" if side == "right" else "hand_l"] = _pack_hand(
            R_w, t_w, track.grasp[idx], conf)
    return out


# --------------------------------------------------------------------------- #
# Phases (no Aria phase segmentation -> whole clip is one manipulation window)
# --------------------------------------------------------------------------- #
def build_phases(n_frames: int) -> dict:
    """Minimal aria_phases_results.json: entire clip = manipulation (mode '0').

    preprocess_indices reads stage_window_check.windows and treats keys '0','4'
    as manipulation, '3' as transition, '1','2' as navigation.
    """
    return {
        "stage_window_check": {
            "windows": {
                "0": [[0, n_frames - 1]],
                "1": [], "2": [], "3": [], "4": [],
            }
        },
        "summary": {"manip_frames": n_frames, "source": "hawor_ingest_full_clip"},
    }


# --------------------------------------------------------------------------- #
# Session orchestration
# --------------------------------------------------------------------------- #
def _find_session_files(session_dir: str):
    """Locate video / SLAM npz / cam_space dir inside a HaWoR session folder."""
    name = os.path.basename(os.path.normpath(session_dir))
    inner = os.path.join(session_dir, name)
    if not os.path.isdir(inner):
        # Some layouts keep SLAM/cam_space directly under session_dir.
        inner = session_dir
    npz = sorted(glob.glob(os.path.join(inner, "SLAM", "hawor_slam_w_scale_*.npz")))
    cam_space = os.path.join(inner, "cam_space")
    video = None
    for ext in (".MP4", ".mp4", ".MOV", ".mov"):
        cand = os.path.join(session_dir, name + ext)
        if os.path.exists(cand):
            video = cand
            break
    if video is None:
        vids = [f for f in glob.glob(os.path.join(session_dir, "*"))
                if f.lower().endswith((".mp4", ".mov"))]
        video = vids[0] if vids else None
    return {
        "name": name,
        "video": video,
        "npz": npz[0] if npz else None,
        "cam_space": cam_space if os.path.isdir(cam_space) else None,
    }


def extract_frames(video: str, out_dir: str, indices, native_fps: float,
                   target_fps: float = HAWOR_EXTRACT_FPS) -> int:
    """Extract one rgb.png per HaWoR frame index, reproducing ffmpeg ``fps=30``.

    HaWoR index i corresponds to native frame round(i * native_fps / target_fps).
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required to extract frames; run in an env with opencv.")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    written = 0
    ratio = native_fps / target_fps
    for i in indices:
        native_idx = int(round(i * ratio))
        cap.set(cv2.CAP_PROP_POS_FRAMES, native_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_dir = os.path.join(out_dir, f"{i:05d}")
        os.makedirs(frame_dir, exist_ok=True)
        cv2.imwrite(os.path.join(frame_dir, "rgb.png"), frame)
        written += 1
    cap.release()
    return written


def ingest_session(session_dir: str, out_dir: str, *,
                   fps: float = HAWOR_EXTRACT_FPS,
                   grasp_rel: float = 0.5,
                   grasp_abs: Optional[float] = None,
                   confidence: float = 1.0,
                   max_frames: Optional[int] = None,
                   write_frames: bool = True,
                   native_fps: Optional[float] = None,
                   finish_tail: int = 1,
                   use_mano: bool = False,
                   hawor_root: Optional[str] = None,
                   mano_device: str = "cuda",
                   grasp_dist_m: float = 0.105) -> dict:
    """Convert one HaWoR session into a HumanEgo session at ``out_dir``."""
    files = _find_session_files(session_dir)
    if not files["npz"]:
        raise FileNotFoundError(f"no HaWoR SLAM npz under {session_dir}")
    if not files["cam_space"]:
        raise FileNotFoundError(f"no cam_space dir under {session_dir}")

    cam = decode_slam_cameras(files["npz"])
    n = cam.n_frames if max_frames is None else min(cam.n_frames, max_frames)

    # Decode whichever hand sides HaWoR produced. With --use_mano, recover the true
    # thumb/index-tip midpoint via the MANO layer; otherwise use the numpy wrist proxy.
    mano_bundle = None
    if use_mano:
        root = hawor_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        mano_bundle = load_mano_layer(root, mano_device)
    hands = {}
    for sub, side in CAM_SPACE_SIDE.items():
        cand = sorted(glob.glob(os.path.join(files["cam_space"], sub, "*.json")))
        if not cand:
            hands[side] = None
        elif mano_bundle is not None:
            hands[side] = decode_cam_space_hand_mano(cand[0], mano_bundle, side, grasp_dist_m)
        else:
            hands[side] = decode_cam_space_hand(cand[0], grasp_rel, grasp_abs)

    pre_dir = os.path.join(out_dir, "preprocess")
    all_data = os.path.join(pre_dir, "all_data")
    os.makedirs(all_data, exist_ok=True)

    indices = list(range(n))
    for i in indices:
        ts = int(round(i * 1e9 / fps))
        frame_dir = os.path.join(all_data, f"{i:05d}")
        os.makedirs(frame_dir, exist_ok=True)
        json.dump(build_cam_json(i, ts, cam, fps),
                  open(os.path.join(frame_dir, "aria_cam_rgb.json"), "w"), indent=4)
        hj = build_hands_json(i, ts, cam, hands, confidence)
        json.dump(hj, open(os.path.join(frame_dir, "aria_hands.json"), "w"), indent=4)
        json.dump(hj, open(os.path.join(frame_dir, "hawor_hands.json"), "w"), indent=4)
        # DatasetGen requires these two per-frame files to exist, else it discards the
        # frame. aria_phases.json: mode 4 => is_finished (mark the demonstration tail);
        # aria_slam.json: per-frame device pose (= camera c2w for an egocentric head cam).
        mode = 4 if i >= n - max(1, finish_tail) else 0
        json.dump({"idx": i, "ts": ts, "mode": mode},
                  open(os.path.join(frame_dir, "aria_phases.json"), "w"), indent=4)
        json.dump({"idx": i, "ts": ts, "c2w": cam.c2w[i].tolist(), "d2w": cam.c2w[i].tolist()},
                  open(os.path.join(frame_dir, "aria_slam.json"), "w"), indent=4)

    # Top-level cam config summary (AriaCam._save_aria_cam_config_json schema).
    json.dump({
        "total_frames": n, "fps": fps, "first_ts": 0,
        "h": cam.height, "w": cam.width,
        "k": cam.K.tolist(), "d": [0.0] * 5, "c2d": np.eye(4).tolist(),
    }, open(os.path.join(pre_dir, "aria_cam_rgb_config.json"), "w"), indent=4)

    json.dump(build_phases(n),
              open(os.path.join(pre_dir, "aria_phases_results.json"), "w"), indent=2)

    frames_written = 0
    if write_frames and files["video"]:
        nf = native_fps
        if nf is None and cv2 is not None:
            cap = cv2.VideoCapture(files["video"])
            nf = cap.get(cv2.CAP_PROP_FPS) or 50.0
            cap.release()
        frames_written = extract_frames(files["video"], all_data, indices, nf or 50.0, fps)

    return {
        "session": files["name"], "out_dir": out_dir,
        "n_frames": n, "frames_written": frames_written,
        "hands": {s: (h is not None) for s, h in hands.items()},
        "width": cam.width, "height": cam.height,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HaWoR -> HumanEgo ingest adapter")
    p.add_argument("--session_dir", required=True,
                   help="HaWoR session folder (contains <name>.MP4 and <name>/SLAM,cam_space)")
    p.add_argument("--out_dir", required=True,
                   help="Output HumanEgo session dir (its preprocess/ will be populated)")
    p.add_argument("--fps", type=float, default=HAWOR_EXTRACT_FPS)
    p.add_argument("--grasp_rel", type=float, default=0.5,
                   help="Sequence-relative grasp threshold in [0,1]: thr = p10 + rel*(p90-p10)")
    p.add_argument("--grasp_abs", type=float, default=None,
                   help="Absolute mean-flexion (rad) grasp threshold; overrides --grasp_rel")
    p.add_argument("--confidence", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--no_frames", action="store_true",
                   help="Skip rgb.png extraction (write only camera/hand JSON)")
    p.add_argument("--native_fps", type=float, default=None,
                   help="Override source video fps (else auto-detected)")
    p.add_argument("--finish_tail", type=int, default=1,
                   help="Mark the last N frames as the demonstration end (is_finished=1)")
    p.add_argument("--use_mano", action="store_true",
                   help="Run the MANO layer to export the true thumb/index-tip midpoint "
                        "control point (else the numpy wrist proxy). Needs torch + "
                        "_DATA/data/mano/MANO_RIGHT.pkl.")
    p.add_argument("--hawor_root", default=None, help="HaWoR repo root (for MANO import)")
    p.add_argument("--mano_device", default="cuda")
    p.add_argument("--grasp_dist_m", type=float, default=0.105,
                   help="thumb-index tip distance below which grasp=1 (MANO mode)")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    stats = ingest_session(
        args.session_dir, args.out_dir,
        fps=args.fps, grasp_rel=args.grasp_rel, grasp_abs=args.grasp_abs,
        confidence=args.confidence, max_frames=args.max_frames,
        write_frames=not args.no_frames, native_fps=args.native_fps,
        finish_tail=args.finish_tail, use_mano=args.use_mano,
        hawor_root=args.hawor_root, mano_device=args.mano_device,
        grasp_dist_m=args.grasp_dist_m,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
