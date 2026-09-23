#!/usr/bin/env python3
"""Write HaWoR camera and hand tracks as an Open-AoE ``hands.npz``.

HaWoR stores hands as camera-frame MANO chunks under ``cam_space/{0,1}`` and the
camera as a DROID trajectory under ``SLAM/hawor_slam_w_scale_*.npz``. The viewer
expects one dense archive:

- hand 0 = left, hand 1 = right
- ``pred_rot`` / ``pred_trans``: world-frame root, axis-angle and translation
- ``pred_rot_cam`` / ``pred_trans_cam``: the same root in the camera frame
- ``pred_hand_pose``: 15 joints as a 45-d axis-angle
- ``pred_betas``, ``pred_valid``
- ``R_c2w`` / ``t_c2w`` / ``R_w2c`` / ``t_w2c``, ``focal``

Frames with no cam_space chunk stay invalid. The 30 fps track is used when both
30 fps and 50 fps outputs exist.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CHUNK_NAME = re.compile(r"^(\d+)_(\d+)\.json$")


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion ``[..., 4]`` in xyzw order to rotation matrices ``[..., 3, 3]``."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n = np.maximum(n, 1e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[..., 3, 3]`` to axis-angle ``[..., 3]``."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    skew = np.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        axis=-1,
    )
    sin_angle = np.sin(angle)
    stable = np.abs(sin_angle) >= 1e-4
    denom = np.where(stable, 2.0 * sin_angle, 1.0)
    axis_angle = skew / denom[..., None] * angle[..., None]
    axis_angle = np.where((angle < 1e-6)[..., None], 0.0, axis_angle)

    near_pi = angle > np.pi - 1e-3
    if np.any(near_pi):
        diag = np.diagonal(rotation, axis1=-2, axis2=-1)
        axis = np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, None))
        axis = axis * np.sign(skew + 1e-12)
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / np.maximum(norm, 1e-12)
        pi_angle = axis * angle[..., None]
        axis_angle = np.where(near_pi[..., None], pi_angle, axis_angle)
    return axis_angle.astype(np.float32)


def pick_slam_npz(session_dir: Path) -> Path:
    slam_dir = session_dir / "SLAM"
    candidates = []
    for path in sorted(slam_dir.glob("*.npz")):
        if "disps" in path.name:
            continue
        with np.load(path, allow_pickle=True) as archive:
            if "traj" not in archive.files:
                continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no SLAM trajectory under {slam_dir}")
    primary = [path for path in candidates if "50fps" not in path.name]
    return primary[0] if primary else candidates[0]


def load_camera(slam_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    archive = np.load(slam_path, allow_pickle=True)
    traj = np.asarray(archive["traj"], dtype=np.float64)
    scale = float(archive["scale"])
    focal = float(archive["img_focal"])
    rotation = quat_xyzw_to_matrix(traj[:, 3:])
    translation = traj[:, :3] * scale
    world_to_cam = np.swapaxes(rotation, -1, -2)
    t_w2c = -np.einsum("nij,nj->ni", world_to_cam, translation)
    return rotation, translation, world_to_cam, t_w2c, focal


def stitch_hand(cam_space: Path, hand_index: int, n_frames: int) -> dict[str, np.ndarray]:
    root = np.zeros((n_frames, 3, 3), dtype=np.float64)
    pose = np.zeros((n_frames, 15, 3, 3), dtype=np.float64)
    trans = np.zeros((n_frames, 3), dtype=np.float64)
    betas = np.zeros((n_frames, 10), dtype=np.float64)
    valid = np.zeros((n_frames,), dtype=np.float32)
    hand_dir = cam_space / str(hand_index)
    if not hand_dir.is_dir():
        return {"root": root, "pose": pose, "trans": trans, "betas": betas, "valid": valid}

    chunks = []
    for path in hand_dir.glob("*.json"):
        match = CHUNK_NAME.match(path.name)
        if match is None:
            continue
        chunks.append((int(match.group(1)), int(match.group(2)), path))
    for start, end, path in sorted(chunks):
        payload = json.loads(path.read_text())
        stop = min(end + 1, n_frames)
        count = stop - start
        if count <= 0:
            continue
        root[start:stop] = np.asarray(payload["init_root_orient"], dtype=np.float64)[0, :count]
        pose[start:stop] = np.asarray(payload["init_hand_pose"], dtype=np.float64)[0, :count]
        trans[start:stop] = np.asarray(payload["init_trans"], dtype=np.float64)[0, :count]
        betas[start:stop] = np.asarray(payload["init_betas"], dtype=np.float64)[0, :count]
        valid[start:stop] = 1.0
    return {"root": root, "pose": pose, "trans": trans, "betas": betas, "valid": valid}


def export_session(session_dir: Path) -> Path:
    """Convert one HaWoR output directory into ``hands.npz``."""
    session_dir = Path(session_dir)
    slam_path = pick_slam_npz(session_dir)
    r_c2w, t_c2w, r_w2c, t_w2c, focal = load_camera(slam_path)
    n_frames = r_c2w.shape[0]
    cam_space = session_dir / "cam_space"

    pred_rot = np.zeros((2, n_frames, 3), dtype=np.float32)
    pred_trans = np.zeros((2, n_frames, 3), dtype=np.float32)
    pred_rot_cam = np.zeros((2, n_frames, 3), dtype=np.float32)
    pred_trans_cam = np.zeros((2, n_frames, 3), dtype=np.float32)
    pred_hand_pose = np.zeros((2, n_frames, 45), dtype=np.float32)
    pred_betas = np.zeros((2, n_frames, 10), dtype=np.float32)
    pred_valid = np.zeros((2, n_frames), dtype=np.float32)

    for hand_index in (0, 1):
        hand = stitch_hand(cam_space, hand_index, n_frames)
        pred_valid[hand_index] = hand["valid"]
        pred_trans_cam[hand_index] = hand["trans"].astype(np.float32)
        pred_rot_cam[hand_index] = matrix_to_axis_angle(hand["root"])
        pred_hand_pose[hand_index] = matrix_to_axis_angle(hand["pose"]).reshape(n_frames, 45)
        pred_betas[hand_index] = hand["betas"].astype(np.float32)
        covered = hand["valid"] > 0.5
        world_root = np.einsum("nij,njk->nik", r_c2w, hand["root"])
        world_trans = np.einsum("nij,nj->ni", r_c2w, hand["trans"]) + t_c2w
        pred_rot[hand_index, covered] = matrix_to_axis_angle(world_root)[covered]
        pred_trans[hand_index, covered] = world_trans[covered].astype(np.float32)

    out = session_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        R_w2c=r_w2c.astype(np.float32),
        t_w2c=t_w2c.astype(np.float32),
        R_c2w=r_c2w.astype(np.float32),
        t_c2w=t_c2w.astype(np.float32),
        pred_trans=pred_trans,
        pred_rot=pred_rot,
        pred_trans_cam=pred_trans_cam,
        pred_rot_cam=pred_rot_cam,
        pred_hand_pose=pred_hand_pose,
        pred_betas=pred_betas,
        pred_valid=pred_valid,
        focal=np.float64(focal),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HaWoR tracks to Open-AoE hands.npz")
    parser.add_argument("--session", required=True, help="HaWoR output directory with cam_space/ and SLAM/")
    args = parser.parse_args()
    path = export_session(Path(args.session))
    archive = np.load(path)
    valid = archive["pred_valid"]
    print(
        f"wrote {path} frames={archive['pred_rot'].shape[1]} "
        f"valid_left={float(valid[0].mean()):.3f} valid_right={float(valid[1].mean()):.3f} "
        f"focal={float(archive['focal']):.2f}"
    )


if __name__ == "__main__":
    main()
