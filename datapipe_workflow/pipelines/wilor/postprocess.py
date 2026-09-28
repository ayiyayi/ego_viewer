"""Mask unstable wrists, then fill only short gaps that are still the same hand.

The viewer files keep the same schema. Camera arrays are copied through.

Thresholds are speeds and durations, matched to the 30 fps FineBio settings:

- 0.08 m per frame is 2.4 m/s of camera motion
- 90 frames is 3.0 s to look for the same hand coming back
- 5 frames is 1/6 s, the longest hole that may be filled

A hole is filled only when it originally had no detection, the camera stayed
under 2.4 m/s, both wrists stay within 0.10 m, the filled step stays at or
below 1.5 m/s, and the finger pose agrees within 0.03 m. A frame that was
detected and then masked is left empty.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

WRIST_SPEED_M_S = 1.5
CAMERA_SPEED_M_S = 2.4
RETURN_S = 3.0
MAX_GAP_S = 5.0 / 30.0
END_DIST_M = 0.10
POSE_M = 0.03


def _allow(dt_frames: int, fps: float) -> float:
    return WRIST_SPEED_M_S * dt_frames / fps


def _clean(wrist: np.ndarray, orig: np.ndarray, start: int, end: int, fps: float) -> np.ndarray:
    n = len(wrist)
    accepted = np.zeros(n, dtype=bool)
    idx = [t for t in range(start, end) if orig[t]]
    if not idx:
        return accepted
    last = idx[0]
    accepted[last] = True
    k = 1
    while k < len(idx):
        t = idx[k]
        if np.linalg.norm(wrist[t] - wrist[last]) <= _allow(t - last, fps):
            accepted[t] = True
            last = t
            k += 1
            continue
        found = None
        for j in range(k + 1, len(idx)):
            u = idx[j]
            if (u - last) / fps > RETURN_S:
                break
            if np.linalg.norm(wrist[u] - wrist[last]) <= _allow(u - last, fps):
                found = j
                break
        if found is not None:
            k = found
            continue
        nxt = idx[k + 1] if k + 1 < len(idx) else None
        if nxt is not None and (nxt - t) / fps <= RETURN_S and np.linalg.norm(wrist[nxt] - wrist[t]) <= _allow(nxt - t, fps):
            last = t
            k += 1
            continue
        if (t - last) / fps > RETURN_S:
            accepted[t] = True
            last = t
        k += 1
    return accepted


def stabilize(joints: np.ndarray, valid: np.ndarray, verts: np.ndarray, cam_pos: np.ndarray, fps: float):
    """Return filtered joints, validity, verts, and a per-hand count report."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    joints = np.array(joints, dtype=np.float32, copy=True)
    verts = np.array(verts, dtype=np.float32, copy=True)
    original = np.asarray(valid) > 0.5
    n = joints.shape[1]
    if cam_pos.shape[0] != n or verts.shape[1] != n or original.shape != (2, n):
        raise ValueError("joints, validity, vertices, and camera length must match")
    cam_speed = np.zeros(n, dtype=np.float32)
    cam_speed[1:] = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1) * fps
    breaks = cam_speed > CAMERA_SPEED_M_S
    starts = [0] + [int(i) for i in np.flatnonzero(breaks)]
    out_valid = np.zeros_like(valid, dtype=np.float32)
    report = {"camera_breaks": int(breaks.sum()), "fps": float(fps)}
    names = ("left", "right")
    for hand, name in enumerate(names):
        wrist = joints[hand, :, 0]
        orig = original[hand]
        accepted = np.zeros(n, dtype=bool)
        for i, start in enumerate(starts):
            stop = starts[i + 1] if i + 1 < len(starts) else n
            accepted |= _clean(wrist, orig, start, stop, fps)
        filled = np.zeros(n, dtype=bool)
        anchors = np.flatnonzero(accepted)
        for a0, b0 in zip(anchors[:-1], anchors[1:]):
            gap = int(b0 - a0 - 1)
            if gap <= 0 or gap / fps > MAX_GAP_S:
                continue
            if np.any(breaks[a0 + 1:b0 + 1]):
                continue
            dist = float(np.linalg.norm(wrist[b0] - wrist[a0]))
            duration = (b0 - a0) / fps
            if dist > END_DIST_M or dist / duration > WRIST_SPEED_M_S:
                continue
            pose = float(np.linalg.norm(
                (joints[hand, b0] - wrist[b0]) - (joints[hand, a0] - wrist[a0]), axis=1
            ).mean())
            if pose > POSE_M:
                continue
            span = b0 - a0
            for t in range(a0 + 1, b0):
                if orig[t]:
                    continue
                alpha = (t - a0) / span
                joints[hand, t] = (1 - alpha) * joints[hand, a0] + alpha * joints[hand, b0]
                verts[hand, t] = (1 - alpha) * verts[hand, a0] + alpha * verts[hand, b0]
                filled[t] = True
        keep = accepted | filled
        joints[hand, ~keep] = 0
        verts[hand, ~keep] = 0
        out_valid[hand, keep] = 1
        show = np.flatnonzero(keep)
        speeds = [
            float(np.linalg.norm(joints[hand, y, 0] - joints[hand, x, 0])) * fps
            for x, y in zip(show[:-1], show[1:])
            if y == x + 1 and not breaks[y]
        ]
        fastest = max(speeds, default=0.0)
        if fastest > WRIST_SPEED_M_S + 1e-3:
            raise RuntimeError(f"{name} adjacent speed {fastest:.3f} m/s exceeds {WRIST_SPEED_M_S}")
        report[name] = {
            "original": int(orig.sum()),
            "kept": int(accepted.sum()),
            "masked": int((orig & ~accepted).sum()),
            "interpolated": int(filled.sum()),
            "final": int(keep.sum()),
            "adjacent_speed_max_m_s": fastest,
        }
    return joints, out_valid, verts, report


def stabilize_viewer_dir(output_dir: Path) -> dict:
    """Rewrite keypoints.npz and mesh.bin in a viewer output directory."""
    output_dir = Path(output_dir)
    keypoints_path = output_dir / "keypoints.npz"
    mesh_path = output_dir / "mesh.bin"
    with np.load(keypoints_path) as src:
        verts = _read_verts(mesh_path, src["joints_world"].shape[1])
        joints, valid, verts, report = stabilize(
            src["joints_world"], src["pred_valid"], verts, src["cam_pos"], float(src["fps"])
        )
        payload = {key: np.array(src[key]) for key in src.files if key not in ("joints_world", "pred_valid")}
    payload["joints_world"] = joints
    payload["pred_valid"] = valid
    camera = payload["cam_pos"].copy()
    focal = payload["focal"].copy()
    np.savez_compressed(keypoints_path, **payload)
    _write_mesh(mesh_path, verts)
    saved = np.load(keypoints_path)
    if not np.allclose(saved["cam_pos"], camera) or not np.allclose(saved["focal"], focal):
        raise RuntimeError("postprocess changed the camera or focal length")
    return report


def _read_verts(path: Path, n_frames: int) -> np.ndarray:
    with path.open("rb") as handle:
        header = struct.unpack("<III", handle.read(12))
        frames, n_verts, n_faces = header
        if frames != n_frames:
            raise ValueError(f"mesh has {frames} frames, keypoints have {n_frames}")
        handle.seek(12 + n_faces * 3 * 4)
        return np.frombuffer(handle.read(), dtype=np.float16).reshape(2, frames, n_verts, 3).astype(np.float32)


def _write_mesh(path: Path, verts: np.ndarray) -> None:
    with path.open("rb") as handle:
        n_frames, n_verts, n_faces = struct.unpack("<III", handle.read(12))
        faces = handle.read(n_faces * 3 * 4)
    if verts.shape[:3] != (2, n_frames, n_verts):
        raise ValueError("filtered vertices do not match the mesh header")
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", n_frames, n_verts, n_faces))
        handle.write(faces)
        handle.write(np.ascontiguousarray(verts.astype(np.float16)).tobytes())
