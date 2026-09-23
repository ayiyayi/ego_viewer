"""Export Open-AoE hands.npz into a compact keypoint archive for the web viewer.

World joints follow the delivery convention: MANO FK with world-frame
pred_rot / pred_trans, 21 OpenPose keypoints. Hand 0 is left, hand 1 is right.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HAWOR_ROOT = Path("/data-hyp/ego_viewer/datapipe_workflow/pipelines/hawor")
DEFAULT_OUT = Path("/data/heyuping/ego_viewer/web_cache/aoe_20260201_193100_p000/keypoints.npz")
DEFAULT_HANDS = Path(
    "/data/heyuping/openaoe_sample_bi/aoe_20260201_193100_p000"
    "/ego_process/ego_hands_reconstruction/hands.npz"
)


def probe_video(video: Path) -> tuple[int, int, float]:
    size = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video),
        ],
        text=True,
    ).strip()
    width, height = (int(part) for part in size.split(",")[:2])
    rate = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video),
        ],
        text=True,
    ).strip().split(",")[0]
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den)
    else:
        fps = float(rate)
    return width, height, fps


def export_hands(hands: Path, out_dir: Path, width: int, height: int, fps: float) -> tuple[Path, Path]:
    os.chdir(HAWOR_ROOT)
    if str(HAWOR_ROOT) not in sys.path:
        sys.path.insert(0, str(HAWOR_ROOT))
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    from lib.models.mano_wrapper import MANO

    data = np.load(hands, allow_pickle=True)
    pred_betas = data["pred_betas"]
    pred_pose = data["pred_hand_pose"]
    pred_rot = data["pred_rot"]
    pred_trans = data["pred_trans"]
    pred_valid = data["pred_valid"].astype(np.float32)
    n = pred_betas.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    joints_world = np.zeros((2, n, 21, 3), dtype=np.float32)
    verts_world = None
    faces = None

    for hand_idx, is_rhand in ((0, False), (1, True)):
        if is_rhand:
            model_path = "_DATA/data/mano"
        else:
            model_path = "_DATA/data_left/mano_left"
        layer = MANO(
            model_path=model_path,
            is_rhand=is_rhand,
            flat_hand_mean=False,
            use_pca=False,
            num_pca_comps=45,
            batch_size=1,
            pose2rot=True,
        ).to(device)
        if not is_rhand:
            with torch.no_grad():
                layer.shapedirs[:, 0, :] *= -1
        layer.eval()
        if faces is None:
            faces = np.ascontiguousarray(np.asarray(layer.faces, dtype=np.int32))
        if verts_world is None:
            verts_world = np.zeros((2, n, int(layer.v_template.shape[0]), 3), dtype=np.float32)
        batch = 256
        for start in range(0, n, batch):
            end = min(n, start + batch)
            rot = torch.from_numpy(pred_rot[hand_idx, start:end]).float().to(device)
            pose = torch.from_numpy(pred_pose[hand_idx, start:end]).float().to(device)
            with torch.no_grad():
                out = layer(
                    global_orient=axis_angle_to_matrix(rot).unsqueeze(1),
                    hand_pose=axis_angle_to_matrix(pose.reshape(-1, 15, 3)),
                    betas=torch.from_numpy(pred_betas[hand_idx, start:end]).float().to(device),
                    transl=torch.from_numpy(pred_trans[hand_idx, start:end]).float().to(device),
                    pose2rot=False,
                )
            joints_world[hand_idx, start:end] = out.joints[:, :21].detach().cpu().numpy()
            verts_world[hand_idx, start:end] = out.vertices.detach().cpu().numpy()
            print(f"hand {hand_idx} {end}/{n}", flush=True)

    invalid = pred_valid < 0.5
    joints_world[invalid] = 0.0
    verts_world[invalid] = 0.0
    t_c2w = np.asarray(data["t_c2w"], dtype=np.float32)
    if t_c2w.ndim == 3:
        t_c2w = t_c2w[..., 0]
    r_c2w = np.asarray(data["R_c2w"], dtype=np.float32)
    cam_z = r_c2w[:, :, 2]
    r_w2c = np.asarray(data["R_w2c"], dtype=np.float32)
    t_w2c = np.asarray(data["t_w2c"], dtype=np.float32)
    if t_w2c.ndim == 3:
        t_w2c = t_w2c[..., 0]

    focal = float(data["focal"])
    frame = min(100, n - 1)
    joints_cam = r_w2c[frame] @ joints_world[:, frame].transpose(0, 2, 1)
    joints_cam = joints_cam.transpose(0, 2, 1) + t_w2c[frame]
    z = np.clip(joints_cam[..., 2], 1e-6, None)
    u = focal * joints_cam[..., 0] / z + width / 2
    v = focal * joints_cam[..., 1] / z + height / 2
    print("proj u", u.min(), u.max(), "v", v.min(), v.max(), "valid", pred_valid[:, frame])

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "keypoints.npz"
    mesh = out.with_name("mesh.bin")
    np.savez_compressed(
        out,
        joints_world=joints_world,
        pred_valid=pred_valid,
        R_w2c=r_w2c,
        t_w2c=t_w2c,
        cam_pos=t_c2w,
        cam_z=cam_z.astype(np.float32),
        cam_R=r_c2w,
        focal=np.float32(focal),
        width=np.int32(width),
        height=np.int32(height),
        fps=np.float32(fps),
    )
    print("wrote", out, out.stat().st_size)
    import struct
    packed = np.ascontiguousarray(verts_world.astype(np.float16))
    with mesh.open("wb") as handle:
        handle.write(struct.pack("<III", n, packed.shape[2], faces.shape[0]))
        handle.write(np.ascontiguousarray(faces).tobytes())
        handle.write(packed.tobytes())
    print("wrote", mesh, mesh.stat().st_size, "verts", packed.shape, "faces", faces.shape)
    return out, mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Export hands.npz to viewer keypoints and mesh.")
    parser.add_argument("--hands", type=Path, default=DEFAULT_HANDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT.parent)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    if args.video is not None:
        width, height, fps = probe_video(args.video)
    else:
        width = args.width if args.width is not None else 1920
        height = args.height if args.height is not None else 1080
        fps = args.fps if args.fps is not None else 30.0
    export_hands(args.hands, args.output_dir, width, height, fps)


if __name__ == "__main__":
    main()
