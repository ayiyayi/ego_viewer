#!/usr/bin/env python3
"""Extract RGB frames + per-frame timestamps from an Aria .vrs, for feeding HaWoR.

Aria recordings are .vrs (not a normal video); HaWoR wants an mp4. This dumps the
camera-rgb stream to JPEGs + an mp4 + frames_ts.json (output-frame-index -> Aria
tracking timestamp in microseconds), so a later HaWoR run can be time-aligned to the
Aria MPS ground truth (see compare_hawor_vs_aria.py).

Runs in the `humanego` env (projectaria_tools + cv2). Example:
    python extract_vrs_rgb.py \
        --vrs <serve_bread>/sample.vrs --out /tmp/sb_rgb --fps 30 --rotate cw
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="target output fps")
    ap.add_argument("--rotate", choices=["none", "cw", "ccw"], default="cw",
                    help="Aria RGB is sideways; cw makes it upright for hand detectors")
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    provider = data_provider.create_vrs_data_provider(args.vrs)
    sid = provider.get_stream_id_from_label("camera-rgb")
    tss = provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME)
    native_fps = 1e9 / np.median(np.diff(tss))
    step = max(1, int(round(native_fps / args.fps)))
    sel = list(range(0, len(tss), step))
    if args.max_frames:
        sel = sel[:args.max_frames]

    writer, frames_ts = None, {}
    for oi, ni in enumerate(sel):
        img = provider.get_image_data_by_index(sid, ni)[0].to_numpy_array()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if args.rotate == "cw":
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "ccw":
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(args.out, f"{oi:06d}.jpg"), bgr)
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(os.path.join(args.out, "video.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(bgr)
        # tss are device-time ns; MPS csv uses tracking_timestamp_us (same clock).
        frames_ts[oi] = int(tss[ni] // 1000)
    if writer:
        writer.release()
    json.dump(frames_ts, open(os.path.join(args.out, "frames_ts.json"), "w"))
    print(json.dumps({"out": args.out, "native_fps": round(native_fps, 2),
                      "step": step, "frames": len(sel),
                      "video": os.path.join(args.out, "video.mp4")}, indent=2))


if __name__ == "__main__":
    main()
