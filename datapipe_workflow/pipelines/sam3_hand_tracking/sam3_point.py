#!/usr/bin/env python3
"""Production SAM3 point-prompt tight boxes from SAM mask.

This is the SAM mask point-prompt strategy without the comparison/baseline
rendering path. It consumes a segment directory containing frames/ and
raw detector side boxes in detection/bboxes_2d.pth, then writes SAM tight boxes
for HaWoR crop inference.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
SAM3_SRC = os.environ.get("SAM3_SRC") or os.environ.get("SAM_SRC") or ""
if SAM3_SRC:
    sys.path.insert(0, SAM3_SRC)

from . import sam_repropagate as simple  # noqa: E402

SIDE_NAMES = simple.SIDE_NAMES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--sam-version", choices=["sam3"], default="sam3")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--score-thresh", type=float, default=0.70)
    parser.add_argument("--bottom-y2-frac", type=float, default=0.98)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--point-expand", type=float, default=1.1)
    parser.add_argument("--prompt-geometry", choices=["ring", "below", "box", "box_below"], default="box_below")
    parser.add_argument("--neg-below-count", type=int, default=5)
    parser.add_argument("--neg-below-frac", type=float, default=0.05)
    parser.add_argument("--sam-output-thresh", type=float, default=0.15)
    parser.add_argument("--default-output-prob-thresh", type=float, default=0.15)
    parser.add_argument("--prompt-text", default="hand")
    parser.add_argument("--left-prompt-text", default="")
    parser.add_argument("--right-prompt-text", default="")
    parser.add_argument("--tracking-mode", choices=["separate"], default="separate")
    parser.add_argument("--init-mode", choices=["point", "semantic"], default="point")
    parser.add_argument("--propagation-scheme", choices=["dual", "single_anchor"], default="single_anchor")
    parser.add_argument("--compact-masks", action="store_true", help="Write lossless packed masks and direct SLAM mask archive.")
    parser.add_argument("--validate-tracks", action="store_true", help="Validate against cached detections and locally reinitialize failed tracks.")
    parser.add_argument("--cuda-invalid-retries", type=int, default=1)
    parser.add_argument("--git-commit", default="unknown")
    return parser.parse_args(argv)


def now():
    return time.perf_counter()


def dump_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def clean_retry_outputs(out_dir: Path):
    for name in ["masks", "framewise_masks"]:
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in [
        "run_config.json",
        "prompt_seed_debug.json",
        "point_prompt_records.jsonl",
        "prompt_points.json",
        "point_prompt_summary.json",
        "sam_tight_bboxes_2d.jsonl",
        "sam_tight_bboxes_2d.csv",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def side_margin(det):
    return simple.side_margin(det)


def prompt_text_for_side(args, side):
    text = args.left_prompt_text if int(side) == 0 else args.right_prompt_text
    return text if text else args.prompt_text


def select_prompt_seeds(det_by_frame, num_frames: int, height: int, args):
    window = int(round(float(args.window_sec) * float(args.fps)))
    assert window > 0, window
    bottom_y2 = float(args.bottom_y2_frac) * float(height)
    seed_by_side = {0: [], 1: []}
    debug = {"window_frames": window, "bottom_y2_px": bottom_y2, "score_thresh": float(args.score_thresh), "sides": {}}
    for side in [0, 1]:
        side_debug = {"valid_candidate_count": 0, "bottom_skip_count": 0, "low_score_skip_count": 0, "windows": []}
        for start in range(0, int(num_frames), window):
            end = min(int(num_frames) - 1, start + window - 1)
            candidates = []
            for frame_idx in range(start, end + 1):
                for det in det_by_frame.get(frame_idx, []):
                    if int(det["side"]) != int(side):
                        continue
                    if float(det["score"]) <= float(args.score_thresh):
                        side_debug["low_score_skip_count"] += 1
                        continue
                    if float(det["box_xyxy"][3]) >= bottom_y2:
                        side_debug["bottom_skip_count"] += 1
                        continue
                    side_debug["valid_candidate_count"] += 1
                    candidates.append((int(frame_idx), det))
            if candidates:
                seed_frame_idx, det = max(candidates, key=lambda item: (float(item[1]["score"]), -int(item[0])))
                seed = {
                    "side": int(side),
                    "side_name": SIDE_NAMES[int(side)],
                    "window_start": int(start),
                    "window_end": int(end),
                    "frame_idx": int(seed_frame_idx),
                    "score": float(det["score"]),
                    "side_margin": float(side_margin(det)),
                    "track_id": int(det["track_id"]),
                    "box_xyxy": [float(x) for x in det["box_xyxy"]],
                    "det_box_area": float(simple.box_area_xyxy(det["box_xyxy"])),
                }
                seed_by_side[side].append(seed)
                side_debug["windows"].append(seed)
            else:
                side_debug["windows"].append({"side": int(side), "window_start": int(start), "window_end": int(end), "status": "no_valid_seed"})
        debug["sides"][SIDE_NAMES[side]] = side_debug
    return seed_by_side, debug








def expand_box_xyxy(box, scale, width, height):
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = (x2 - x1) * float(scale), (y2 - y1) * float(scale)
    return np.array([
        max(0.0, cx - bw * 0.5),
        max(0.0, cy - bh * 0.5),
        min(float(width - 1), cx + bw * 0.5),
        min(float(height - 1), cy + bh * 0.5),
    ], dtype=np.float32)


def bbox_to_instance_points(box, width, height, expand, geometry="ring", neg_below_count=5, neg_below_frac=0.05):
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    def below_negatives():
        # A row of negatives just below the hand box (forearm enters from below in
        # head-mounted video). x uniform across box width, y a fraction of box height
        # below the bottom edge. No side/top negatives so fingers are not clipped.
        bh = max(1.0, y2 - y1)
        ny = min(float(height - 1), y2 + float(neg_below_frac) * bh)
        return [[float(x) / width, ny / height] for x in np.linspace(x1, x2, int(neg_below_count))]

    if geometry == "below":
        # 1 positive center + N forearm negatives below.
        negs = below_negatives()
        points = [[cx / width, cy / height]] + negs
        labels = [1] + [0] * len(negs)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [x1, y1, x2, y2]

    if geometry in ("box", "box_below"):
        # SAM2-style box prompt: two corners encoded as points with labels 2 (top-left)
        # and 3 (bottom-right). Optionally add forearm negatives ("box_below").
        points = [[x1 / width, y1 / height], [x2 / width, y2 / height]]
        labels = [2, 3]
        if geometry == "box_below":
            negs = below_negatives()
            points = points + negs
            labels = labels + [0] * len(negs)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [x1, y1, x2, y2]

    # geometry == "ring" (current default behavior): 1 positive center + 12 negatives on
    # the (optionally expanded) box edges.
    ex1, ey1, ex2, ey2 = [float(x) for x in expand_box_xyxy(box, expand, width, height)]
    points = [[cx / width, cy / height]]
    for x in np.linspace(ex1, ex2, 3):
        points.append([float(x) / width, ey1 / height])
    for x in np.linspace(ex1, ex2, 3):
        points.append([float(x) / width, ey2 / height])
    for y in np.linspace(ey1, ey2, 3):
        points.append([ex1 / width, float(y) / height])
    for y in np.linspace(ey1, ey2, 3):
        points.append([ex2 / width, float(y) / height])
    labels = [1] + [0] * 12
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32), [float(x) for x in [ex1, ey1, ex2, ey2]]


def choose_obj(outputs, prompt_box, width, height):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    rows = []
    for pos, obj_id in enumerate(ids):
        box = simple.output_box_xyxy(boxes_rel[pos], masks[pos], width, height)
        rows.append((simple.xyxy_iou(np.asarray(prompt_box, dtype=np.float32), box), int(masks[pos].sum()), float(probs[pos]), int(obj_id), pos, box))
    assert rows, "semantic init returned no SAM objects"
    return max(rows, key=lambda x: (x[0], x[1], x[2]))


def choose_semantic_objects(outputs, init_by_side, width, height):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    side_ids = sorted(init_by_side)
    assert len(ids) >= len(side_ids), f"semantic init returned {len(ids)} objects for {len(side_ids)} side prompts"
    rows_by_side = {}
    for side in side_ids:
        rows = []
        prompt_box = np.asarray(init_by_side[side]["box_xyxy"], dtype=np.float32)
        for pos, obj_id in enumerate(ids):
            box = simple.output_box_xyxy(boxes_rel[pos], masks[pos], width, height)
            rows.append(
                {
                    "side": int(side),
                    "obj_id": int(obj_id),
                    "pos": int(pos),
                    "prompt_iou": float(simple.xyxy_iou(prompt_box, box)),
                    "mask_area": int(masks[pos].sum()),
                    "prob": float(probs[pos]),
                    "box_xyxy": [float(x) for x in box],
                }
            )
        rows_by_side[side] = rows
    best = None
    for positions in permutations(range(len(ids)), len(side_ids)):
        picked = [rows_by_side[side][pos] for side, pos in zip(side_ids, positions)]
        obj_ids = [row["obj_id"] for row in picked]
        if len(set(obj_ids)) != len(obj_ids):
            continue
        quality = (sum(row["prompt_iou"] for row in picked), sum(row["mask_area"] for row in picked), sum(row["prob"] for row in picked))
        if best is None or quality > best[0]:
            best = (quality, picked)
    assert best is not None, f"could not assign unique semantic objects for sides {side_ids}"
    return {int(row["side"]): row for row in best[1]}, rows_by_side


def find_obj(outputs, obj_id: int):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    matches = np.where(ids == int(obj_id))[0]
    assert len(matches) == 1, f"expected obj_id={obj_id} once, got {ids.tolist()}"
    pos = int(matches[0])
    return boxes_rel[pos], masks[pos], float(probs[pos])


def find_obj_or_none(outputs, obj_id: int):
    ids, boxes_rel, masks, probs = simple.normalize_outputs(outputs)
    matches = np.where(ids == int(obj_id))[0]
    if len(matches) == 0:
        return None
    assert len(matches) == 1, f"expected obj_id={obj_id} once, got {ids.tolist()}"
    pos = int(matches[0])
    return boxes_rel[pos], masks[pos], float(probs[pos])


def write_record_from_obj(mask_dir: Path, frame_idx: int, side: int, source: str, obj_id: int, boxes_rel, mask, prob, width: int, height: int, prompt_frame: int):
    box = simple.mask_to_box(mask) if int(mask.sum()) else simple.output_box_xyxy(boxes_rel, mask, width, height)
    mask_area = int(mask.sum())
    mask_path = None
    if mask_area > 0:
        mask_path = mask_dir / SIDE_NAMES[int(side)] / f"{source}_obj{obj_id}_frame{frame_idx:06d}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_dir.name == 'packed_masks':
            from .mask_io import write_mask
            mask_path = Path(write_mask(mask_path, mask))
        else:
            ok = cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            assert ok, f"failed to write {mask_path}"
    return {
        "frame_idx": int(frame_idx),
        "side": int(side),
        "side_name": SIDE_NAMES[int(side)],
        "obj_id": int(obj_id),
        "source": source,
        "prompt_frame_idx": int(prompt_frame),
        "prob": float(prob),
        "mask_area": int(mask_area),
        "box_xyxy": [float(x) for x in box],
        "mask_path": None if mask_path is None else str(mask_path),
    }


def write_record(mask_dir: Path, frame_idx: int, side: int, source: str, obj_id: int, outputs, width: int, height: int, prompt_frame: int):
    boxes_rel, mask, prob = find_obj(outputs, obj_id)
    return write_record_from_obj(mask_dir, frame_idx, side, source, obj_id, boxes_rel, mask, prob, width, height, prompt_frame)


def write_record_if_present(mask_dir: Path, frame_idx: int, side: int, source: str, obj_id: int, outputs, width: int, height: int, prompt_frame: int):
    obj = find_obj_or_none(outputs, obj_id)
    if obj is None:
        return None
    boxes_rel, mask, prob = obj
    return write_record_from_obj(mask_dir, frame_idx, side, source, obj_id, boxes_rel, mask, prob, width, height, prompt_frame)


def merge_records(records):
    by_key = {}
    for rec in records:
        if int(rec["mask_area"]) <= 0:
            continue
        key = (int(rec["frame_idx"]), int(rec["side"]))
        quality = (float(rec["prob"]), -abs(int(rec["frame_idx"]) - int(rec["prompt_frame_idx"])), int(rec["mask_area"]))
        if key not in by_key or quality > by_key[key][0]:
            by_key[key] = (quality, rec)
    return {key: rec for key, (_, rec) in by_key.items()}


def export_framewise(frames, selected, out_dir: Path, width: int, height: int, compact=False):
    from .mask_io import read_mask, reduce_union
    if compact:
        rows, packed = [], []
        for t in range(len(frames)):
            row = {'frame_idx': t}; masks = []
            for side, name in enumerate(('left','right')):
                rec = selected.get((t,side))
                row[f'{name}_mask_path'] = None if rec is None else rec['mask_path']
                row[f'{name}_mask_area'] = 0 if rec is None else rec['mask_area']
                row[f'{name}_sam_tight_box_xyxy'] = None if rec is None else rec['box_xyxy']
                if rec is not None: masks.append(read_mask(rec['mask_path']))
            low = reduce_union(masks, height, width)
            packed.append(np.packbits(low)); rows.append(row)
        np.savez_compressed(out_dir/'slam_masks.npz', packed=np.stack(packed), shape=np.array(low.shape))
        (out_dir/'sam_tight_bboxes_2d.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return rows
    framewise_dir = out_dir / "framewise_masks"
    zero = np.zeros((height, width), dtype=np.uint8)
    rows = []
    for side in [0, 1]:
        (framewise_dir / SIDE_NAMES[side]).mkdir(parents=True, exist_ok=True)
    for frame_idx in range(len(frames)):
        row = {"frame_idx": int(frame_idx)}
        for side in [0, 1]:
            side_name = SIDE_NAMES[side]
            out_path = framewise_dir / side_name / f"{frame_idx:06d}.png"
            rec = selected.get((frame_idx, side))
            if rec is None:
                ok = cv2.imwrite(str(out_path), zero)
                assert ok, f"failed to write {out_path}"
                row[f"{side_name}_mask_path"] = str(out_path)
                row[f"{side_name}_mask_area"] = 0
                row[f"{side_name}_sam_tight_box_xyxy"] = None
                continue
            mask = read_mask(rec["mask_path"])
            ok = cv2.imwrite(str(out_path), mask.astype(np.uint8) * 255)
            assert ok, f"failed to write {out_path}"
            row[f"{side_name}_mask_path"] = str(out_path)
            row[f"{side_name}_mask_area"] = int(mask.sum())
            row[f"{side_name}_sam_tight_box_xyxy"] = [float(x) for x in simple.mask_to_box(mask)]
        rows.append(row)
    with (out_dir / "sam_tight_bboxes_2d.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "sam_tight_bboxes_2d.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_idx", "left_mask_path", "left_mask_area", "left_sam_tight_box_xyxy", "right_mask_path", "right_mask_area", "right_sam_tight_box_xyxy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_tracks(args, predictor, frames, det_by_frame, selected, out_dir, mask_dir, width, height, inmemory_frames=None):
    from .validation import validate_and_recover
    import tempfile
    def recover(side, start, stop, anchor, detection):
        # A fresh short session cannot inherit the failed track's memory.
        with tempfile.TemporaryDirectory(prefix='sam_recovery_', dir=out_dir) as folder:
            if inmemory_frames is not None:
                resource = frames[start:stop]
            else:
                for i, path in enumerate(frames[start:stop]):
                    (Path(folder) / f'{i:06d}.jpg').symlink_to(Path(path).resolve())
                resource = folder
            session = predictor.handle_request({'type': 'start_session', 'resource_path': resource})
            sid = session['session_id']; obj = side + 1; local = anchor - start
            recovered = {}
            try:
                points, labels, _ = bbox_to_instance_points(detection['box_xyxy'], width, height,
                    args.point_expand, geometry=args.prompt_geometry,
                    neg_below_count=args.neg_below_count, neg_below_frac=args.neg_below_frac)
                response = predictor.handle_request(dict(type='add_prompt', session_id=sid,
                    frame_index=local, points=points, point_labels=labels, obj_id=obj,
                    rel_coordinates=True, output_prob_thresh=args.sam_output_thresh))
                name = f'recovery_{start:06d}'
                rec = write_record_if_present(mask_dir, anchor, side, name, obj,
                    response['outputs'], width, height, anchor)
                if rec is not None: recovered[anchor] = rec
                for direction, count in [('forward', stop-anchor), ('backward', local+1)]:
                    for item in predictor.handle_stream_request(dict(type='propagate_in_video',
                        session_id=sid, propagation_direction=direction, start_frame_index=local,
                        max_frame_num_to_track=count, output_prob_thresh=args.sam_output_thresh)):
                        t = start + int(item['frame_index'])
                        if not start <= t < stop: continue
                        rec = write_record_if_present(mask_dir, t, side, name, obj,
                            item['outputs'], width, height, anchor)
                        if rec is not None: recovered[t] = rec
            finally:
                predictor.handle_request({'type': 'close_session', 'session_id': sid})
            return recovered
    selected, blocked, validation_report = validate_and_recover(
        selected, det_by_frame, len(frames), args.fps, recover)
    with (out_dir / 'validated_records.jsonl').open('w') as handle:
        for key in sorted(selected): handle.write(json.dumps(selected[key]) + '\n')
    np.savez_compressed(out_dir / 'tracking_validation.npz', interpolation_blocked=blocked)
    dump_json(out_dir / 'tracking_validation.json', {'version': 1, 'confirmation_score': .7,
        'max_unconfirmed_span_s': .5, 'recovery_window_s': .5, 'sides': validation_report})
    return selected


def run_point_prompt(args, predictor=None, inmemory_frames=None):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "run_config.json", vars(args))
    segment_dir = Path(args.segment_dir)
    assert (segment_dir / "detection" / "bboxes_2d.npz").exists(), segment_dir / "detection" / "bboxes_2d.npz"
    if inmemory_frames is not None:
        # 25-12 in-memory SAM3: frames are a list[PIL.Image] (RGB, at the sam3 resized size);
        # start_session accepts the list directly (load_resource_as_video_frames list branch),
        # so there is no segment_dir/frames on disk and no JPEG re-encode.
        frames = inmemory_frames
        sam3_resource = inmemory_frames
        width, height = frames[0].size  # PIL size is (W, H)
    else:
        assert (segment_dir / "frames").exists(), segment_dir / "frames"
        frames = simple.sorted_frames(segment_dir)
        first = cv2.imread(str(frames[0]))
        assert first is not None, f"failed to read first frame: {frames[0]}"
        height, width = first.shape[:2]
        sam3_resource = str(segment_dir / "frames")
    det = simple.load_detection(segment_dir / "detection" / "bboxes_2d.npz")
    det_by_frame = simple.detection_index(det)
    seeds_by_side, seed_debug = select_prompt_seeds(det_by_frame, len(frames), height, args)
    dump_json(out_dir / "prompt_seed_debug.json", seed_debug)

    t_total = now()
    runtime = defaultdict(float)
    t = now()
    if not any(seeds_by_side.values()) and not args.validate_tracks:
        runtime["model_load_sec"] = 0.0
    elif predictor is not None:
        # Injected warm predictor (persistent worker, D-02): reuse the checkpoint
        # loaded once at worker startup; no per-segment build / interpreter spawn.
        runtime["model_load_sec"] = 0.0
        runtime["predictor_reused"] = True
    else:
        # Standalone / subprocess entry: build exactly as before (numerics unchanged).
        predictor = simple.build_predictor(args.checkpoint_path, args.default_output_prob_thresh, args.sam_version)
        runtime["model_load_sec"] = now() - t
        runtime["predictor_reused"] = False
    mask_dir = out_dir / ("packed_masks" if args.compact_masks else "masks")
    all_records = []
    all_prompt_rows = []
    side_summaries = {}

    for side in [0, 1]:
        side_name = SIDE_NAMES[side]
        side_seeds = seeds_by_side[side]
        if not side_seeds:
            side_summaries[side_name] = {"status": "no_valid_prompt_seed", "prompt_seed_count": 0, "tracking_mode": args.tracking_mode}
            continue
        init_seed = max(side_seeds, key=lambda s: (float(s["score"]), -int(s["frame_idx"])))
        t = now()
        session = predictor.handle_request({"type": "start_session", "resource_path": sam3_resource})
        session_id = session["session_id"]
        runtime["session_init_sec"] += now() - t
        try:
            if args.init_mode == "semantic":
                t = now()
                prompt_box_rel = simple.xyxy_abs_to_xywh_rel(np.asarray(init_seed["box_xyxy"], dtype=np.float32), width, height)
                semantic = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(init_seed["frame_idx"]),
                    "text": prompt_text_for_side(args, side),
                    "bounding_boxes": prompt_box_rel[None],
                    "bounding_box_labels": np.ones((1,), dtype=np.int64),
                    "output_prob_thresh": float(args.sam_output_thresh),
                })
                runtime["semantic_init_sec"] += now() - t
                _iou, _area, _prob, obj_id, _pos, _box = choose_obj(semantic["outputs"], init_seed["box_xyxy"], width, height)
                all_records.append(write_record(mask_dir, int(init_seed["frame_idx"]), side, "semantic_init", obj_id, semantic["outputs"], width, height, int(init_seed["frame_idx"])))
            else:
                obj_id = int(side) + 1

            for seed_i, seed in enumerate(side_seeds):
                points, labels, expanded_box = bbox_to_instance_points(seed["box_xyxy"], width, height, args.point_expand, geometry=args.prompt_geometry, neg_below_count=args.neg_below_count, neg_below_frac=args.neg_below_frac)
                t = now()
                resp = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(seed["frame_idx"]),
                    "points": points,
                    "point_labels": labels,
                    "obj_id": int(obj_id),
                    "rel_coordinates": True,
                    "output_prob_thresh": float(args.sam_output_thresh),
                })
                runtime["point_prompt_sec"] += now() - t
                all_records.append(write_record(mask_dir, int(seed["frame_idx"]), side, f"point_seed{seed_i:03d}", obj_id, resp["outputs"], width, height, int(seed["frame_idx"])))
                all_prompt_rows.append({**seed, "obj_id": int(obj_id), "points_rel": points.tolist(), "point_labels": labels.tolist(), "expanded_box_xyxy": expanded_box})

            min_frame = min(int(s["frame_idx"]) for s in side_seeds)
            max_frame = max(int(s["frame_idx"]) for s in side_seeds)
            if args.propagation_scheme == "single_anchor":
                # One estimate per frame: forward B->end, backward B->0 from the init
                # seed; window seeds stay cond frames in both directions.
                anchor_frame = int(init_seed["frame_idx"])
                forward_start, backward_start = anchor_frame, anchor_frame
                forward_prompt, backward_prompt = anchor_frame, anchor_frame
                runtime[f"propagation_anchor_frame_{side_name}"] = anchor_frame
            else:
                forward_start, backward_start = min_frame, max_frame
                forward_prompt, backward_prompt = min_frame, max_frame
            t = now()
            for stream_out in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": int(forward_start),
                "max_frame_num_to_track": int(len(frames) - forward_start),
                "output_prob_thresh": float(args.sam_output_thresh),
            }):
                rec = write_record_if_present(mask_dir, int(stream_out["frame_index"]), side, "forward", obj_id, stream_out["outputs"], width, height, forward_prompt)
                if rec is None:
                    runtime[f"forward_missing_obj_frames_{side_name}"] += 1
                    continue
                all_records.append(rec)
            runtime["forward_propagate_sec"] += now() - t
            t = now()
            for stream_out in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "backward",
                "start_frame_index": int(backward_start),
                "max_frame_num_to_track": int(backward_start + 1),
                "output_prob_thresh": float(args.sam_output_thresh),
            }):
                rec = write_record_if_present(mask_dir, int(stream_out["frame_index"]), side, "backward", obj_id, stream_out["outputs"], width, height, backward_prompt)
                if rec is None:
                    runtime[f"backward_missing_obj_frames_{side_name}"] += 1
                    continue
                all_records.append(rec)
            runtime["backward_propagate_sec"] += now() - t
            side_summaries[side_name] = {"status": "ok", "obj_id": int(obj_id), "prompt_seed_count": len(side_seeds), "init_seed": init_seed, "tracking_mode": args.tracking_mode, "init_mode": args.init_mode}
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    selected = merge_records(all_records)
    if args.validate_tracks:
        selected = validate_tracks(args, predictor, frames, det_by_frame, selected,
                                   out_dir, mask_dir, width, height, inmemory_frames)
    rows = export_framewise(frames, selected, out_dir, width, height, compact=args.compact_masks)
    records_path = out_dir / "point_prompt_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    dump_json(out_dir / "prompt_points.json", {"prompts": all_prompt_rows})
    runtime["total_sec"] = now() - t_total
    summary = {
        "status": "ok",
        "bbox_source": "sam3_point_prompt",
        "tracking_mode": args.tracking_mode,
        "init_mode": args.init_mode,
        "propagation_scheme": args.propagation_scheme,
        "git_commit": args.git_commit,
        "segment_dir": str(segment_dir),
        "output_dir": str(out_dir),
        "total_frames": int(len(frames)),
        "image_size_hw": [int(height), int(width)],
        "side_summaries": side_summaries,
        "runtime": dict(runtime),
        "records_jsonl": str(records_path),
        "sam_tight_bboxes_jsonl": str(out_dir / "sam_tight_bboxes_2d.jsonl"),
        "sam_tight_bboxes_csv": str(out_dir / "sam_tight_bboxes_2d.csv"),
        "visible_rows": int(sum(1 for row in rows for side in ["left", "right"] if int(row.get(f"{side}_mask_area") or 0) > 0)),
    }
    dump_json(out_dir / "point_prompt_summary.json", summary)
    print("POINT_PROMPT_RESULT " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    args = parse_args()
    try:
        run_point_prompt(args)
    except RuntimeError as exc:
        attempt = int(os.environ.get("PHASE18_SAM3_CUDA_INVALID_RETRY_ATTEMPT", "0"))
        if "CUDA driver error: invalid argument" in str(exc) and attempt < int(args.cuda_invalid_retries):
            out_dir = Path(args.output_dir)
            clean_retry_outputs(out_dir)
            env = dict(os.environ)
            env["PHASE18_SAM3_CUDA_INVALID_RETRY_ATTEMPT"] = str(attempt + 1)
            print(
                f"SAM3_CUDA_INVALID_RETRY attempt={attempt + 1} max={int(args.cuda_invalid_retries)} output_dir={out_dir}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(3.0)
            os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
        raise


if __name__ == "__main__":
    main()
