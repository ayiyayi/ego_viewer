"""Separate-process SAM3 adapter; no torch or SAM import in the caller."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def environment_config():
    names = ('EGO_VIEWER_SAM3_CHECKPOINT', 'EGO_VIEWER_SAM3_PYTHON', 'EGO_VIEWER_SAM3_SRC')
    values = [os.environ.get(n) for n in names]
    if not any(values):
        return None
    if not all(values):
        raise ValueError('SAM3 requires all of: ' + ', '.join(names))
    checkpoint = Path(values[0]).expanduser().resolve()
    # Venv interpreters are symlinks to the base Python. Resolving them drops
    # pyvenv.cfg and runs the system interpreter, which has no SAM3 packages.
    python_bin = Path(values[1]).expanduser()
    if not python_bin.is_absolute():
        python_bin = Path.cwd() / python_bin
    python_bin = python_bin.absolute()
    sam3_src = Path(values[2]).expanduser().resolve()
    config = dict(checkpoint=checkpoint, python_bin=python_bin, sam3_src=sam3_src)
    validate_config(**config)
    return config


def validate_config(checkpoint, python_bin, sam3_src):
    for path in (checkpoint, python_bin, Path(sam3_src) / 'sam3' / 'model_builder.py'):
        if not Path(path).is_file():
            raise FileNotFoundError(f'SAM3 required file missing: {path}')
    python = Path(python_bin)
    if any(python.is_relative_to(Path('/data/heyuping/ego_viewer/envs') / name) for name in ('hawor', 'wilor')):
        raise ValueError('SAM3 requires a separate environment, not hawor/wilor')


def load_box_archive(path, frame_count, width, height):
    with np.load(path, allow_pickle=False) as data:
        boxes, valid = data['boxes'].copy(), data['valid'].astype(bool)
        if boxes.shape != (frame_count, 2, 4) or valid.shape != (frame_count, 2):
            raise ValueError('SAM3 boxes/frame count mismatch')
        if int(data['width']) != width or int(data['height']) != height:
            raise ValueError('SAM3 boxes/image size mismatch')
        if str(data['side_order'].item()) != 'left,right':
            raise ValueError('SAM3 side order must be left,right')
        selected = boxes[valid]
        if not np.isfinite(boxes).all() or np.any(selected[:, 2:] <= selected[:, :2]):
            raise ValueError('Invalid SAM3 box coordinates')
    return boxes, valid


def save_box_archive(path, rows, width, height):
    boxes = np.zeros((len(rows), 2, 4), dtype=np.float32)
    valid = np.zeros((len(rows), 2), dtype=bool)
    for t, row in enumerate(rows):
        for side in (0, 1):
            if row.get(side) is not None:
                box = np.asarray(row[side], dtype=np.float32)
                if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                    raise ValueError(f'Invalid SAM3 box at {t}, side {side}')
                boxes[t, side] = box
                valid[t, side] = True
    np.savez_compressed(path, boxes=boxes, valid=valid, width=width, height=height, side_order='left,right')


def refine_boxes_with_sam3(frames_dir, detections, work_dir, checkpoint, python_bin, sam3_src,
                           *, fps, prompt_geometry='box_below', propagation_scheme='single_anchor',
                           chunk_frames=600):
    """Missing masks stay missing. Sessions are bounded; source frames are not resampled.

    0.70 is the detector seed threshold; SAM output threshold is 0.15.
    """
    validate_config(checkpoint, python_bin, sam3_src)
    if fps <= 0 or chunk_frames <= 0:
        raise ValueError('fps and chunk_frames must be positive')
    paths = sorted((p for p in Path(frames_dir).iterdir() if p.suffix.lower() == '.jpg'), key=lambda p: int(p.stem))
    if len(paths) < len(detections) or not detections:
        raise ValueError('Not enough aligned source frames for SAM3')
    work_dir = Path(work_dir).resolve()
    if work_dir.is_relative_to(Path('/data-hyp')):
        raise ValueError('SAM3 scratch must be on /data/heyuping or /tmp, not /data-hyp')
    work_dir.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='sam3-', dir=work_dir))
    env = os.environ.copy()
    env['SAM3_SRC'] = str(Path(sam3_src).resolve())
    rows = []
    for start in range(0, len(detections), chunk_frames):
        subset = detections[start:start + chunk_frames]
        segment = run / f'{start:06d}'
        (segment / 'frames').mkdir(parents=True)
        (segment / 'detection').mkdir()
        for t, source in enumerate(paths[start:start + len(subset)]):
            (segment / 'frames' / f'{t:06d}.jpg').symlink_to(source.resolve())
        records = [(t, side, item) for t, row in enumerate(subset) for side, item in row.items() if item is not None]
        np.savez(segment / 'detection' / 'bboxes_2d.npz',
                 frame_idx=np.array([t for t, _, _ in records], dtype=np.int64),
                 track_id=np.array([s for _, s, _ in records], dtype=np.int64),
                 boxes_xyxy=np.array([v['box'] for _, _, v in records], dtype=np.float32).reshape(-1, 4),
                 scores=np.array([v['score'] for _, _, v in records], dtype=np.float32),
                 handedness=np.array([s for _, s, _ in records], dtype=np.int64))
        out = segment / 'output'
        cmd = [str(python_bin), str(Path(__file__).with_name('run_sam3_point_prompt.py')),
               '--segment-dir', str(segment), '--output-dir', str(out), '--checkpoint-path', str(checkpoint),
               '--prompt-geometry', prompt_geometry, '--propagation-scheme', propagation_scheme,
               '--fps', str(fps), '--tracking-mode', 'separate', '--init-mode', 'point']
        subprocess.run(cmd, check=True, env=env)
        with (out / 'sam_tight_bboxes_2d.jsonl').open() as handle:
            exported = [json.loads(line) for line in handle]
        if [r['frame_idx'] for r in exported] != list(range(len(subset))):
            raise ValueError('SAM3 returned an incomplete or reordered timeline')
        rows.extend({0: r['left_sam_tight_box_xyxy'], 1: r['right_sam_tight_box_xyxy']} for r in exported)
    return rows
