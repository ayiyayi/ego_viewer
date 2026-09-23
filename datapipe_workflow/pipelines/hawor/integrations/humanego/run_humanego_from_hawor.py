#!/usr/bin/env python3
"""Run the HumanEgo (Aria-path) preprocessing downstream on a HaWoR clip — no Aria.

Pipeline::

    HaWoR session ──(hawor_to_humanego adapter)──▶ HumanEgo session
                                                         │
            skip  preprocess_aria / init_preprocess      │  (no VRS/MPS available)
                                                         ▼
        indices ▶ dinosam ▶ kptsselector ▶ cotracker ▶ camtriangulator
                ▶ lama ▶ visualkpts ▶ datasetgen ▶ training_data.json

The Aria `preprocess_aria` stage (which needs `sample.vrs` + MPS) is replaced by the
adapter, which writes the per-frame `aria_cam_rgb.json` (camera `c2w`+`k`),
`aria_hands.json` (+ `hawor_hands.json`), and the top-level `aria_phases_results.json`
/ `aria_cam_rgb_config.json`. Everything downstream is file-driven and runs unchanged.

Stages from `dinosam` onward need a GPU and model weights (Grounding DINO, SAM 2,
CoTracker, LaMa) plus a task config with object prompts (see
`cfg/preprocess/tasks/scilab.yaml`). Use `--through indices` to validate the seam
without GPU/weights.

Example::

    python run_humanego_from_hawor.py \
        --hawor_session /mnt/data/liuyu/project/scilab/scilab_sample/DJI_20260409180326_0364_D_A012 \
        --out_dir       /mnt/data/liuyu/project/HumanEgo/data/scilab/aria/mps_scilab_000_vrs \
        --humanego_root /mnt/data/liuyu/project/HumanEgo \
        --task scilab --through datasetgen
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hawor_to_humanego.py")

# Aria-path downstream stages, in order. `indices` always runs first (it populates
# the image lists every later stage consumes).
STAGE_ORDER = [
    "indices", "dinosam", "kptsselector", "cotracker",
    "camtriangulator", "lama", "visualkpts", "datasetgen",
]


def _load_adapter():
    spec = importlib.util.spec_from_file_location("hawor_to_humanego", ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # needed so @dataclass can resolve its module
    spec.loader.exec_module(mod)
    return mod


def _count_frames(out_dir: str) -> int:
    all_data = os.path.join(out_dir, "preprocess", "all_data")
    return len([d for d in os.listdir(all_data)
                if os.path.isdir(os.path.join(all_data, d)) and d.isdigit()])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hawor_session", required=True, help="HaWoR session folder")
    p.add_argument("--out_dir", required=True, help="Target HumanEgo session dir")
    p.add_argument("--humanego_root", required=True, help="Path to HumanEgo repo root")
    p.add_argument("--cfg_path", default="./cfg/preprocess/base/Preprocess.yaml")
    p.add_argument("--task", default="scilab")
    p.add_argument("--through", default="datasetgen", choices=STAGE_ORDER,
                   help="Run stages up to and including this one")
    p.add_argument("--skip_ingest", action="store_true",
                   help="Assume the HumanEgo session is already populated by the adapter")
    p.add_argument("--no_frames", action="store_true", help="Adapter: skip rgb.png extraction")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--grasp_rel", type=float, default=0.5)
    p.add_argument("--use_mano", action="store_true",
                   help="Adapter: export true thumb/index-tip midpoint via MANO "
                        "(needs torch + _DATA/data/mano/MANO_RIGHT.pkl)")
    p.add_argument("--export_video", action="store_true")
    p.add_argument("--export_gif", action="store_true")
    args = p.parse_args()

    # 1) Ingest HaWoR → HumanEgo session layout.
    if not args.skip_ingest:
        adapter = _load_adapter()
        stats = adapter.ingest_session(
            args.hawor_session, args.out_dir,
            grasp_rel=args.grasp_rel, max_frames=args.max_frames,
            write_frames=not args.no_frames, use_mano=args.use_mano,
        )
        print("║ [Ingest]", stats)

    # 2) Drive the HumanEgo Aria-path downstream from the HumanEgo repo root
    #    (its configs use cwd-relative paths like ./cfg/preprocess/base/...).
    os.chdir(args.humanego_root)
    sys.path.insert(0, args.humanego_root)
    from preprocess.Preprocess import Preprocess

    pre = Preprocess(
        mps_path=os.path.abspath(args.out_dir),
        cfg_path=args.cfg_path, task=args.task,
        export_video=args.export_video, export_gif=args.export_gif,
    )
    # Substitutes the value init_preprocess() would set from MPS hand-tracking length.
    pre.num_total_frames = _count_frames(args.out_dir)
    print(f"║ [Runner] num_total_frames={pre.num_total_frames}, task={args.task}")

    stage_fn = {
        "indices": pre.preprocess_indices,
        "dinosam": pre.preprocess_dinosam,
        "kptsselector": pre.preprocess_kptsselector,
        "cotracker": pre.preprocess_cotracker,
        "camtriangulator": pre.preprocess_camtriangulator,
        "lama": pre.preprocess_lama,
        "visualkpts": pre.preprocess_visualkpts,
        "datasetgen": pre.preprocess_datasetgen,
    }
    last = STAGE_ORDER.index(args.through)
    for stage in STAGE_ORDER[:last + 1]:
        print(f"\n║ ===== stage: {stage} =====")
        stage_fn[stage]()
    print("\n║ [Runner] done.")


if __name__ == "__main__":
    main()
