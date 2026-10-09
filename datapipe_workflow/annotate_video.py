#!/usr/bin/env python3
"""Annotate one ego video: actions, scene, and hands.

The WiLoR hand path is detection, resident SAM3, SAM-mask SLAM/Metric3D,
and WiLoR poses. It does not run HaWoR hand reconstruction or infilling.
After the lift, wrists faster than 1.5 m/s are masked, and eligible gaps up to
0.3 s are interpolated in MANO parameter space, followed by bounded wrist
smoothing. Pass --hands hawor to keep HaWoR's own hands.

Only the video path is required. API settings come from
pipelines/caption/api_config.json. HaWoR uses the Python in
env_profiles/default.yaml. WiLoR uses /data/heyuping/ego_viewer/envs/wilor.
SAM settings are read from the environment or env_profiles/sam3.env.

From the datapipe_workflow directory:

  python annotate_video.py /path/to/video.mp4 --hands-only
  python annotate_video.py /path/to/video.mp4 --actions-only --labels /path/to/labels.txt

The result directory is <video_stem>_annotation next to the video, in the
layout the viewer discovers under datapipe_workflow/examples:

  input/<video>.mp4
  output/ego_action_annotation.json
  output/keypoints.npz
  output/mesh.bin
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTION_DIR = ROOT / "pipelines" / "caption"
HAWOR_PROCESSOR = ROOT / "pipelines" / "hawor" / "scripts" / "hawor_video_processor.py"
WILOR_SCRIPT = ROOT / "pipelines" / "wilor" / "export_viewer.py"
WILOR_PYTHON = Path("/data/heyuping/ego_viewer/envs/wilor/bin/python")
ENV_PROFILE = ROOT / "env_profiles" / "default.yaml"


def profile_python() -> Path:
    text = ENV_PROFILE.read_text(encoding="utf-8")
    match = re.search(r"^\s*python:\s*(\S+)", text, flags=re.M)
    if not match:
        raise SystemExit(f"no python path in {ENV_PROFILE}")
    return Path(match.group(1))


def profile_gpu() -> str:
    text = ENV_PROFILE.read_text(encoding="utf-8")
    match = re.search(r'gpus:\s*\["?(\d+)"?', text)
    return match.group(1) if match else "0"


def ensure_hawor_python() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        python = profile_python()
        if not python.is_file():
            raise SystemExit(f"HaWoR python not found: {python}")
        if python.resolve() != Path(sys.executable).resolve():
            os.execv(str(python), [str(python), *sys.argv])
        raise SystemExit("the HaWoR interpreter cannot import torch")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def place_video(video: Path, output: Path) -> Path:
    dest_dir = output / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / video.name
    if dest.resolve() == video.resolve():
        return dest
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    shutil.copy2(video, dest)
    return dest


def run_actions(caption, video: Path, output: Path, labels: Path | None = None) -> dict:
    work = output / "_caption"
    if work.exists():
        shutil.rmtree(work)
    reference_for_clip = None
    quantize_step = 0.0
    if labels is not None:
        labels = labels.expanduser().resolve()
        if not labels.is_file():
            raise SystemExit(f"labels not found: {labels}")
        from pipelines.caption.finebio_reference import STEP, collapse_reference, load_rows, reference_text

        reference = collapse_reference(load_rows(labels), duration=1e9)

        def reference_for_clip(start, end, items=reference):
            return reference_text(items, start, end)

        quantize_step = STEP
    cfg = caption.load_api_config(None)
    base_url = cfg.get("base_url") or cfg.get("url") or os.environ.get("OPENAI_BASE_URL")
    api_key = (
        cfg.get("api_key")
        or cfg.get("key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    model = cfg.get("model") or "gemini-3.6-flash"
    if not base_url or not api_key or api_key == "<YOUR_API_KEY>":
        raise SystemExit("set base_url and api_key in pipelines/caption/api_config.json")

    caption._pop_proxy()
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    result = caption.annotate_video(
        video,
        work,
        client,
        model,
        caption.DEFAULT_PROMPT,
        caption.CLIP_SEC,
        0.1,
        "",
        caption.DEFAULT_SCENE_PROMPT,
        reference_for_clip=reference_for_clip,
        quantize_step=quantize_step,
    )
    anno_dir = output / "output"
    anno_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "ego_action_annotation.json", anno_dir / "ego_action_annotation.json")
    shutil.rmtree(work)
    return result


def load_sam3_env() -> None:
    """Fill SAM paths from env_profiles/sam3.env when the shell did not."""
    needed = ("EGO_VIEWER_SAM3_CHECKPOINT", "EGO_VIEWER_SAM3_PYTHON", "EGO_VIEWER_SAM3_SRC")
    if all(os.environ.get(name) for name in needed):
        return
    profile = ROOT / "env_profiles" / "sam3.env"
    if not profile.is_file():
        return
    for line in profile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def hand_work_dir(video: Path, backend: str) -> Path:
    root = Path(os.environ.get('EGO_VIEWER_WORK_ROOT', '/data/heyuping/ego_viewer/demo')).expanduser().resolve()
    if root.is_relative_to(Path('/data-hyp')):
        raise ValueError('Hand pipeline scratch cannot be on /data-hyp')
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'{video.stem}_{backend}_', dir=root))


def run_hands(video: Path, output: Path) -> Path:
    from pipelines.sam3_hand_tracking.refine import environment_config
    environment_config()
    hawor = load_module("hawor_video_processor", HAWOR_PROCESSOR)
    work = hand_work_dir(video, "hawor")
    config = hawor.HaWoRProcessorConfig(
        python_bin=sys.executable,
        vis_mode="off",
        run_post_steps=False,
        cleanup_intermediate=True,
    )
    hawor.process_video(
        video,
        work,
        gpu_ids=(profile_gpu(),),
        config=config,
        overwrite_output=True,
    )
    produced = work / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    if not produced.is_file():
        raise SystemExit(f"HaWoR did not write {produced}")
    dest = output / "output" / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, dest)
    if (work / "sam_boxes.npz").is_file():
        shutil.copy2(work / "sam_boxes.npz", output / "output" / "sam_boxes.npz")
    shutil.rmtree(work)
    export_viewer_hands(video, dest)
    return dest


def run_wilor(video: Path, output: Path, *, batch_size=16, force=False) -> tuple[Path, dict]:
    from pipelines.wilor.cache import lock
    with lock(output / "output" / ".hand_pipeline.lock"):
        return _run_wilor_cached(video, output, batch_size=batch_size, force=force)


def _run_wilor_cached(video, output, *, batch_size, force):
    from pipelines.sam3_hand_tracking.refine import environment_config
    from pipelines.wilor.cache import stamp, key, valid, complete
    from pipelines.wilor.export_viewer import WILOR_ROOT
    load_sam3_env()
    cfg = environment_config()
    if not cfg: raise SystemExit("SAM3 environment is required")
    if batch_size < 1: raise ValueError("batch_size must be positive")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = profile_gpu()
    dest = output / "output"; dest.mkdir(parents=True, exist_ok=True)
    code = [ROOT/"pipelines/wilor/camera_pipeline.py", ROOT/"pipelines/hand_association.py", ENV_PROFILE]
    code += [p for p in (ROOT/"pipelines/sam3_hand_tracking").glob("*.py") if not p.name.startswith("test_")]
    code += [ROOT/"pipelines/wilor/cache.py", ROOT/"pipelines/hawor/scripts/scripts_test_video/hawor_slam.py"]
    code += [ROOT/"pipelines/hawor/lib/pipeline/tools.py", ROOT/"pipelines/hawor/scripts/segmented_demo_pipeline.py"]
    camera_key = key(dict(video=stamp(video), sam={k:str(v) for k,v in cfg.items()},
        checkpoint=stamp(cfg['checkpoint']), detector=stamp(ROOT/'pipelines/hawor/weights/external/detector.pt'),
        backend_weights=[stamp(p) for p in sorted((ROOT/'pipelines/hawor/weights').rglob('*')) if p.is_file() and p.suffix in {'.pth','.pt','.ckpt'}], version=2,
        force=None), code)
    work_root = Path(os.environ.get("EGO_VIEWER_WORK_ROOT", "/data/heyuping/ego_viewer/demo"))
    work = work_root / "pipeline_cache" / (f"{camera_key}-{time.time_ns()}" if force else camera_key)
    camera_files = [dest/name for name in ('camera_slam.npz','sam_boxes.npz','tracking_validation.json','camera_pipeline_complete.json')]
    started=time.perf_counter()
    camera_hit=not force and valid(dest/'camera_cache.json',camera_key)
    if camera_hit:
        saved=json.loads((dest/"camera_pipeline_complete.json").read_text())
        work=Path(saved.get("cache_work_dir",str(work)))
        print("CACHE HIT: camera + validated SAM boxes",flush=True)
    else:
        work.mkdir(parents=True,exist_ok=True)
        print(f"Camera pipeline log: {dest/'camera_pipeline.log'}; resume cache: {work}",flush=True)
        with (dest/'camera_pipeline.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'pipelines/wilor/camera_pipeline.py'),
                '--video',str(video),'--work',str(work)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((work/'camera_pipeline_complete.json').read_text())
        shutil.copy2(result['slam'],dest/'camera_slam.npz')
        result['slam']=str(dest/'camera_slam.npz')
        result['cache_work_dir']=str(work)
        (dest/'camera_pipeline_complete.json').write_text(json.dumps(result,indent=2))
        for name in ('sam_boxes.npz','stage_timings.json','sam3_jobs.timings.json','tracking_validation.json'):
            shutil.copy2(work/name,dest/name)
        complete(dest/'camera_cache.json',camera_key,camera_files)
    camera_seconds=time.perf_counter()-started
    frames=work/'extracted_images'
    raw_key=key(dict(camera=camera_key, camera_file=stamp(dest/'camera_slam.npz'),
        boxes=stamp(dest/'sam_boxes.npz'), batch_size=batch_size,
        checkpoint=stamp(WILOR_ROOT/'pretrained_models/wilor_final.ckpt'),
        config=stamp(WILOR_ROOT/'pretrained_models/model_config.yaml'),
        input_mode='frames' if frames.is_dir() else 'video'),
        [WILOR_SCRIPT,ROOT/'pipelines/wilor/batching.py',WILOR_ROOT/'wilor/models/wilor.py',WILOR_ROOT/'wilor/datasets/vitdet_dataset.py'])
    raw_files=[dest/'raw_wilor'/name for name in ('keypoints.npz','mesh.bin','mano.npz')]
    raw_hit=valid(dest/'wilor_cache.json',raw_key)
    started=time.perf_counter()
    if raw_hit:
        print("CACHE HIT: raw WiLoR + MANO parameters",flush=True)
    else:
        cmd=[str(WILOR_PYTHON),'-u',str(WILOR_SCRIPT),'--temporal-v3','--video',str(video),
             '--output-dir',str(dest),'--slam',str(dest/'camera_slam.npz'),
             '--hand-boxes',str(dest/'sam_boxes.npz'),'--batch-size',str(batch_size)]
        if frames.is_dir():cmd += ['--frames',str(frames)]
        subprocess.check_call(cmd,env=env)
        complete(dest/'wilor_cache.json',raw_key,raw_files)
    post_key=key(dict(raw=raw_key,files=[stamp(p) for p in raw_files]),
        [ROOT/'pipelines/wilor'/name for name in ('temporal.py','postprocess.py','reprocess.py')])
    post_files=[dest/name for name in ('keypoints.npz','mesh.bin','processed_mano.npz','temporal_report.json')]
    post_hit=raw_hit and valid(dest/'temporal_cache.json',post_key)
    if raw_hit and not post_hit:
        print("Reprocessing saved MANO on CPU; no detection/SAM/SLAM/WiLoR inference",flush=True)
        subprocess.check_call([str(WILOR_PYTHON),str(ROOT/'pipelines/wilor/reprocess.py'),
                               '--output-dir',str(dest)],env=env)
    if not post_hit:complete(dest/'temporal_cache.json',post_key,post_files)
    report=json.loads((dest/'temporal_report.json').read_text())
    timing=dict(camera_pipeline_seconds=camera_seconds,wilor_and_postprocess_seconds=time.perf_counter()-started,
                camera_cache_hit=camera_hit,wilor_cache_hit=raw_hit,temporal_cache_hit=post_hit,
                work_dir=str(work),mask_source='sam3',batch_size=batch_size)
    (dest/'hand_pipeline_timings.json').write_text(json.dumps(timing,indent=2))
    return dest/'keypoints.npz',report


def export_viewer_hands(video: Path, hands: Path) -> None:
    script = ROOT.parent / "viewer" / "export_keypoints.py"
    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--hands",
            str(hands),
            "--output-dir",
            str(hands.parents[2]),
            "--video",
            str(video),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate one ego video.")
    parser.add_argument("video", type=Path, help="Input video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result directory. Defaults to <video_stem>_annotation next to the video.",
    )
    parser.add_argument(
        "--hands",
        choices=("wilor", "hawor"),
        default="wilor",
        help="wilor uses SAM boxes, the HaWoR camera trajectory, and WiLoR hands. hawor keeps HaWoR's own hands.",
    )
    parser.add_argument("--postprocess-only", action="store_true", help="Reprocess existing raw_wilor MANO on CPU, without camera or network inference.")
    parser.add_argument("--wilor-batch-size", type=int, default=16, help="Hand crops per GPU batch.")
    parser.add_argument("--force-hands", action="store_true", help="Use a fresh camera/hand cache instead of resuming.")
    parser.add_argument("--hands-only", action="store_true", help="Skip caption API and action annotations.")
    parser.add_argument("--actions-only", action="store_true", help="Skip hand reconstruction and write action labels only.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="FineBio human label CSV. Used as a reference, snapped to 0.5 s. Omit when there is no human annotation.",
    )
    args = parser.parse_args()
    if args.hands_only and args.actions_only:
        raise SystemExit("pass only one of --hands-only and --actions-only")
    if args.hands_only and args.labels is not None:
        raise SystemExit("--labels is unused with --hands-only")

    if args.postprocess_only and (args.actions_only or args.hands != 'wilor' or args.force_hands or args.labels is not None):
        raise SystemExit('--postprocess-only requires WiLoR and cannot combine with actions/force/labels')
    ensure_hawor_python()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    output = (args.output or video.with_name(video.stem + "_annotation")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        from pipelines.wilor.cache import lock
        with lock(output/"output"/".hand_pipeline.lock"):
            subprocess.check_call([str(WILOR_PYTHON),str(ROOT/'pipelines/wilor/reprocess.py'),
                               '--output-dir',str(output/'output')])
        return
    placed = place_video(video, output)
    summary = {"output": str(output), "video": str(placed), "hands_backend": args.hands}
    if not args.hands_only:
        caption = load_module("atomic_subtask_demo", CAPTION_DIR / "atomic_subtask_demo.py")
        actions = run_actions(caption, video, output, labels=args.labels)
        summary.update(annotation=str(output / "output" / "ego_action_annotation.json"),
                       n_segments=len(actions["segments"]), scene=actions.get("scene", ""))
    if args.actions_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.hands == "wilor":
        keypoints, report = run_wilor(video, output, batch_size=args.wilor_batch_size, force=args.force_hands)
        summary["keypoints"] = str(keypoints)
        summary["mesh"] = str(keypoints.with_name("mesh.bin"))
        summary["postprocess"] = report
    else:
        summary["hands"] = str(run_hands(video, output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
