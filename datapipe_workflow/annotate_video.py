#!/usr/bin/env python3
"""Annotate one ego video: actions, scene, hands, and camera.

Only the video path is required. API settings come from
pipelines/caption/api_config.json. HaWoR uses the Python in
env_profiles/default.yaml.

From the datapipe_workflow directory:

  python annotate_video.py /path/to/video.mp4

The result directory is <video_stem>_annotation next to the video, in the
layout the viewer discovers under datapipe_workflow/examples:

  input/<video>.mp4
  output/ego_action_annotation.json
  output/ego_process/ego_hands_reconstruction/hands.npz
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTION_DIR = ROOT / "pipelines" / "caption"
HAWOR_PROCESSOR = ROOT / "pipelines" / "hawor" / "scripts" / "hawor_video_processor.py"
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


def run_actions(caption, video: Path, output: Path) -> dict:
    work = output / "_caption"
    if work.exists():
        shutil.rmtree(work)
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
    )
    anno_dir = output / "output"
    anno_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "ego_action_annotation.json", anno_dir / "ego_action_annotation.json")
    shutil.rmtree(work)
    return result


def run_hands(video: Path, output: Path) -> Path:
    hawor = load_module("hawor_video_processor", HAWOR_PROCESSOR)
    work = output / "_hawor"
    if work.exists():
        shutil.rmtree(work)
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
    shutil.rmtree(work)
    export_viewer_hands(video, dest)
    return dest


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
    args = parser.parse_args()

    ensure_hawor_python()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    output = (args.output or video.with_name(video.stem + "_annotation")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    caption = load_module("atomic_subtask_demo", CAPTION_DIR / "atomic_subtask_demo.py")
    placed = place_video(video, output)
    actions = run_actions(caption, video, output)
    hands = run_hands(video, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "video": str(placed),
                "annotation": str(output / "output" / "ego_action_annotation.json"),
                "hands": str(hands),
                "n_segments": len(actions["segments"]),
                "scene": actions.get("scene", ""),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
