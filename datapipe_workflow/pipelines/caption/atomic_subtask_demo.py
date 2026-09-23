#!/usr/bin/env python3
"""One-video demo: 16s clips → contact sheets → Gemini → stitched atomic intervals.

Each sheet is labeled from 0s for that clip. Offsets are added only when
joining clips back onto the full video. Sequential, no agent, no thread pool.

Example, from the datapipe_workflow directory:
  python pipelines/caption/atomic_subtask_demo.py \\
    --video examples/caption_93009/input/93009_1min.mp4 \\
    --output examples/caption_93009/output
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "atomic_subtask_clip.txt"
DEFAULT_SCENE_PROMPT = HERE / "prompts" / "scene.txt"
SCENE_FRAMES = 5
DEFAULT_CONFIGS = (
    HERE / "api_config.json",
    Path("/data-hyp/Ego_pipe/benchmark/api_config.json"),
)

FPS = 2.0
LONG_SIDE = 224
TILES_PER_SHEET = 20
COLUMNS = 5
CLIP_SEC = 16.0
ALLOWED_VERBS = (
    "Assembles", "Attaches", "Blow-dries", "Braids", "Brushes", "Buttons", "Carries", "Catches",
    "Clips", "Closes", "Connects", "Cracks", "Crumples", "Cuts", "Detaches", "Dips", "Draws",
    "Dries", "Fills", "Flips", "Fluffs", "Folds", "Hands over", "Hangs", "Idle", "Inserts",
    "Irons", "Kneads", "Knits", "Moves", "Opens", "Pats", "Peels", "Picks up", "Places", "Plugs",
    "Pours", "Presses", "Pulls", "Pushes", "Rolls", "Rotates", "Scrapes", "Scoops", "Screws",
    "Seals", "Shakes", "Sprays", "Squeezes", "Stacks", "Stirs", "Sweeps", "Takes off", "Tears",
    "Threads", "Throws", "Ties", "Turns off", "Turns on", "Unclips", "Unfolds", "Unplugs",
    "Unrolls", "Unscrews", "Unties", "Unzips", "Walks", "Washes", "Wipes", "Wraps", "Wrings",
    "Writes", "Zips",
)


def _pop_proxy():
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


def load_api_config(path):
    if path:
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"api config not found: {path}")
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise SystemExit(f"api config must be a JSON object: {path}")
        return cfg
    for candidate in DEFAULT_CONFIGS:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def probe_fps(video):
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
    ).strip()
    rate = out.split(",")[0].strip()
    if "/" in rate:
        num, den = rate.split("/", 1)
        den = float(den)
        if den == 0:
            raise RuntimeError(f"invalid frame rate for {video}: {rate}")
        return float(num) / den
    return float(rate)


def probe_duration(video):
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out.split(",")[0])


def probe_size(video):
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
    ).strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def tile_size(src_w, src_h, long_side=LONG_SIDE):
    if src_w >= src_h:
        return long_side, max(1, round(src_h * long_side / src_w))
    return max(1, round(src_w * long_side / src_h)), long_side


def font():
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, 14)
        except OSError:
            pass
    return ImageFont.load_default()


def clip_windows(duration, clip_sec):
    t = 0.0
    while t < duration - 1e-6:
        end = min(t + clip_sec, duration)
        if end > t:
            yield t, end
        t = end


def extract_clip_frames(video, start, end, tw, th):
    """Decode one clip. Returned times are local to the clip (first tile = 0.0s)."""
    dur = max(0.0, end - start)
    vf = f"fps={FPS:g},scale={tw}:{th}"
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-t",
            f"{dur:.3f}",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    nbytes = tw * th * 3
    frames = []
    try:
        while (buf := proc.stdout.read(nbytes)) and len(buf) == nbytes:
            frames.append(Image.frombytes("RGB", (tw, th), buf))
    finally:
        proc.stdout.close()
        proc.wait()
    return frames


def save_sheets(frames, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("sheet_*.jpg"):
        old.unlink()
    if not frames:
        return []
    tw, th = frames[0].size
    face = font()
    paths = []
    for sheet_i in range(0, len(frames), TILES_PER_SHEET):
        batch = frames[sheet_i : sheet_i + TILES_PER_SHEET]
        rows = (len(batch) + COLUMNS - 1) // COLUMNS
        sheet = Image.new("RGB", (tw * COLUMNS, th * rows), "black")
        draw = ImageDraw.Draw(sheet)
        for j, im in enumerate(batch):
            x = (j % COLUMNS) * tw
            y = (j // COLUMNS) * th
            sheet.paste(im, (x, y))
            # Clip-local time: this sheet batch starts at 0s for the clip.
            label = f"{(sheet_i + j) / FPS:.1f}s"
            draw.rectangle((x, y, x + 58, y + 16), fill="black")
            draw.text((x + 3, y + 1), label, fill="white", font=face)
        path = dest / f"sheet_{sheet_i // TILES_PER_SHEET:04d}.jpg"
        sheet.save(path, quality=90)
        paths.append(str(path))
    return paths


def load_prompt(path, duration, sheet_count, instruction=""):
    text = Path(path).read_text(encoding="utf-8").strip()
    text = text.replace("{video_duration_seconds}", f"{duration:.3f}")
    text = text.replace("{sheet_count}", str(sheet_count))
    extra = [
        f"clip_duration_seconds: {duration:.3f}",
        f"contact_sheet_count: {sheet_count}",
        "All tile timestamps start at 0.0s for this clip.",
    ]
    if instruction:
        extra.insert(0, f"Episode instruction: {instruction}")
    return text + "\n" + "\n".join(extra) + "\n"


def sheet_messages(prompt, sheets):
    content = [{"type": "text", "text": prompt}]
    for sheet in sheets:
        data = base64.b64encode(Path(sheet).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return [{"role": "user", "content": content}]


def _loads(raw):
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            return json.loads(match.group(0), strict=False)
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if match:
            return json.loads(match.group(0), strict=False)
        raise


def split_action(action):
    text = (action or "").strip()
    for verb in sorted(ALLOWED_VERBS, key=len, reverse=True):
        if text.startswith(verb):
            return verb, text[len(verb):].strip(" .")
    return "", ""


def segment_record(start, end, action, verb="", object_=""):
    action = (action or "").strip()
    verb = (verb or "").strip()
    object_ = (object_ or "").strip()
    if not verb or not object_:
        guessed_verb, guessed_object = split_action(action)
        verb = verb or guessed_verb
        object_ = object_ or guessed_object
    return {
        "start_sec": round(float(start), 3),
        "end_sec": round(float(end), 3),
        "verb": verb,
        "object": object_,
        "action": action,
    }


def idle_segment(duration):
    return [segment_record(0.0, duration, "Idle.", verb="Idle", object_="")]


def parse_segments(text):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    obj = _loads(raw)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = obj.get("segments", [obj] if "start_sec" in obj or "start" in obj else [])
    else:
        items = []
    segments = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start_sec", item.get("start", 0)))
        end = float(item.get("end_sec", item.get("end", 0)))
        action = str(item.get("action") or item.get("subtask") or item.get("action_text") or "").strip()
        verb = str(item.get("verb") or "").strip()
        object_ = str(item.get("object") or "").strip()
        if end > start and (action or verb):
            segments.append(segment_record(start, end, action, verb, object_))
    segments.sort(key=lambda s: (s["start_sec"], s["end_sec"]))
    return segments


def stitch(segments, duration):
    duration = float(duration)
    segs = [s for s in segments if s["end_sec"] > s["start_sec"]]
    if duration <= 0:
        return segs
    if not segs:
        return idle_segment(duration)
    clipped = []
    for seg in segs:
        start = min(max(seg["start_sec"], 0.0), duration)
        end = min(max(seg["end_sec"], 0.0), duration)
        if end > start:
            clipped.append(segment_record(start, end, seg["action"], seg.get("verb", ""), seg.get("object", "")))
    if not clipped:
        return idle_segment(duration)
    clipped.sort(key=lambda s: (s["start_sec"], s["end_sec"]))
    merged = [dict(clipped[0])]
    for seg in clipped[1:]:
        prev = merged[-1]
        if seg["start_sec"] < prev["end_sec"]:
            if seg["end_sec"] <= prev["end_sec"]:
                continue
            seg = segment_record(prev["end_sec"], seg["end_sec"], seg["action"], seg.get("verb", ""), seg.get("object", ""))
            if seg["end_sec"] <= seg["start_sec"]:
                continue
        merged.append(seg)
    merged[0]["start_sec"] = 0.0
    for i in range(len(merged) - 1):
        merged[i]["end_sec"] = merged[i + 1]["start_sec"]
    merged[-1]["end_sec"] = duration
    out = []
    for seg in merged:
        if seg["end_sec"] > seg["start_sec"] + 1e-6:
            out.append(segment_record(seg["start_sec"], seg["end_sec"], seg["action"], seg.get("verb", ""), seg.get("object", "")))
    if not out:
        return idle_segment(duration)
    compact = [out[0]]
    for seg in out[1:]:
        if seg["action"] == compact[-1]["action"] and seg["verb"] == compact[-1]["verb"] and seg["object"] == compact[-1]["object"]:
            compact[-1]["end_sec"] = seg["end_sec"]
        else:
            compact.append(seg)
    compact[0]["start_sec"] = 0.0
    compact[-1]["end_sec"] = round(duration, 3)
    return compact


def shift_segments(segments, offset):
    shifted = []
    for seg in segments:
        shifted.append(
            segment_record(
                seg["start_sec"] + offset,
                seg["end_sec"] + offset,
                seg["action"],
                seg.get("verb", ""),
                seg.get("object", ""),
            )
        )
    return shifted


def scene_sample_times(duration, count):
    duration = max(float(duration), 0.0)
    count = max(int(count), 1)
    if duration <= 0:
        return [0.0] * count
    span = max(duration - 0.05, 0.0)
    return [span * (i + 0.5) / count for i in range(count)]


def extract_scene_frames(video, duration, dest, count=SCENE_FRAMES):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tw, th = tile_size(*probe_size(video), long_side=448)
    paths = []
    for index, timestamp in enumerate(scene_sample_times(duration, count)):
        path = dest / f"frame_{index:02d}.jpg"
        subprocess.check_call(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={tw}:{th}",
                str(path),
            ]
        )
        paths.append(path)
    return paths


def parse_scene(text):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    obj = _loads(raw)
    if isinstance(obj, dict):
        scene = str(obj.get("scene") or "").strip()
    elif isinstance(obj, str):
        scene = obj.strip()
    else:
        scene = ""
    if not scene:
        raise ValueError(f"scene response missing a place name: {text!r}")
    return scene


def recognize_scene(client, model, frame_paths, prompt_file, temperature):
    prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    completion = client.chat.completions.create(
        model=model,
        messages=sheet_messages(prompt, frame_paths),
        temperature=temperature,
    )
    return parse_scene(completion.choices[0].message.content)


def ask_gemini(client, model, messages, temperature):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return parse_segments(completion.choices[0].message.content)


def annotation_rows(segments):
    """Webpage annotation. No atomic_action, hand, description, confidence, or bbox."""
    rows = []
    for seg in segments:
        rows.append(
            {
                "id": seg["id"],
                "start_ts": seg["start_sec"],
                "end_ts": seg["end_sec"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "scene": seg.get("scene", ""),
                "verb": seg.get("verb", ""),
                "object": seg.get("object", ""),
                "action": seg.get("action", ""),
            }
        )
    return rows


def assign_ids_and_frames(segments, fps):
    numbered = []
    for index, seg in enumerate(segments, start=1):
        row = dict(seg)
        row["id"] = index
        row["start_frame"] = int(round(float(seg["start_sec"]) * fps))
        row["end_frame"] = int(round(float(seg["end_sec"]) * fps))
        if row["end_frame"] < row["start_frame"]:
            row["end_frame"] = row["start_frame"]
        numbered.append(row)
    return numbered


def annotate_video(video, out_dir, client, model, prompt_file, clip_sec, temperature, instruction, scene_prompt_file=DEFAULT_SCENE_PROMPT):
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video)
    fps = probe_fps(video)
    scene_paths = extract_scene_frames(video, duration, out_dir / "scene_frames", SCENE_FRAMES)
    scene = recognize_scene(client, model, scene_paths, scene_prompt_file, temperature)
    tw, th = tile_size(*probe_size(video))
    windows = list(clip_windows(duration, clip_sec))
    clip_rows = []
    shifted_all = []

    print(
        json.dumps(
            {
                "video": str(video),
                "duration_sec": round(duration, 3),
                "fps": fps,
                "scene": scene,
                "n_clips": len(windows),
                "clip_sec": clip_sec,
                "model": model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for i, (t0, t1) in enumerate(windows):
        sheet_dir = out_dir / "sheets" / f"clip_{i:04d}"
        frames = extract_clip_frames(video, t0, t1, tw, th)
        sheets = save_sheets(frames, sheet_dir)
        local_dur = t1 - t0
        prompt = load_prompt(prompt_file, local_dur, len(sheets), instruction=instruction)
        local = ask_gemini(client, model, sheet_messages(prompt, sheets), temperature)
        local = stitch(local, local_dur)
        shifted = shift_segments(local, t0)
        shifted_all.extend(shifted)
        row = {
            "clip_index": i,
            "offset_sec": round(t0, 3),
            "clip_end_sec": round(t1, 3),
            "clip_duration_sec": round(local_dur, 3),
            "n_sheets": len(sheets),
            "local_segments": local,
            "shifted_segments": shifted,
        }
        clip_rows.append(row)
        print(
            json.dumps(
                {
                    "clip": i + 1,
                    "of": len(windows),
                    "offset_sec": row["offset_sec"],
                    "n_local": len(local),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    segments = assign_ids_and_frames(stitch(shifted_all, duration), fps)
    for seg in segments:
        seg["scene"] = scene
    result = {
        "video": str(video),
        "duration_sec": round(duration, 3),
        "fps": fps,
        "scene": scene,
        "clip_sec": clip_sec,
        "model": model,
        "n_clips": len(windows),
        "segments": segments,
    }
    annotation = annotation_rows(segments)
    result["annotation"] = annotation
    (out_dir / "segments.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "clips.json").write_text(json.dumps(clip_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "ego_action_annotation.json").write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="16s-clip Gemini atomic subtask demo.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--scene-prompt-file", type=Path, default=DEFAULT_SCENE_PROMPT)
    parser.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--instruction", default="", help="Optional task text shown to Gemini.")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if args.clip_sec <= 0:
        raise SystemExit("--clip-sec must be > 0")

    cfg = load_api_config(args.config)
    base_url = args.base_url or cfg.get("base_url") or cfg.get("url") or os.environ.get("OPENAI_BASE_URL")
    api_key = (
        args.api_key
        or cfg.get("api_key")
        or cfg.get("key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    model = args.model or cfg.get("model") or "gemini-3.6-flash"
    if not base_url or not api_key or api_key == "<YOUR_API_KEY>":
        raise SystemExit("set base_url and api_key via --config, --base-url/--api-key, or env")

    _pop_proxy()
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    result = annotate_video(
        args.video,
        args.output,
        client,
        model,
        args.prompt_file,
        args.clip_sec,
        args.temperature,
        args.instruction,
        args.scene_prompt_file,
    )
    print(
        json.dumps(
            {
                "wrote": str(Path(args.output) / "ego_action_annotation.json"),
                "n_segments": len(result["segments"]),
                "scene": result.get("scene", ""),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
