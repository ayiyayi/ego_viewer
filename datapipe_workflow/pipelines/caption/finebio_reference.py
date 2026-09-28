"""FineBio human labels as a reference for the 2 fps action annotation.

The text file starts with coarse task spans. Fine-grained rows begin where the
clock drops back near zero. Gemini sees only the fine-grained rows, after
simultaneous hands have been reduced to one main action and times have been
snapped onto the 2 fps grid.
"""

from __future__ import annotations

import csv
from pathlib import Path

HANDS = {"left_hand", "right_hand"}
# Higher means this verb is the manipulation, not a supporting button press.
VERB_RANK = {
    "close": 80,
    "open": 80,
    "insert": 70,
    "eject": 70,
    "detach": 60,
    "take": 50,
    "put": 50,
    "press": 20,
    "release": 10,
}
STEP = 0.5


def load_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fine_rows(rows: list[dict]) -> list[dict]:
    """Rows after the coarse prefix, detected by the clock dropping backward."""
    fine = []
    started = False
    previous_end = 0.0
    for row in rows:
        start = float(row["start_sec"])
        if not started and start + 1.0 < previous_end:
            started = True
        if started and (row.get("verb") or "").strip():
            fine.append(row)
        previous_end = float(row["end_sec"])
    if not fine:
        raise ValueError("no fine-grained rows after the coarse prefix")
    return fine


def _phrase(row: dict) -> str:
    verb = (row.get("verb") or "").strip().replace("_", " ")
    obj = (row.get("manipulated_object") or "").strip().replace("_", " ")
    affected = (row.get("affected_object") or "").strip().replace("_", " ")
    if verb == "hand over":
        return f"hand over {obj}".strip()
    if affected and affected not in {"left hand", "right hand"}:
        return f"{verb} {obj} into {affected}".strip()
    return f"{verb} {obj}".strip()


def choose_main(rows: list[dict]) -> dict:
    """One action for a shared interval. A handoff stays one transfer."""
    if len(rows) == 1:
        return dict(rows[0])
    objects = {(row.get("manipulated_object") or "").strip() for row in rows}
    verbs = {(row.get("verb") or "").strip() for row in rows}
    if verbs <= {"take", "put"} and len(objects) == 1:
        if any((row.get("affected_object") or "").strip() in HANDS for row in rows):
            kept = dict(next(row for row in rows if row["verb"] == "put") if "put" in verbs else rows[0])
            kept["verb"] = "hand over"
            kept["affected_object"] = ""
            kept["hand_side"] = "both"
            return kept
    if len({((row.get("verb") or "").strip(), (row.get("manipulated_object") or "").strip()) for row in rows}) == 1:
        return dict(rows[0])
    external = [
        row for row in rows
        if (row.get("affected_object") or "").strip() not in HANDS
    ]
    pool = external or list(rows)
    pool.sort(key=lambda row: (VERB_RANK.get((row.get("verb") or "").strip(), 0), (row.get("hand_side") or "") == "right"))
    return dict(pool[-1])


def snap_time(value: float, step: float = STEP) -> float:
    return round(round(float(value) / step) * step, 3)


def snap_interval(start: float, end: float, step: float = STEP) -> tuple[float, float]:
    snapped_start = snap_time(start, step)
    snapped_end = snap_time(end, step)
    if snapped_end <= snapped_start:
        snapped_end = round(snapped_start + step, 3)
    return snapped_start, snapped_end


def collapse_reference(rows: list[dict], duration: float, step: float = STEP) -> list[dict]:
    """Main action per shared interval, then snap, sort, and drop collisions."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in fine_rows(rows):
        groups.setdefault((row["start_sec"], row["end_sec"]), []).append(row)
    chosen = []
    for group in groups.values():
        row = choose_main(group)
        start, end = snap_interval(float(row["start_sec"]), float(row["end_sec"]), step)
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if end <= start:
            continue
        chosen.append({
            "start_sec": start,
            "end_sec": end,
            "text": _phrase(row),
            "verb": (row.get("verb") or "").strip(),
            "object": (row.get("manipulated_object") or "").strip().replace("_", " "),
        })
    chosen.sort(key=lambda item: (item["start_sec"], item["end_sec"], -VERB_RANK.get(item["verb"], 0)))
    kept = []
    for item in chosen:
        if kept and item["start_sec"] < kept[-1]["end_sec"]:
            if item["end_sec"] <= kept[-1]["end_sec"]:
                continue
            item = dict(item)
            item["start_sec"] = kept[-1]["end_sec"]
            if item["end_sec"] <= item["start_sec"]:
                continue
        kept.append(item)
    return kept


def reference_text(items: list[dict], start: float, end: float) -> str:
    """Clip-local reference. Empty when this clip has no human label."""
    lines = []
    for item in items:
        if item["end_sec"] <= start or item["start_sec"] >= end:
            continue
        local_start = max(item["start_sec"], start) - start
        local_end = min(item["end_sec"], end) - start
        lines.append(f"- {local_start:.1f}-{local_end:.1f}s {item['text']}")
    if not lines:
        return ""
    return (
        "Human fine-grained labels for this clip, in seconds from the start of this clip. "
        "The list can miss atomic actions, and some descriptions may be wrong. "
        "Use it only as a reference. If the sheets show an action the list skips, add that action.\n"
        + "\n".join(lines)
    )


def quantize_segments(segments: list[dict], duration: float, step: float = STEP) -> list[dict]:
    """Snap model times onto the 2 fps grid and restore time order."""
    from pipelines.caption.atomic_subtask_demo import segment_record

    snapped = []
    for seg in segments:
        start, end = snap_interval(seg["start_sec"], seg["end_sec"], step)
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if end <= start:
            continue
        snapped.append(segment_record(start, end, seg.get("action", ""), seg.get("verb", ""), seg.get("object", "")))
    snapped.sort(key=lambda seg: (seg["start_sec"], seg["end_sec"]))
    return snapped


def annotate_finebio(video: Path, labels: Path, out_dir: Path, config: Path) -> dict:
    """Run Gemini with the fine-grained labels as a reference and write the viewer JSON."""
    import os

    from pipelines.caption.atomic_subtask_demo import (
        DEFAULT_PROMPT,
        DEFAULT_SCENE_PROMPT,
        _pop_proxy,
        annotate_video,
        load_api_config,
    )

    video = Path(video)
    out_dir = Path(out_dir)
    cfg = load_api_config(config)
    base_url = cfg.get("base_url") or cfg.get("url")
    api_key = cfg.get("api_key") or cfg.get("key")
    model = cfg.get("model") or "gemini-3.6-flash"
    if not base_url or not api_key:
        raise SystemExit("caption API config is missing base_url or api_key")
    rows = load_rows(labels)
    # Duration is filled once the video is probed inside annotate_video. The
    # reference is clipped per window, so a generous upper bound is enough here.
    reference = collapse_reference(rows, duration=1e9)

    def reference_for_clip(start, end):
        return reference_text(reference, start, end)

    _pop_proxy()
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    return annotate_video(
        video,
        out_dir,
        client,
        model,
        DEFAULT_PROMPT,
        20.0,
        0.1,
        "",
        DEFAULT_SCENE_PROMPT,
        reference_for_clip=reference_for_clip,
        quantize_step=STEP,
    )


def assert_time_order(segments: list[dict]) -> None:
    previous = None
    for seg in segments:
        if seg["end_sec"] <= seg["start_sec"]:
            raise ValueError(f"segment ends before it starts: {seg}")
        if previous is not None and seg["start_sec"] + 1e-3 < previous:
            raise ValueError(f"segments are out of order at {seg['start_sec']}")
        previous = seg["start_sec"]
