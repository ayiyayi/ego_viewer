"""Local viewer for an Open-AoE style ego page.

Serves the undistorted video, ego_action_annotation.json, and pre-exported
MANO keypoints. Large media stays on /data/heyuping; this process only reads it.
"""

from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLES_PATH = ROOT / "samples.json"
EXAMPLES_ROOT = ROOT.parent / "datapipe_workflow" / "examples"
SCHEMA_PATH = ROOT / "schema" / "ego_action_annotation.schema.json"
HOST = "0.0.0.0"
PORT = int(os.environ.get("VIEWER_PORT", "8770"))


def discover_examples() -> list[dict]:
    """One sample per folder under the examples directory.

    A folder counts when it has an mp4 in input/. Annotation and hands.npz
    are picked up when the pipeline has written them.
    """
    if not EXAMPLES_ROOT.is_dir():
        return []
    found = []
    for folder in sorted(path for path in EXAMPLES_ROOT.iterdir() if path.is_dir()):
        inputs = sorted((folder / "input").glob("*.mp4"))
        if not inputs:
            continue
        video = inputs[0]
        annotation = folder / "output" / "ego_action_annotation.json"
        hands = next(folder.rglob("hands.npz"), None)
        keypoints = folder / "output" / "keypoints.npz"
        found.append({
            "id": folder.name,
            "title": video.stem,
            "root": str(folder),
            "video": str(video.relative_to(folder)),
            "annotation": (
                str(annotation.relative_to(folder))
                if annotation.is_file()
                else "output/ego_action_annotation.json"
            ),
            "hands_npz": str(hands.relative_to(folder)) if hands else None,
            "keypoints": str(keypoints.resolve()) if keypoints.is_file() else None,
            "origin": "examples",
        })
    return found


def load_samples() -> list[dict]:
    listed = json.loads(SAMPLES_PATH.read_text())["samples"]
    for sample in listed:
        sample.setdefault("origin", "samples")
    return listed + discover_examples()


def sample_by_id(sample_id: str) -> dict | None:
    for sample in load_samples():
        if sample["id"] == sample_id:
            return sample
    return None


def resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("path escapes sample root")
    path = root / relative_path
    if path.resolve() != root and root not in path.resolve().parents and not path.is_symlink():
        raise ValueError("path escapes sample root")
    return path


def num(value) -> float:
    return float(value)


def validate_annotation(data) -> list[str]:
    errors = []
    if not isinstance(data, list):
        return ["annotation must be a JSON array of segments"]
    for index, segment in enumerate(data):
        where = f"segment[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{where} must be an object")
            continue
        for key in ("id", "start_ts", "end_ts", "start_frame", "end_frame", "scene"):
            if key not in segment:
                errors.append(f"{where} missing {key}")
        legacy = segment.get("atomic_action")
        legacy_action = legacy[0] if isinstance(legacy, list) and legacy and isinstance(legacy[0], dict) else {}
        if "verb" not in segment and "verb" not in legacy_action:
            errors.append(f"{where} missing verb")
        if "object" not in segment and "object" not in legacy_action:
            errors.append(f"{where} missing object")
        if "action" not in segment and "description" not in legacy_action:
            errors.append(f"{where} missing action")
        try:
            if num(segment.get("end_ts", 0)) < num(segment.get("start_ts", 0)):
                errors.append(f"{where} end_ts is before start_ts")
        except (TypeError, ValueError):
            errors.append(f"{where} start_ts/end_ts are not numeric")
    return errors


_KEYPOINT_CACHE: dict[str, dict] = {}


def keypoints_payload(path: Path) -> dict:
    key = str(path)
    cached = _KEYPOINT_CACHE.get(key)
    if cached is not None and cached[0] == path.stat().st_mtime:
        return cached[1]
    archive = np.load(path)
    world = np.ascontiguousarray(archive["joints_world"], dtype=np.float32)
    valid = np.ascontiguousarray(archive["pred_valid"] >= 0.5, dtype=np.uint8)
    rotation = np.ascontiguousarray(archive["R_w2c"].reshape(archive["R_w2c"].shape[0], 9), dtype=np.float32)
    translation = np.ascontiguousarray(archive["t_w2c"], dtype=np.float32)
    cam_pos = np.ascontiguousarray(archive["cam_pos"], dtype=np.float32)
    cam_z = np.ascontiguousarray(archive["cam_z"], dtype=np.float32)
    cam_r = np.ascontiguousarray(archive["cam_R"].reshape(archive["cam_R"].shape[0], 9), dtype=np.float32)

    def b64(array: np.ndarray) -> str:
        import base64
        return base64.b64encode(array.tobytes()).decode("ascii")

    payload = {
        "ready": True,
        "n": int(world.shape[1]),
        "focal": float(archive["focal"]),
        "width": int(archive["width"]),
        "height": int(archive["height"]),
        "fps": float(archive["fps"]),
        "world": b64(world),
        "valid": b64(valid),
        "R": b64(rotation),
        "t": b64(translation),
        "camPos": b64(cam_pos),
        "camZ": b64(cam_z),
        "camR": b64(cam_r),
    }
    _KEYPOINT_CACHE[key] = (path.stat().st_mtime, payload)
    return payload


# moov-after-mdat files cannot start playback until the header at the end is
# read. Rebuild that header in memory (about 25KB here) and serve the file as
# if the index were already at the front, so one sequential response can play.
_FASTSTART_CACHE: dict[str, tuple] = {}


def _box_header(data: bytes | bytearray, pos: int, limit: int) -> tuple[int, int]:
    size = int.from_bytes(data[pos:pos + 4], "big")
    header = 8
    if size == 1:
        size = int.from_bytes(data[pos + 8:pos + 16], "big")
        header = 16
    elif size == 0:
        size = limit - pos
    return size, header


def _shift_chunk_offsets(moov: bytearray, delta: int) -> None:
    def walk(start: int, end: int) -> None:
        pos = start
        while pos + 8 <= end:
            size, header = _box_header(moov, pos, end)
            kind = bytes(moov[pos + 4:pos + 8])
            body = pos + header
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                walk(body, pos + size)
            elif kind == b"stco":
                count = int.from_bytes(moov[body + 4:body + 8], "big")
                for i in range(count):
                    at = body + 8 + i * 4
                    value = int.from_bytes(moov[at:at + 4], "big") + delta
                    moov[at:at + 4] = value.to_bytes(4, "big")
            elif kind == b"co64":
                count = int.from_bytes(moov[body + 4:body + 8], "big")
                for i in range(count):
                    at = body + 8 + i * 8
                    value = int.from_bytes(moov[at:at + 8], "big") + delta
                    moov[at:at + 8] = value.to_bytes(8, "big")
            pos += size
            if size < 8:
                break
    walk(0, len(moov))


def faststart_layout(path: Path):
    """Virtual bytes with moov before mdat. None when the file is already playable."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _FASTSTART_CACHE.get(str(path))
    if cached and cached[0] == key:
        return cached[1]
    atoms = []
    offset = 0
    with path.open("rb") as handle:
        while offset + 8 <= stat.st_size:
            handle.seek(offset)
            raw = handle.read(16)
            size = int.from_bytes(raw[:4], "big")
            kind = raw[4:8]
            if size == 1:
                size = int.from_bytes(raw[8:16], "big")
            elif size == 0:
                size = stat.st_size - offset
            atoms.append((offset, size, kind))
            offset += size
            if offset <= atoms[-1][0]:
                break
    moov = next((item for item in atoms if item[2] == b"moov"), None)
    mdat = next((item for item in atoms if item[2] == b"mdat"), None)
    layout = None
    if moov and mdat and moov[0] > mdat[0]:
        with path.open("rb") as handle:
            prefix = handle.read(mdat[0])
            handle.seek(moov[0])
            moved = bytearray(handle.read(moov[1]))
        _shift_chunk_offsets(moved, moov[1])
        header = prefix + bytes(moved)
        layout = {
            "header": header,
            "mdat_off": mdat[0],
            "mdat_len": mdat[1],
            "size": len(header) + mdat[1],
        }
    _FASTSTART_CACHE[str(path)] = (key, layout)
    return layout


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[viewer] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path in ("/", "/index.html"):
                self._send_file(STATIC / "v7" / "index.html")
            elif path.startswith("/static/"):
                asset = (STATIC / path[len("/static/"):]).resolve()
                if asset != STATIC and STATIC not in asset.parents:
                    self._send_json({"error": "not found"}, status=404)
                    return
                self._send_file(asset)
            elif path == "/api/samples":
                self._send_json(self._samples_payload())
            elif path == "/api/schema":
                self._send_json(json.loads(SCHEMA_PATH.read_text()))
            elif path.startswith("/api/samples/"):
                self._sample_route(path[len("/api/samples/"):])
            else:
                self._send_json({"error": "not found"}, status=404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _samples_payload(self) -> list[dict]:
        rows = []
        for sample in load_samples():
            root = Path(sample["root"])
            annotation = resolve_under(root, sample["annotation"])
            count = 0
            if annotation.is_file():
                data = json.loads(annotation.read_text())
                count = len(data) if isinstance(data, list) else 0
            keypoints = sample.get("keypoints")
            mesh = Path(keypoints).with_name("mesh.bin") if keypoints else None
            rows.append({
                "id": sample["id"],
                "title": sample["title"],
                "origin": sample.get("origin", "samples"),
                "annotation_count": count,
                "annotation_path": str(annotation),
                "hands_ready": bool(keypoints and Path(keypoints).is_file()),
                "mesh_ready": bool(mesh and mesh.is_file()),
                "video_url": f"/api/samples/{sample['id']}/video?play=1",
                "annotation_url": f"/api/samples/{sample['id']}/annotation",
                "keypoints_url": f"/api/samples/{sample['id']}/keypoints",
                "mesh_url": f"/api/samples/{sample['id']}/mesh",
            })
        return rows

    def _sample_route(self, rest: str) -> None:
        sample_id, _, kind = rest.partition("/")
        sample = sample_by_id(sample_id)
        if sample is None:
            self._send_json({"error": "unknown sample"}, status=404)
            return
        root = Path(sample["root"]).resolve()
        if kind == "annotation":
            path = resolve_under(root, sample["annotation"])
            data = json.loads(path.read_text()) if path.is_file() else []
            self._send_json({
                "segments": data,
                "errors": validate_annotation(data),
                "path": str(path),
            })
        elif kind == "video":
            video = resolve_under(root, sample["video"])
            # moov-at-front copy, when present. The source file keeps its
            # original layout for the pipelines that wrote it.
            fast = video.with_name(video.stem + ".faststart" + video.suffix)
            if fast.is_file():
                video = fast
            self._send_file(video, ranges=True)
        elif kind == "keypoints":
            keypoints = sample.get("keypoints")
            if not keypoints or not Path(keypoints).is_file():
                self._send_json({
                    "ready": False,
                    "reason": "Hand keypoints are not exported yet. Export hands.npz into the web cache first.",
                })
                return
            self._send_json(keypoints_payload(Path(keypoints)))
        elif kind == "mesh":
            keypoints = sample.get("keypoints")
            mesh = Path(keypoints).with_name("mesh.bin") if keypoints else None
            if mesh is None or not mesh.is_file():
                self._send_json({"error": "mesh not exported"}, status=404)
                return
            self._send_file(mesh)
        else:
            self._send_json({"error": "not found"}, status=404)

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ranges: bool = False) -> None:
        path = path.resolve()
        if not path.is_file():
            self._send_json({"error": f"missing {path.name}"}, status=404)
            return
        layout = faststart_layout(path) if ranges else None
        size = layout["size"] if layout else path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = 200
        # Without a front-loaded index, an open-ended "bytes=0-" would push
        # the whole file before the player can see moov. Cap that probe only.
        # Once the index is at the front, stream the file: chopping playback
        # into 1MB pieces stalled this ~16Mbit clip about twice a second.
        probe_cap = 1024 * 1024
        range_header = self.headers.get("Range") if ranges else None
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1].split(",", 1)[0]
            left, _, right = spec.partition("-")
            if left == "":
                length = int(right)
                start = max(0, size - length)
            else:
                start = int(left)
                if right:
                    end = int(right)
                elif start == 0 and layout is None:
                    end = min(size - 1, probe_cap - 1)
                else:
                    end = size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            self._write_file_range(path, layout, start, end)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _write_file_range(self, path: Path, layout, start: int, end: int) -> None:
        if layout is None:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(1024 * 256, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return
        header = layout["header"]
        header_len = len(header)
        if start < header_len:
            cut = min(end, header_len - 1)
            self.wfile.write(header[start:cut + 1])
            start = header_len
        if start > end:
            return
        with path.open("rb") as handle:
            handle.seek(layout["mdat_off"] + (start - header_len))
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(1024 * 256, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open-AoE viewer http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
