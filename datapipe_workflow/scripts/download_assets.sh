#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$ROOT_DIR/assets/assets.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Asset manifest not found: $MANIFEST" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY_CHECK_YAML' >/dev/null 2>&1
import yaml
PY_CHECK_YAML
then
  echo "PyYAML is required. Install project dependencies first:" >&2
  echo "  $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY_CHECK_GDOWN' >/dev/null 2>&1
import gdown
PY_CHECK_GDOWN
then
  echo "gdown is required for Google Drive assets; installing into current Python env." >&2
  "$PYTHON_BIN" -m pip install gdown
fi

export ROOT_DIR MANIFEST
"$PYTHON_BIN" - <<'PY_DOWNLOAD_ASSETS'
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

root = Path(os.environ["ROOT_DIR"])
manifest_path = Path(os.environ["MANIFEST"])
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def run(cmd: list[str]) -> None:
    print("[download]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download_url(url: str, target: Path) -> None:
    run(["curl", "-L", "--fail", "--retry", "3", "--continue-at", "-", "-o", str(target), url])


def download_gdrive(file_id: str, target: Path) -> None:
    run([sys.executable, "-m", "gdown", "--fuzzy", f"https://drive.google.com/file/d/{file_id}/view", "-O", str(target)])

for asset in manifest.get("assets", []):
    target = root / asset["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {asset['name']}: {target}")
        continue
    print(f"[asset] {asset['name']} -> {target}")
    if asset.get("url"):
        download_url(asset["url"], target)
    elif asset.get("gdrive_id"):
        download_gdrive(asset["gdrive_id"], target)
    else:
        raise SystemExit(f"No download source for asset: {asset['name']}")

missing_manual = []
for asset in manifest.get("manual_assets", []):
    target = root / asset["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[manual-ok] {asset['name']}: {target}")
    else:
        missing_manual.append(asset)

if missing_manual:
    print("\nManual assets still required:")
    for asset in missing_manual:
        print(f"- {asset['name']}")
        print(f"  target: {asset['target']}")
        print(f"  source: {asset.get('source_page', '')}")
        print(f"  note: {asset.get('instructions', '')}")
    raise SystemExit(2)

print("[done] all downloadable and manual assets are present")
PY_DOWNLOAD_ASSETS
