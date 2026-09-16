#!/usr/bin/env bash
set -euo pipefail

# Sync the current repository to a running Colab VM.
#
# The Colab CLI cannot attach the local cwd as a real remote filesystem. This
# script packages the local project, uploads one zip file, then extracts it into
# REMOTE_ROOT on the Colab VM.

SESSION="${SESSION:-unlearning}"
GPU="${GPU:-T4}"
REMOTE_ROOT="${REMOTE_ROOT:-/content/unlearning}"
REMOTE_ARCHIVE="${REMOTE_ARCHIVE:-/content/unlearning_project.zip}"
LOCAL_ARCHIVE="${LOCAL_ARCHIVE:-/tmp/unlearning_colab_project.zip}"
DELETE_REMOTE="${DELETE_REMOTE:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if ! colab status -s "$SESSION" >/dev/null 2>&1; then
  echo "[local] Colab session '$SESSION' not found. Creating a new $GPU session..."
  colab new -s "$SESSION" --gpu "$GPU"
fi

echo "[local] Packaging project from: $PROJECT_ROOT"
python3 - "$LOCAL_ARCHIVE" <<'PY'
import sys
import zipfile
from pathlib import Path

archive_path = Path(sys.argv[1])
root = Path.cwd()

skip_dirs = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "packet-feature-cache",
}
skip_suffixes = {
    ".pt",
    ".pth",
    ".ckpt",
    ".pcap",
    ".cap",
    ".zip",
    ".tar",
    ".gz",
}

archive_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if path.is_file() and path.suffix.lower() in skip_suffixes:
            continue
        if path.is_file():
            zipf.write(path, rel.as_posix())

print(f"[local] Wrote archive: {archive_path}")
print(f"[local] Size MB: {archive_path.stat().st_size / (1024 * 1024):.2f}")
PY

echo "[local] Uploading archive to Colab: $REMOTE_ARCHIVE"
colab upload -s "$SESSION" "$LOCAL_ARCHIVE" "$REMOTE_ARCHIVE"

BOOTSTRAP="$(mktemp /tmp/colab_extract_project.XXXXXX.py)"
cat > "$BOOTSTRAP" <<PY
import shutil
import zipfile
from pathlib import Path

remote_root = Path("$REMOTE_ROOT")
remote_archive = Path("$REMOTE_ARCHIVE")
delete_remote = "$DELETE_REMOTE" == "1"

if not remote_archive.exists():
    raise FileNotFoundError(f"Archive not found on Colab VM: {remote_archive}")

if remote_root.exists() and delete_remote:
    shutil.rmtree(remote_root)
remote_root.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(remote_archive, "r") as zipf:
    zipf.extractall(remote_root)

print(f"[remote] Extracted project to: {remote_root}")
print("[remote] Top-level files:")
for path in sorted(remote_root.iterdir()):
    print(" -", path)
PY

echo "[local] Extracting project on Colab..."
colab exec -s "$SESSION" -f "$BOOTSTRAP" --timeout 300
rm -f "$BOOTSTRAP"

echo "[local] Sync complete."
echo "[local] Remote project root: $REMOTE_ROOT"
