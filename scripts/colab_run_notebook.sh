#!/usr/bin/env bash
set -euo pipefail

# Execute the main notebook inside the synced Colab project folder.
#
# Run scripts/colab_sync_project.sh first, or set SYNC_FIRST=1 to do it here.
# Google Drive data still needs to be mounted inside the Colab session once.

SESSION="${SESSION:-unlearning}"
REMOTE_ROOT="${REMOTE_ROOT:-/content/unlearning}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-MLP-Classfication/code/train_pipeline.ipynb}"
EXECUTED_NOTEBOOK_PATH="${EXECUTED_NOTEBOOK_PATH:-MLP-Classfication/code/train_pipeline.executed.ipynb}"
RUN_LOG_PATH="${RUN_LOG_PATH:-MLP-Classfication/code/RUN_LOG.md}"
DATA_DIR="${DATA_DIR:-/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)}"
TIMEOUT="${TIMEOUT:-7200}"
SYNC_FIRST="${SYNC_FIRST:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$SYNC_FIRST" == "1" ]]; then
  "$PROJECT_ROOT/scripts/colab_sync_project.sh"
fi

RUNNER="$(mktemp /tmp/colab_run_notebook.XXXXXX.py)"
EXEC_LOG="$(mktemp /tmp/colab_run_notebook_output.XXXXXX.log)"
cat > "$RUNNER" <<PY
import os
import subprocess
import sys
import shutil
from pathlib import Path

remote_root = Path("$REMOTE_ROOT")
notebook_path = remote_root / "$NOTEBOOK_PATH"
executed_notebook_path = remote_root / "$EXECUTED_NOTEBOOK_PATH"
run_log_path = remote_root / "$RUN_LOG_PATH"
data_dir = Path("$DATA_DIR")

print("[remote] Python:", sys.version)
print("[remote] CWD target:", remote_root)
print("[remote] Notebook:", notebook_path)
print("[remote] Data dir:", data_dir)

if not remote_root.exists():
    print(
        "[remote][error] "
        f"Remote project root does not exist: {remote_root}. "
        "Run scripts/colab_sync_project.sh first."
    )
    raise SystemExit(1)
if not notebook_path.exists():
    print(f"[remote][error] Notebook not found: {notebook_path}")
    raise SystemExit(1)
if not data_dir.exists():
    print(
        "[remote][error] "
        f"Dataset path not found: {data_dir}\\n"
        "Mount Google Drive in this Colab session first, then run this script again."
    )
    raise SystemExit(1)

os.chdir(remote_root)

print("[remote] Installing notebook execution dependencies...")
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "nbformat",
    "nbclient",
    "ipykernel",
])

requirements = remote_root / "MLP-Classfication/requirements.txt"
if requirements.exists():
    print("[remote] Installing project requirements except torch...")
    filtered_requirements = Path("/tmp/unlearning_requirements_no_torch.txt")
    filtered_lines = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            filtered_lines.append(line)
            continue
        package_name = stripped.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("~=", 1)[0].strip().lower()
        if package_name in {"torch", "torchvision", "torchaudio"}:
            print(f"[remote] Skip {stripped}; use Colab runtime PyTorch/CUDA instead.")
            continue
        filtered_lines.append(line)
    filtered_requirements.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(filtered_requirements),
    ])

def print_gpu_preflight() -> None:
    print("[remote] GPU preflight:")
    print("  COLAB_GPU=", os.environ.get("COLAB_GPU"))
    print("  CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("  NVIDIA_VISIBLE_DEVICES=", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    nvidia_smi = shutil.which("nvidia-smi")
    print("  nvidia-smi=", nvidia_smi)
    if nvidia_smi:
        subprocess.run([nvidia_smi, "-L"], check=False)
    else:
        print("  nvidia-smi -L skipped: command not found")
    try:
        import torch

        print("  torch.__version__=", torch.__version__)
        print("  torch.version.cuda=", torch.version.cuda)
        print("  torch.cuda.is_available()=", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  torch.cuda.get_device_name(0)=", torch.cuda.get_device_name(0))
    except Exception as error:
        print(f"  torch import/check failed: {type(error).__name__}: {error}")

print_gpu_preflight()

import nbformat
from nbclient import NotebookClient

print("[remote] Executing notebook...")
with notebook_path.open("r", encoding="utf-8") as handle:
    notebook = nbformat.read(handle, as_version=4)

client = NotebookClient(
    notebook,
    timeout=$TIMEOUT,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": str(remote_root)}},
)
client.execute()

executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)
with executed_notebook_path.open("w", encoding="utf-8") as handle:
    nbformat.write(notebook, handle)
print("[remote] Saved executed notebook:", executed_notebook_path)

exporter = remote_root / "MLP-Classfication/code/export_notebook_outputs.py"
if exporter.exists():
    print("[remote] Exporting notebook outputs to RUN_LOG.md...")
    subprocess.check_call([
        sys.executable,
        str(exporter),
        str(executed_notebook_path),
        str(run_log_path),
    ])
    print("[remote] Saved log:", run_log_path)

print("[remote] Notebook run complete.")
PY

set +e
colab exec -s "$SESSION" -f "$RUNNER" --timeout "$TIMEOUT" 2>&1 | tee "$EXEC_LOG"
COLAB_STATUS="${PIPESTATUS[0]}"
set -e
rm -f "$RUNNER"

if grep -q "\[remote\]\[error\]" "$EXEC_LOG"; then
  rm -f "$EXEC_LOG"
  exit 1
fi

rm -f "$EXEC_LOG"
exit "$COLAB_STATUS"
