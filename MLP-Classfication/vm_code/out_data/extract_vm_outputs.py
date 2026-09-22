#!/usr/bin/env python3
"""Extract cell outputs from train_pipeline_vm.ipynb into a JSON file.

Default paths are resolved relative to this script, so it works
regardless of the current working directory:

    notebook -> ../train_pipeline_vm.ipynb
    output   -> ./vm_outputs.json  (same folder as this script)

Usage:
    python3 extract_vm_outputs.py
    python3 extract_vm_outputs.py --notebook /path/to/train_pipeline_vm.ipynb --output /path/to/vm_outputs.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def first_source_line(source: str) -> str:
    for line in source.splitlines():
        if line.strip():
            return line.strip()
    return "(empty cell)"


def normalize_output(output: dict) -> dict:
    """Keep the useful fields of one Jupyter output object."""
    normalized: dict = {"output_type": output.get("output_type")}
    if "name" in output:
        normalized["name"] = output.get("name")
    if "text" in output:
        text = output.get("text")
        normalized["text"] = "".join(text) if isinstance(text, list) else str(text)
    if "ename" in output:
        normalized["ename"] = output.get("ename", "")
        normalized["evalue"] = output.get("evalue", "")
        normalized["traceback"] = list(output.get("traceback") or [])
    if "data" in output:
        normalized["data"] = dict(output.get("data") or {})
    if "metadata" in output:
        normalized["metadata"] = dict(output.get("metadata") or {})
    if "execution_count" in output:
        normalized["execution_count"] = output.get("execution_count")
    return normalized


def extract_outputs(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    extracted_cells: list[dict] = []
    code_cells = 0
    with_outputs = 0

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else str(source)
        outputs = cell.get("outputs") or []
        if outputs:
            with_outputs += 1
        extracted_cells.append(
            {
                "cell_index": index,
                "execution_count": cell.get("execution_count"),
                "source_first_line": first_source_line(source_text),
                "source": source_text,
                "num_outputs": len(outputs),
                "outputs": [normalize_output(o) for o in outputs],
            }
        )

    return {
        "notebook": str(notebook_path),
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nbformat": notebook.get("nbformat"),
        "nbformat_minor": notebook.get("nbformat_minor"),
        "total_cells": len(cells),
        "code_cells": code_cells,
        "cells_with_outputs": with_outputs,
        "cells": extracted_cells,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_notebook = script_dir.parent / "train_pipeline_vm.ipynb"
    default_output = script_dir / "vm_outputs.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=str(default_notebook))
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    output_path = Path(args.output)
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy notebook: {notebook_path}")

    result = extract_outputs(notebook_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Notebook: {notebook_path}")
    print(f"Code cells: {result['code_cells']}, cells with outputs: {result['cells_with_outputs']}")
    print(f"Saved JSON outputs to {output_path}")


if __name__ == "__main__":
    main()
