#!/usr/bin/env python3
"""Export executed notebook outputs into a compact Markdown log."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def text_from_output(output: dict[str, Any]) -> str:
    """Convert one Jupyter output object to plain text."""
    if "text" in output:
        value = output["text"]
        return "".join(value) if isinstance(value, list) else str(value)

    if "ename" in output:
        traceback = output.get("traceback") or []
        if traceback:
            return "\n".join(str(line) for line in traceback)
        return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"

    data = output.get("data", {})
    if "text/plain" in data:
        value = data["text/plain"]
        return "".join(value) if isinstance(value, list) else str(value)

    return ""


def first_source_line(cell: dict[str, Any]) -> str:
    source = cell.get("source") or []
    text = "".join(source) if isinstance(source, list) else str(source)
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "(empty cell)"


def export_outputs(notebook_path: Path, output_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    lines: list[str] = [
        f"# Notebook Run Log - {notebook_path.name}",
        "",
        f"- Exported at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Source notebook: `{notebook_path}`",
        "",
    ]

    output_cells = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs") or []
        if not outputs:
            continue

        output_cells += 1
        execution_count = cell.get("execution_count")
        lines.extend(
            [
                f"## Cell {index}",
                "",
                f"- execution_count: `{execution_count}`",
                f"- overview/source: `{first_source_line(cell)}`",
                "",
            ]
        )

        for output_index, output in enumerate(outputs, start=1):
            text = text_from_output(output).strip()
            if not text:
                continue
            lines.extend(
                [
                    f"### Output {output_index}",
                    "",
                    "```text",
                    text,
                    "```",
                    "",
                ]
            )

    if output_cells == 0:
        lines.extend(["Notebook chưa có output nào được lưu.", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        default="MLP-Classfication/code/train_pipeline.ipynb",
        help="Path tới notebook .ipynb cần export.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="MLP-Classfication/code/RUN_LOG.md",
        help="Path file Markdown output.",
    )
    args = parser.parse_args()
    export_outputs(Path(args.notebook), Path(args.output))
    print(f"Exported notebook outputs to {args.output}")


if __name__ == "__main__":
    main()
