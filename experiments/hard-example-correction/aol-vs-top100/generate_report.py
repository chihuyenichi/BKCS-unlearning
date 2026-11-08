#!/usr/bin/env python3
"""Create a compact Excel report from hard-example correction outputs.

The script reads ``summary.json`` produced by ``pipeline.py`` / ``pipeline.ipynb``
and writes an XLSX workbook with two small sheets:

* ``Summary``: one row per method and nine decision-relevant columns.
* ``Runs``: one row per repeat and nine columns for reproducibility.

It is designed for Kaggle. With the default arguments, run it after the training
pipeline finishes:

    python generate_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_INPUT = Path(
    "/kaggle/working/linh-san-binary-hard-example/hard_example_correction"
)
DEFAULT_REPORT_NAME = "hard_example_report.xlsx"
MAX_REPORT_COLUMNS = 9

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "uniform_finetune": "Fine-tune đồng đều",
    "hard_weighted_ce": "Weighted CE",
    "hard_weighted_center": "Weighted CE + Center",
    "hard_weighted_mkmmd": "Weighted CE + MK-MMD",
}

SUMMARY_HEADERS = (
    "Phương pháp",
    "Số lần chạy",
    "Sửa lỗi (%)",
    "Giữ mẫu đúng (%)",
    "Lỗi mới",
    "Lỗi train còn lại",
    "Test BAcc (%)",
    "Δ BAcc (điểm %)",
    "Thời gian (s)",
)

RUN_HEADERS = (
    "Phương pháp",
    "Lần chạy",
    "Sửa lỗi (%)",
    "Giữ mẫu đúng (%)",
    "Lỗi mới",
    "Lỗi train còn lại",
    "Val BAcc (%)",
    "Test BAcc (%)",
    "Thời gian (s)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo báo cáo Excel gọn từ output hard-example correction."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "summary.json hoặc thư mục chứa summary.json. "
            f"Mặc định: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Đường dẫn XLSX đầu ra. Mặc định đặt hard_example_report.xlsx "
            "cạnh summary.json."
        ),
    )
    return parser.parse_args()


def configure_console() -> None:
    """Keep Vietnamese CLI messages readable on Windows and Kaggle terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON gốc phải là object: {path}")
    return payload


def is_compatible_summary(payload: dict[str, Any]) -> bool:
    rows = payload.get("benchmark_rows")
    experiments = payload.get("experiments")
    return bool(
        (isinstance(rows, list) and rows)
        or (isinstance(experiments, list) and experiments)
    )


def discover_summary(input_path: Path) -> tuple[Path, dict[str, Any]]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        direct = [
            path / "summary.json",
            path / "hard_example_correction" / "summary.json",
        ]
        candidates = [candidate for candidate in direct if candidate.is_file()]
        if not candidates:
            candidates = sorted(path.rglob("summary.json"))
    else:
        raise FileNotFoundError(f"Không tìm thấy input: {path}")

    compatible: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        payload = load_json(candidate)
        if is_compatible_summary(payload):
            compatible.append((candidate, payload))

    if not compatible:
        raise ValueError(f"Không tìm thấy summary.json tương thích bên dưới: {path}")
    if len(compatible) > 1:
        options = "\n  - ".join(str(item[0]) for item in compatible)
        raise ValueError(
            "Tìm thấy nhiều summary.json. Hãy truyền đúng file cần báo cáo:\n"
            f"  - {options}"
        )
    return compatible[0]


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def required_number(value: Any, field: str, row_number: int) -> float:
    number = finite_number(value)
    if number is None:
        raise ValueError(f"benchmark row {row_number}: thiếu số hợp lệ cho '{field}'")
    return number


def optional_rate(value: Any, field: str, row_number: int) -> float | None:
    number = finite_number(value)
    if number is None:
        return None
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"benchmark row {row_number}: '{field}' phải nằm trong [0, 1]"
        )
    return number


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def rows_from_experiments(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiments = summary.get("experiments", [])
    if not isinstance(experiments, list):
        return rows
    for experiment in experiments:
        if not isinstance(experiment, dict):
            continue
        correction = experiment.get("after_correction", {})
        rows.append(
            {
                "mode": experiment.get("mode"),
                "repeat": experiment.get("repeat"),
                "initial_hard_count": correction.get("initial_hard_count"),
                "correction_rate": correction.get("correction_rate"),
                "retention_rate": correction.get("retention_rate"),
                "new_error_count": correction.get("new_error_count"),
                "current_hard_count": correction.get("current_hard_count"),
                "validation_balanced_accuracy": nested(
                    experiment, "after_validation", "full", "balanced_accuracy"
                ),
                "test_balanced_accuracy": nested(
                    experiment, "after_test", "full", "balanced_accuracy"
                ),
                "runtime_seconds": experiment.get("runtime_seconds"),
            }
        )
    return rows


def normalize_runs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = summary.get("benchmark_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raw_rows = rows_from_experiments(summary)
    if not raw_rows:
        raise ValueError("summary.json không có benchmark row hoặc experiment hợp lệ")

    normalized: list[dict[str, Any]] = []
    seen_runs: set[tuple[str, int]] = set()
    for row_number, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark row {row_number} không phải object")
        mode = str(raw.get("mode", "")).strip()
        if not mode:
            raise ValueError(f"benchmark row {row_number}: thiếu 'mode'")
        repeat = required_number(raw.get("repeat"), "repeat", row_number)
        if repeat < 0 or not repeat.is_integer():
            raise ValueError(
                f"benchmark row {row_number}: 'repeat' phải là số nguyên không âm"
            )

        run_key = (mode, int(repeat))
        if run_key in seen_runs:
            raise ValueError(
                f"benchmark row {row_number}: trùng mode/repeat {run_key}"
            )
        seen_runs.add(run_key)

        counts: dict[str, int] = {}
        for field in (
            "initial_hard_count",
            "new_error_count",
            "current_hard_count",
        ):
            count = required_number(raw.get(field), field, row_number)
            if count < 0 or not count.is_integer():
                raise ValueError(
                    f"benchmark row {row_number}: '{field}' phải là số nguyên không âm"
                )
            counts[field] = int(count)

        correction_rate = optional_rate(
            raw.get("correction_rate"), "correction_rate", row_number
        )
        retention_rate = optional_rate(
            raw.get("retention_rate"), "retention_rate", row_number
        )
        if counts["initial_hard_count"] > 0 and correction_rate is None:
            raise ValueError(
                f"benchmark row {row_number}: thiếu correction_rate dù có hard examples"
            )
        if correction_rate is not None:
            expected_current = (
                counts["initial_hard_count"] * (1.0 - correction_rate)
                + counts["new_error_count"]
            )
            if abs(counts["current_hard_count"] - expected_current) > 1e-6:
                raise ValueError(
                    f"benchmark row {row_number}: current_hard_count không khớp "
                    "initial_hard_count, correction_rate và new_error_count"
                )

        validation_bacc = optional_rate(
            raw.get("validation_balanced_accuracy"),
            "validation_balanced_accuracy",
            row_number,
        )
        test_bacc = optional_rate(
            raw.get("test_balanced_accuracy"),
            "test_balanced_accuracy",
            row_number,
        )
        if validation_bacc is None or test_bacc is None:
            raise ValueError(
                f"benchmark row {row_number}: thiếu balanced accuracy hợp lệ"
            )
        runtime = required_number(
            raw.get("runtime_seconds"), "runtime_seconds", row_number
        )
        if runtime < 0:
            raise ValueError(
                f"benchmark row {row_number}: 'runtime_seconds' không được âm"
            )

        normalized.append(
            {
                "mode": mode,
                "repeat": int(repeat),
                **counts,
                # NaN is legitimate only when the corresponding population is empty.
                "correction_rate": correction_rate,
                "retention_rate": retention_rate,
                "validation_balanced_accuracy": validation_bacc,
                "test_balanced_accuracy": test_bacc,
                "runtime_seconds": runtime,
            }
        )

    initial_counts = {row["initial_hard_count"] for row in normalized}
    if len(initial_counts) != 1:
        raise ValueError(
            "initial_hard_count không nhất quán giữa các mode/repeat; "
            "không thể so sánh trực tiếp"
        )
    if not any(row["mode"] == "baseline" for row in normalized):
        raise ValueError("Thiếu mode 'baseline' để tính chênh lệch")
    return normalized


def method_order(summary: dict[str, Any], runs: Sequence[dict[str, Any]]) -> list[str]:
    present = {str(row["mode"]) for row in runs}
    declared = [
        str(mode)
        for mode in summary.get("modes", [])
        if str(mode) in present
    ]
    ordered = list(dict.fromkeys(declared + sorted(present - set(declared))))
    if "baseline" in ordered:
        ordered.remove("baseline")
        ordered.insert(0, "baseline")
    return ordered


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def aggregate_runs(
    summary: dict[str, Any], runs: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[str(row["mode"])].append(row)

    aggregated: list[dict[str, Any]] = []
    numeric_fields = (
        "correction_rate",
        "retention_rate",
        "new_error_count",
        "current_hard_count",
        "test_balanced_accuracy",
        "runtime_seconds",
    )
    for mode in method_order(summary, runs):
        values = grouped[mode]
        item: dict[str, Any] = {"mode": mode, "repeats": len(values)}
        for field in numeric_fields:
            item[field] = mean(row[field] for row in values)
        aggregated.append(item)

    baseline = next(item for item in aggregated if item["mode"] == "baseline")
    baseline_bacc = baseline["test_balanced_accuracy"]
    if baseline_bacc is None:
        raise ValueError("Baseline không có test_balanced_accuracy hợp lệ")
    for item in aggregated:
        test_bacc = item["test_balanced_accuracy"]
        item["delta_test_bacc_pp"] = (
            100.0 * (test_bacc - baseline_bacc)
            if test_bacc is not None
            else None
        )
    return aggregated


def display_name(mode: str) -> str:
    return DISPLAY_NAMES.get(mode, mode.replace("_", " ").title())


def _openpyxl_api() -> dict[str, Any]:
    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.formatting.rule import CellIsRule
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.worksheet.table import Table, TableStyleInfo
    except ImportError as exc:  # pragma: no cover - depends on target runtime
        raise RuntimeError(
            "Thiếu openpyxl. Trên Kaggle chạy: pip install openpyxl"
        ) from exc
    return {
        "Workbook": Workbook,
        "load_workbook": load_workbook,
        "CellIsRule": CellIsRule,
        "Alignment": Alignment,
        "Border": Border,
        "Font": Font,
        "PatternFill": PatternFill,
        "Side": Side,
        "Table": Table,
        "TableStyleInfo": TableStyleInfo,
    }


def _write_table(
    worksheet: Any,
    *,
    start_row: int,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    table_name: str,
    api: dict[str, Any],
) -> int:
    if len(headers) > MAX_REPORT_COLUMNS:
        raise ValueError(
            f"Report vượt giới hạn {MAX_REPORT_COLUMNS} cột: {len(headers)}"
        )
    header_fill = api["PatternFill"]("solid", fgColor="1F4E78")
    header_font = api["Font"](name="Arial", size=10, bold=True, color="FFFFFF")
    thin_side = api["Side"](style="thin", color="D9E2F3")
    body_border = api["Border"](bottom=thin_side)

    for column, header in enumerate(headers, start=1):
        cell = worksheet.cell(row=start_row, column=column, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = api["Alignment"](
            horizontal="center", vertical="center", wrap_text=True
        )
    worksheet.row_dimensions[start_row].height = 30

    for row_offset, values in enumerate(rows, start=1):
        for column, value in enumerate(values, start=1):
            cell = worksheet.cell(
                row=start_row + row_offset, column=column, value=value
            )
            cell.font = api["Font"](name="Arial", size=10, color="1F2937")
            cell.alignment = api["Alignment"](
                horizontal="left" if column == 1 else "right",
                vertical="center",
            )
            cell.border = body_border

    end_row = start_row + len(rows)
    end_column_letter = chr(ord("A") + len(headers) - 1)
    table = api["Table"](
        displayName=table_name,
        ref=f"A{start_row}:{end_column_letter}{end_row}",
    )
    table.tableStyleInfo = api["TableStyleInfo"](
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    worksheet.add_table(table)
    return end_row


def build_workbook(
    summary: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
    source_path: Path,
) -> Any:
    api = _openpyxl_api()
    workbook = api["Workbook"]()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    runs_sheet = workbook.create_sheet("Runs")

    workbook.properties.title = "Binary hard-example correction report"
    workbook.properties.subject = "Compact method comparison"
    workbook.properties.creator = "BKCS research pipeline"

    threshold = finite_number(summary.get("threshold"))
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError("summary.json thiếu threshold hợp lệ trong [0, 1]")
    initial_hard_count = int(runs[0]["initial_hard_count"])
    best = max(
        aggregated,
        key=lambda row: float(row["test_balanced_accuracy"]),
    )

    summary_sheet.sheet_view.showGridLines = False
    summary_sheet.sheet_properties.tabColor = "1F4E78"
    summary_sheet["A2"] = "Kết quả sửa hard examples"
    summary_sheet["A2"].font = api["Font"](
        name="Arial", size=14, bold=True, color="17365D"
    )
    summary_sheet["A3"] = (
        "Phân loại nhị phân KNOWN/UNKNOWN · giá trị là trung bình theo repeat"
    )
    summary_sheet["A3"].font = api["Font"](
        name="Arial", size=10, italic=True, color="667085"
    )

    metadata = (
        ("A5", "Threshold", "B5", threshold, "0.00"),
        ("D5", "Lỗi baseline", "E5", initial_hard_count, "#,##0"),
        ("G5", "Test BAcc tốt nhất", "H5", display_name(best["mode"]), None),
    )
    for label_cell, label, value_cell, value, number_format in metadata:
        summary_sheet[label_cell] = label
        summary_sheet[label_cell].font = api["Font"](
            name="Arial", size=10, bold=True, color="475467"
        )
        summary_sheet[value_cell] = value
        summary_sheet[value_cell].font = api["Font"](
            name="Arial", size=10, bold=True, color="1F2937"
        )
        if number_format:
            summary_sheet[value_cell].number_format = number_format

    summary_sheet["A7"] = "Tổng hợp theo phương pháp"
    summary_sheet["A7"].font = api["Font"](
        name="Arial", size=11, bold=True, color="17365D"
    )
    summary_rows = [
        (
            display_name(row["mode"]),
            row["repeats"],
            row["correction_rate"],
            row["retention_rate"],
            row["new_error_count"],
            row["current_hard_count"],
            row["test_balanced_accuracy"],
            row["delta_test_bacc_pp"],
            row["runtime_seconds"],
        )
        for row in aggregated
    ]
    summary_end = _write_table(
        summary_sheet,
        start_row=8,
        headers=SUMMARY_HEADERS,
        rows=summary_rows,
        table_name="MethodSummary",
        api=api,
    )
    for row in range(9, summary_end + 1):
        summary_sheet.cell(row, 2).number_format = "0"
        for column in (3, 4, 7):
            summary_sheet.cell(row, column).number_format = "0.00%"
        for column in (5, 6):
            summary_sheet.cell(row, column).number_format = "0.0"
        summary_sheet.cell(row, 8).number_format = "+0.00;-0.00;0.00"
        summary_sheet.cell(row, 9).number_format = "0.0"

    best_fill = api["PatternFill"]("solid", fgColor="E2F0D9")
    for row in range(9, summary_end + 1):
        if summary_sheet.cell(row, 1).value == display_name(best["mode"]):
            summary_sheet.cell(row, 7).fill = best_fill
            summary_sheet.cell(row, 7).font = api["Font"](
                name="Arial", size=10, bold=True, color="375623"
            )

    delta_range = f"H9:H{summary_end}"
    summary_sheet.conditional_formatting.add(
        delta_range,
        api["CellIsRule"](
            operator="greaterThan",
            formula=["0"],
            fill=api["PatternFill"]("solid", fgColor="E2F0D9"),
        ),
    )
    summary_sheet.conditional_formatting.add(
        delta_range,
        api["CellIsRule"](
            operator="lessThan",
            formula=["0"],
            fill=api["PatternFill"]("solid", fgColor="FCE4D6"),
        ),
    )

    notes_row = summary_end + 3
    summary_sheet.cell(notes_row, 1, "Cách đọc")
    summary_sheet.cell(notes_row, 1).font = api["Font"](
        name="Arial", size=10, bold=True, color="17365D"
    )
    notes = (
        "Sửa lỗi: tỷ lệ mẫu baseline dự đoán sai đã được sửa đúng.",
        "Giữ mẫu đúng: tỷ lệ mẫu baseline dự đoán đúng vẫn còn đúng.",
        "Δ BAcc: chênh lệch Test Balanced Accuracy so với baseline, tính theo điểm phần trăm.",
    )
    for offset, note in enumerate(notes, start=1):
        summary_sheet.cell(notes_row + offset, 1, note)
        summary_sheet.cell(notes_row + offset, 1).font = api["Font"](
            name="Arial", size=9, italic=True, color="667085"
        )

    summary_sheet.column_dimensions["A"].width = 25
    for column, width in zip("BCDEFGHI", (12, 14, 16, 12, 18, 16, 18, 14)):
        summary_sheet.column_dimensions[column].width = width
    summary_sheet.freeze_panes = "A9"

    runs_sheet.sheet_view.showGridLines = False
    runs_sheet.sheet_properties.tabColor = "5B9BD5"
    runs_sheet["A2"] = "Chi tiết từng lần chạy"
    runs_sheet["A2"].font = api["Font"](
        name="Arial", size=14, bold=True, color="17365D"
    )
    runs_sheet["A3"] = (
        f"Nguồn: {source_path.name} · dùng để kiểm tra độ ổn định giữa các repeat"
    )
    runs_sheet["A3"].font = api["Font"](
        name="Arial", size=10, italic=True, color="667085"
    )
    order_index = {mode: index for index, mode in enumerate(method_order(summary, runs))}
    ordered_runs = sorted(
        runs, key=lambda row: (order_index[str(row["mode"])], int(row["repeat"]))
    )
    run_rows = [
        (
            display_name(row["mode"]),
            int(row["repeat"]),
            row["correction_rate"],
            row["retention_rate"],
            row["new_error_count"],
            row["current_hard_count"],
            row["validation_balanced_accuracy"],
            row["test_balanced_accuracy"],
            row["runtime_seconds"],
        )
        for row in ordered_runs
    ]
    runs_end = _write_table(
        runs_sheet,
        start_row=5,
        headers=RUN_HEADERS,
        rows=run_rows,
        table_name="RepeatDetails",
        api=api,
    )
    for row in range(6, runs_end + 1):
        runs_sheet.cell(row, 2).number_format = "0"
        for column in (3, 4, 7, 8):
            runs_sheet.cell(row, column).number_format = "0.00%"
        for column in (5, 6):
            runs_sheet.cell(row, column).number_format = "0"
        runs_sheet.cell(row, 9).number_format = "0.0"
    runs_sheet.column_dimensions["A"].width = 25
    for column, width in zip("BCDEFGHI", (12, 14, 16, 12, 18, 16, 16, 14)):
        runs_sheet.column_dimensions[column].width = width
    runs_sheet.freeze_panes = "A6"

    return workbook


def verify_workbook(
    path: Path, expected_summary_rows: int, expected_run_rows: int
) -> None:
    api = _openpyxl_api()
    workbook = api["load_workbook"](path, data_only=False, read_only=False)
    if workbook.sheetnames != ["Summary", "Runs"]:
        raise ValueError(f"Sheet không đúng: {workbook.sheetnames}")

    summary_sheet = workbook["Summary"]
    runs_sheet = workbook["Runs"]
    summary_headers = tuple(
        summary_sheet.cell(8, column).value for column in range(1, 10)
    )
    run_headers = tuple(runs_sheet.cell(5, column).value for column in range(1, 10))
    if summary_headers != SUMMARY_HEADERS or run_headers != RUN_HEADERS:
        raise ValueError("Header report không đúng schema gọn 9 cột")
    if summary_sheet.max_column > MAX_REPORT_COLUMNS:
        raise ValueError("Summary có nhiều hơn 9 cột")
    if runs_sheet.max_column > MAX_REPORT_COLUMNS:
        raise ValueError("Runs có nhiều hơn 9 cột")

    actual_summary_rows = sum(
        summary_sheet.cell(row, 1).value is not None
        for row in range(9, 9 + expected_summary_rows)
    )
    actual_run_rows = sum(
        runs_sheet.cell(row, 1).value is not None
        for row in range(6, 6 + expected_run_rows)
    )
    if actual_summary_rows != expected_summary_rows:
        raise ValueError("Thiếu dòng trong Summary")
    if actual_run_rows != expected_run_rows:
        raise ValueError("Thiếu dòng trong Runs")

    for worksheet in (summary_sheet, runs_sheet):
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.data_type == "e":
                    raise ValueError(
                        f"Lỗi Excel tại {worksheet.title}!{cell.coordinate}: {cell.value}"
                    )
    workbook.close()


def main() -> int:
    configure_console()
    args = parse_args()
    try:
        summary_path, summary = discover_summary(args.input)
        runs = normalize_runs(summary)
        aggregated = aggregate_runs(summary, runs)
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else summary_path.parent / DEFAULT_REPORT_NAME
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        workbook = build_workbook(summary, runs, aggregated, summary_path)
        workbook.save(output_path)
        verify_workbook(output_path, len(aggregated), len(runs))
        print(
            f"Created {output_path} "
            f"({len(aggregated)} methods, {len(runs)} runs, 9 columns per table)"
        )
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
