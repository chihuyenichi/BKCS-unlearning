#!/usr/bin/env python3
"""Generate Google Sheets-friendly XLSX reports from unlearning outputs.

The script consumes the ``summary.json`` schema produced by ``pipeline.ipynb``
on Kaggle or VM. It can process one ``forget_<count>`` directory or
recursively discover every benchmark summary below a larger output directory.
"""

from __future__ import annotations

import argparse
from copy import copy
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.formatting.rule import CellIsRule, FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError as exc:  # pragma: no cover - depends on the runtime image
    raise SystemExit(
        "Missing dependency 'openpyxl'. Install it with: pip install openpyxl"
    ) from exc


DISPLAY_NAMES = {
    "baseline": "Baseline",
    "only_head": "Only Head",
    "full_mlp": "Full MLP",
    "last_encoder_block": "Last Encoder",
    "full_model": "Full Model",
    "l2ul_neggrad": "L2UL NegGrad",
    "l2ul_adv": "L2UL ADV",
    "l2ul_adv_imp": "L2UL ADV + IMP",
    "amnesiac": "Amnesiac",
    "blindspot": "Blindspot",
    "bnd": "BND",
    "mkmmd": "MKMMD",
}

COMMON_HYPERPARAMETERS = (
    ("epochs", "Epoch tối đa", "0"),
    ("learning_rate", "Learning rate", "0.00E+00"),
    ("weight_decay", "Weight decay", "0.00E+00"),
    ("max_steps_per_epoch", "Max steps / epoch", "0"),
    ("lambda_forget", "Lambda forget", "0.00"),
    ("lambda_retain", "Lambda retain", "0.00"),
    ("min_epochs", "Min epochs", "0"),
    ("early_stop_forget_success", "Early-stop target", "0.00%"),
)

MODE_HYPERPARAMETERS = {
    "l2ul_neggrad": (
        ("optimizer", "Optimizer", "General"),
        ("momentum", "Momentum", "0.00"),
    ),
    "l2ul_adv": (
        ("optimizer", "Optimizer", "General"),
        ("momentum", "Momentum", "0.00"),
        ("lambda_adv", "Lambda ADV", "0.00"),
        ("adversarial_variants", "ADV / instance", "0"),
        ("pgd_steps", "PGD steps", "0"),
        ("pgd_epsilon", "PGD L2 epsilon", "0.000"),
        ("pgd_step_size", "PGD step size", "0.000"),
        ("max_packet_size", "Packet-size scale", "0.0"),
        ("round_packet_size", "Round packet size", "General"),
        ("pgd_random_start", "PGD random start", "General"),
    ),
    "l2ul_adv_imp": (
        ("optimizer", "Optimizer", "General"),
        ("momentum", "Momentum", "0.00"),
        ("lambda_adv", "Lambda ADV", "0.00"),
        ("lambda_importance", "Lambda IMP", "0.00"),
        ("adversarial_variants", "ADV / instance", "0"),
        ("pgd_steps", "PGD steps", "0"),
        ("pgd_epsilon", "PGD L2 epsilon", "0.000"),
        ("pgd_step_size", "PGD step size", "0.000"),
        ("max_packet_size", "Packet-size scale", "0.0"),
        ("round_packet_size", "Round packet size", "General"),
        ("pgd_random_start", "PGD random start", "General"),
    ),
    "blindspot": (
        ("blindspot_temperature", "Temperature", "0.00"),
        ("blindspot_unknown_confidence", "Unknown confidence", "0.00%"),
    ),
    "bnd": (("boundary_margin", "Boundary margin", "0.00"),),
    "mkmmd": (
        ("lambda_mkmmd_forget", "Lambda MKMMD forget", "0.00"),
        ("lambda_mkmmd_retain", "Lambda MKMMD retain", "0.00"),
        ("mkmmd_bandwidths", "MKMMD bandwidths", "General"),
    ),
}


@dataclass(frozen=True)
class MetricSpec:
    group: str
    label: str
    direction: str
    key: str
    number_format: str


METRICS = (
    MetricSpec("Unlearning", "Forget Success", "Cao hơn tốt", "forget_success", "0.00%"),
    MetricSpec(
        "Unlearning",
        "Paper Misclassification",
        "Cao hơn tốt",
        "paper_misclassification",
        "0.00%",
    ),
    MetricSpec(
        "Unlearning",
        "Xác suất UNKNOWN trên Df",
        "Cao hơn tốt",
        "forget_probability_unknown",
        "0.00%",
    ),
    MetricSpec("Phân loại", "Accuracy", "Cao hơn tốt", "accuracy", "0.00%"),
    MetricSpec(
        "Phân loại",
        "Precision (UNKNOWN positive)",
        "Cao hơn tốt",
        "unknown_precision",
        "0.00%",
    ),
    MetricSpec(
        "Phân loại",
        "Recall (UNKNOWN positive)",
        "Cao hơn tốt",
        "unknown_recall",
        "0.00%",
    ),
    MetricSpec(
        "Phân loại",
        "F1 (UNKNOWN positive)",
        "Cao hơn tốt",
        "unknown_f1",
        "0.00%",
    ),
    MetricSpec(
        "Phân loại",
        "Balanced Accuracy",
        "Cao hơn tốt",
        "balanced_accuracy",
        "0.00%",
    ),
    MetricSpec("Retain", "Known Recall", "Cao hơn tốt", "known_recall", "0.00%"),
    MetricSpec("Retain", "Unknown Recall", "Cao hơn tốt", "unknown_recall", "0.00%"),
    MetricSpec(
        "Retain",
        "Same-label Known Recall",
        "Cao hơn tốt",
        "same_label_known_recall",
        "0.00%",
    ),
    MetricSpec(
        "Retain",
        "Other-known Recall",
        "Cao hơn tốt",
        "other_known_recall",
        "0.00%",
    ),
    MetricSpec(
        "Retain",
        "Unknown-100 Recall",
        "Cao hơn tốt",
        "unknown_100_recall",
        "0.00%",
    ),
    MetricSpec(
        "Confusion",
        "TN: KNOWN dự đoán KNOWN",
        "Cao hơn tốt",
        "tn",
        "#,##0.00",
    ),
    MetricSpec(
        "Confusion",
        "FP: KNOWN dự đoán UNKNOWN",
        "Thấp hơn tốt",
        "fp",
        "#,##0.00",
    ),
    MetricSpec(
        "Confusion",
        "FN: UNKNOWN dự đoán KNOWN",
        "Thấp hơn tốt",
        "fn",
        "#,##0.00",
    ),
    MetricSpec(
        "Confusion",
        "TP: UNKNOWN dự đoán UNKNOWN",
        "Cao hơn tốt",
        "tp",
        "#,##0.00",
    ),
    MetricSpec(
        "Confusion",
        "Tổng lỗi (FP + FN)",
        "Thấp hơn tốt",
        "total_error",
        "#,##0.00",
    ),
    MetricSpec(
        "Runtime",
        "Thời gian mỗi repeat",
        "Thấp hơn tốt",
        "runtime_seconds",
        "#,##0.00",
    ),
)

NAVY = "17365D"
BLUE = "2F75B5"
LIGHT_BLUE = "D9EAF7"
PALE_BLUE = "EAF3F8"
GREEN = "E2F0D9"
GREEN_TEXT = "375623"
RED = "FCE4D6"
RED_TEXT = "9C0006"
GRAY = "F2F2F2"
MID_GRAY = "D9E1F2"
WHITE = "FFFFFF"
TEXT = "1F2937"
THIN_GRAY = Side(style="thin", color="D9E2F3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one XLSX benchmark report per discovered forget_<count> summary.json"
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="summary.json, a forget_<count> directory, or a parent output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output .xlsx for one summary, or an output directory when multiple "
            "summaries are discovered"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-open each generated workbook and validate its sheets and formulas",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Also write one flat CSV report; valid only when one summary is selected",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def is_benchmark_summary(payload: dict[str, Any]) -> bool:
    required = {"forget_count", "threshold", "before", "experiments"}
    return required.issubset(payload) and isinstance(payload.get("experiments"), list)


def discover_summaries(input_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir() and (input_path / "summary.json").is_file():
        candidates = [input_path / "summary.json"]
    elif input_path.is_dir():
        candidates = sorted(input_path.rglob("summary.json"))
    else:
        raise FileNotFoundError(input_path)

    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        payload = load_json(path)
        if is_benchmark_summary(payload):
            summaries.append((path, payload))
    if not summaries:
        raise ValueError(f"No compatible benchmark summary.json found below: {input_path}")

    def sort_key(item: tuple[Path, dict[str, Any]]) -> tuple[int, str]:
        return int(item[1].get("forget_count", 0)), str(item[0])

    return sorted(summaries, key=sort_key)


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def confusion_values(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    confusion = metrics.get("confusion_matrix")
    if (
        not isinstance(confusion, list)
        or len(confusion) != 2
        or not all(isinstance(row, list) and len(row) == 2 for row in confusion)
    ):
        raise ValueError("Expected confusion_matrix=[[TN, FP], [FN, TP]]")
    tn, fp = map(float, confusion[0])
    fn, tp = map(float, confusion[1])
    return tn, fp, fn, tp


def extract_metrics(
    forget: dict[str, Any],
    paper_forget: dict[str, Any],
    test: dict[str, Any],
    runtime_seconds: Any,
) -> dict[str, float | None]:
    full = test.get("full_retain")
    if not isinstance(full, dict):
        raise ValueError("Missing test.full_retain metrics")
    tn, fp, fn, tp = confusion_values(full)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        safe_div(2.0 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )

    return {
        "forget_success": as_number(forget.get("forget_success_rate")),
        "paper_misclassification": as_number(
            paper_forget.get("misclassification_rate")
        ),
        "forget_probability_unknown": as_number(
            forget.get("mean_probability_unknown")
        ),
        "accuracy": as_number(full.get("accuracy")),
        "unknown_precision": precision,
        "unknown_recall": as_number(full.get("unknown_recall")),
        "unknown_f1": f1,
        "balanced_accuracy": as_number(full.get("balanced_accuracy")),
        "known_recall": as_number(full.get("known_recall")),
        "same_label_known_recall": as_number(
            nested(test, "same_original_label", "known_recall")
        ),
        "other_known_recall": as_number(nested(test, "other_known", "known_recall")),
        "unknown_100_recall": as_number(
            nested(test, "unknown_100", "unknown_recall")
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_error": fp + fn,
        "runtime_seconds": as_number(runtime_seconds),
    }


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return statistics.fmean(clean), statistics.pstdev(clean)


def baseline_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    experiments = summary.get("experiments", [])
    baseline_runtime = next(
        (
            experiment.get("runtime_seconds")
            for experiment in experiments
            if experiment.get("mode") == "baseline"
        ),
        None,
    )
    before = summary.get("before", {})
    return extract_metrics(
        before.get("forget", {}),
        before.get("paper_forget", {}),
        before.get("test", {}),
        baseline_runtime,
    )


def experiments_by_mode(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for experiment in summary.get("experiments", []):
        mode = str(experiment.get("mode", "")).strip()
        if not mode:
            raise ValueError("Experiment is missing mode")
        grouped.setdefault(mode, []).append(experiment)
    if not grouped:
        raise ValueError("summary.json contains no experiments")
    return grouped


def ordered_modes(summary: dict[str, Any], grouped: dict[str, Any]) -> list[str]:
    declared = [str(mode) for mode in summary.get("modes", []) if mode in grouped]
    remaining = sorted(set(grouped) - set(declared))
    modes = declared + remaining
    if "baseline" in modes:
        modes.remove("baseline")
        modes.insert(0, "baseline")
    return modes


def experiment_metric_rows(
    experiments: Sequence[dict[str, Any]],
) -> list[dict[str, float | None]]:
    return [
        extract_metrics(
            experiment.get("after_forget", {}),
            experiment.get("after_paper_forget", {}),
            experiment.get("after_test", {}),
            experiment.get("runtime_seconds"),
        )
        for experiment in experiments
    ]


def display_name(mode: str) -> str:
    return DISPLAY_NAMES.get(mode, mode.replace("_", " ").title())


def unique_sheet_name(base: str, used: set[str]) -> str:
    invalid = set("[]:*?/\\")
    cleaned = "".join("_" if char in invalid else char for char in base)[:31] or "Method"
    candidate = cleaned
    index = 2
    while candidate.casefold() in used:
        suffix = f" {index}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def value_for_excel(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def thin_bottom_border(color: str = "A6B9CC") -> Border:
    return Border(bottom=Side(style="thin", color=color))


def section_header(cells: Iterable[Any]) -> None:
    fill = PatternFill("solid", fgColor=NAVY)
    for cell in cells:
        cell.fill = fill
        cell.font = Font(name="Arial", size=10, bold=True, color=WHITE)
        cell.alignment = Alignment(horizontal="center", vertical="center")


def apply_base_sheet_style(ws: Any) -> None:
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows():
        for cell in row:
            font = copy(cell.font)
            font.name = "Arial"
            font.size = font.size or 10
            cell.font = font
            alignment = copy(cell.alignment)
            alignment.vertical = "center"
            cell.alignment = alignment


def style_title(ws: Any, title: str, subtitle: str, width: int) -> None:
    ws.cell(2, 1, title)
    ws.cell(2, 1).font = Font(name="Arial", size=16, bold=True, color=NAVY)
    ws.cell(3, 1, subtitle)
    ws.cell(3, 1).font = Font(name="Arial", size=10, italic=True, color="52606D")
    ws.cell(3, 1).alignment = Alignment(wrap_text=False)
    for col in range(1, width + 1):
        ws.cell(4, col).border = thin_bottom_border()
    ws.row_dimensions[2].height = 24
    ws.row_dimensions[3].height = 18


def scope_for(summary: dict[str, Any], mode: str, experiment: dict[str, Any]) -> str:
    mode_spec = summary.get("mode_specs", {}).get(mode, {})
    if not isinstance(mode_spec, dict):
        mode_spec = {}
    experiment_spec = experiment.get("mode_spec", {})
    if not isinstance(experiment_spec, dict):
        experiment_spec = {}
    return str(
        mode_spec.get("trainable_scope")
        or experiment_spec.get("trainable_scope")
        or "none"
    )


def objective_for(summary: dict[str, Any], mode: str, experiment: dict[str, Any]) -> str:
    mode_spec = summary.get("mode_specs", {}).get(mode, {})
    if not isinstance(mode_spec, dict):
        mode_spec = {}
    experiment_spec = experiment.get("mode_spec", {})
    if not isinstance(experiment_spec, dict):
        experiment_spec = {}
    return str(mode_spec.get("objective") or experiment_spec.get("objective") or mode)


def epochs_for(experiment: dict[str, Any]) -> float:
    history = experiment.get("history")
    return float(len(history)) if isinstance(history, list) else 0.0


def method_hyperparameters(
    summary: dict[str, Any], mode: str, experiment: dict[str, Any]
) -> dict[str, Any]:
    from_summary = summary.get("mode_hyperparameters", {}).get(mode)
    if isinstance(from_summary, dict):
        return from_summary
    from_experiment = experiment.get("mode_hyperparameters")
    return from_experiment if isinstance(from_experiment, dict) else {}


def add_summary_sheet(
    wb: Workbook,
    summary: dict[str, Any],
    modes: Sequence[str],
    grouped: dict[str, list[dict[str, Any]]],
    baseline: dict[str, float | None],
) -> None:
    ws = wb.active
    ws.title = "Summary"
    apply_base_sheet_style(ws)
    forget_count = int(summary["forget_count"])
    threshold = float(summary["threshold"])
    style_title(
        ws,
        "Instance-wise Unlearning Benchmark",
        (
            f"Df={forget_count} | Positive class=UNKNOWN | Threshold={threshold:g} | "
            "So sánh mean theo repeat"
        ),
        15,
    )

    headers = (
        "Mode",
        "Scope",
        "Repeats",
        "Epoch thực chạy",
        "Forget Success",
        "Accuracy",
        "Balanced Accuracy",
        "Known Recall",
        "Unknown Recall",
        "FP",
        "FN",
        "Tổng lỗi",
        "Delta lỗi",
        "Giảm lỗi so baseline",
        "Runtime (s)",
    )
    for col, header in enumerate(headers, 1):
        ws.cell(6, col, header)
    section_header(ws[6][0:15])
    ws.row_dimensions[6].height = 30

    baseline_row = 7
    for offset, mode in enumerate(modes):
        row = 7 + offset
        experiments = grouped[mode]
        metric_rows = experiment_metric_rows(experiments)
        first = experiments[0]
        epoch_mean, _ = mean_std(epochs_for(item) for item in experiments)
        means = {
            key: mean_std(values.get(key) for values in metric_rows)[0]
            for key in baseline
        }
        if mode == "baseline":
            baseline_row = row
        values = (
            display_name(mode),
            scope_for(summary, mode, first),
            len(experiments),
            epoch_mean,
            means["forget_success"],
            means["accuracy"],
            means["balanced_accuracy"],
            means["known_recall"],
            means["unknown_recall"],
            means["fp"],
            means["fn"],
            means["total_error"],
            None,
            None,
            means["runtime_seconds"],
        )
        for col, value in enumerate(values, 1):
            ws.cell(row, col, value)
        ws.cell(row, 13, f"=L{row}-$L${baseline_row}")
        ws.cell(row, 14, f'=IF($L${baseline_row}=0,"",-M{row}/$L${baseline_row})')
        fill = PatternFill("solid", fgColor=PALE_BLUE if row % 2 else WHITE)
        for cell in ws[row][0:15]:
            cell.fill = fill
            cell.border = Border(bottom=THIN_GRAY)

    last_row = 6 + len(modes)
    for row in range(7, last_row + 1):
        for col in range(5, 10):
            ws.cell(row, col).number_format = "0.00%"
        for col in range(10, 14):
            ws.cell(row, col).number_format = "#,##0.00"
        ws.cell(row, 14).number_format = "0.00%"
        ws.cell(row, 15).number_format = "#,##0.00"

    legend_row = last_row + 3
    legend = (
        ("Ký hiệu", "Trường hợp", "Chiều tốt"),
        ("TN", "KNOWN -> KNOWN", "Tăng"),
        ("FP", "KNOWN -> UNKNOWN", "Giảm"),
        ("FN", "UNKNOWN -> KNOWN", "Giảm"),
        ("TP", "UNKNOWN -> UNKNOWN", "Tăng"),
    )
    for row_offset, values in enumerate(legend):
        for col, value in enumerate(values, 1):
            ws.cell(legend_row + row_offset, col, value)
    section_header(ws[legend_row][0:3])
    for row in range(legend_row + 1, legend_row + 5):
        for col in range(1, 4):
            ws.cell(row, col).border = Border(bottom=THIN_GRAY)

    ws.freeze_panes = "A7"
    ws.auto_filter.ref = f"A6:O{last_row}"
    widths = (18, 24, 11, 16, 15, 13, 18, 15, 15, 11, 11, 12, 12, 20, 14)
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.sheet_properties.tabColor = NAVY
    apply_base_sheet_style(ws)


def add_method_sheet(
    wb: Workbook,
    sheet_name: str,
    summary: dict[str, Any],
    mode: str,
    experiments: Sequence[dict[str, Any]],
    baseline: dict[str, float | None],
) -> None:
    ws = wb.create_sheet(sheet_name)
    apply_base_sheet_style(ws)
    first = experiments[0]
    title = display_name(mode)
    objective = objective_for(summary, mode, first)
    scope = scope_for(summary, mode, first)
    style_title(
        ws,
        f"{title}: đánh giá so với Baseline",
        f"Objective={objective} | Scope={scope} | {len(experiments)} repeat",
        9,
    )

    ws["A6"] = "Run context"
    ws["B6"] = "Giá trị"
    ws["D6"] = "Hyperparameter"
    ws["E6"] = "Giá trị"
    section_header((ws["A6"], ws["B6"], ws["D6"], ws["E6"]))

    epoch_mean, epoch_std = mean_std(epochs_for(item) for item in experiments)
    context = (
        ("Forget count", int(summary["forget_count"]), "0"),
        ("Threshold UNKNOWN", float(summary["threshold"]), "0.00"),
        ("Positive class", "UNKNOWN", "General"),
        ("Repeats", len(experiments), "0"),
        (
            "Epoch thực chạy (mean ± std)",
            f"{epoch_mean or 0:.2f} ± {epoch_std or 0:.2f}",
            "General",
        ),
    )
    for index, (label, value, number_format) in enumerate(context, 7):
        ws.cell(index, 1, label)
        ws.cell(index, 2, value)
        ws.cell(index, 2).number_format = number_format
        ws.cell(index, 1).fill = PatternFill("solid", fgColor=GRAY)
        ws.cell(index, 1).font = Font(name="Arial", size=10, bold=True, color=TEXT)

    config = method_hyperparameters(summary, mode, first)
    hyper_specs = COMMON_HYPERPARAMETERS + MODE_HYPERPARAMETERS.get(mode, ())
    for index, (key, label, number_format) in enumerate(hyper_specs, 7):
        value = value_for_excel(config.get(key))
        ws.cell(index, 4, label)
        ws.cell(index, 5, value)
        ws.cell(index, 5).number_format = number_format
        ws.cell(index, 4).fill = PatternFill("solid", fgColor=GRAY)
        ws.cell(index, 4).font = Font(name="Arial", size=10, bold=True, color=TEXT)

    hyperparameter_last_row = 6 + len(hyper_specs)
    metric_header_row = max(21, hyperparameter_last_row + 2)
    metric_headers = (
        "Nhóm",
        "Metric",
        "Chiều đánh giá",
        "Baseline",
        "Method mean",
        "Std",
        "Delta",
        "% thay đổi",
        "Kết luận",
    )
    for col, header in enumerate(metric_headers, 1):
        ws.cell(metric_header_row, col, header)
    section_header(ws[metric_header_row][0:9])
    ws.row_dimensions[metric_header_row].height = 28

    extracted = experiment_metric_rows(experiments)
    first_metric_row = metric_header_row + 1
    for offset, spec in enumerate(METRICS):
        row = first_metric_row + offset
        mean, std = mean_std(item.get(spec.key) for item in extracted)
        ws.cell(row, 1, spec.group)
        ws.cell(row, 2, spec.label)
        ws.cell(row, 3, spec.direction)
        ws.cell(row, 4, baseline.get(spec.key))
        ws.cell(row, 5, mean)
        ws.cell(row, 6, std)
        ws.cell(row, 7, f"=E{row}-D{row}")
        ws.cell(row, 8, f'=IF(D{row}=0,"",G{row}/ABS(D{row}))')
        ws.cell(
            row,
            9,
            (
                f'=IF(ABS(G{row})<0.0000001,"Không đổi",'
                f'IF(C{row}="Cao hơn tốt",IF(G{row}>0,"Tốt hơn","Kém hơn"),'
                f'IF(C{row}="Thấp hơn tốt",IF(G{row}<0,"Tốt hơn","Kém hơn"),'
                '"Tham khảo")))'
            ),
        )
        for col in range(4, 8):
            ws.cell(row, col).number_format = spec.number_format
        ws.cell(row, 8).number_format = "0.00%"
        fill = PatternFill("solid", fgColor=PALE_BLUE if offset % 2 == 0 else WHITE)
        for cell in ws[row][0:9]:
            cell.fill = fill
            cell.border = Border(bottom=THIN_GRAY)

    last_metric_row = first_metric_row + len(METRICS) - 1
    conclusion_range = f"I{first_metric_row}:I{last_metric_row}"
    ws.conditional_formatting.add(
        conclusion_range,
        FormulaRule(
            formula=[f'$I{first_metric_row}="Tốt hơn"'],
            fill=PatternFill("solid", fgColor=GREEN),
            font=Font(color=GREEN_TEXT, bold=True),
        ),
    )
    ws.conditional_formatting.add(
        conclusion_range,
        FormulaRule(
            formula=[f'$I{first_metric_row}="Kém hơn"'],
            fill=PatternFill("solid", fgColor=RED),
            font=Font(color=RED_TEXT, bold=True),
        ),
    )
    delta_range = f"G{first_metric_row}:G{last_metric_row}"
    ws.conditional_formatting.add(
        delta_range,
        CellIsRule(
            operator="greaterThan",
            formula=["0"],
            fill=PatternFill("solid", fgColor=LIGHT_BLUE),
        ),
    )

    ws.freeze_panes = f"A{first_metric_row}"
    ws.auto_filter.ref = f"A{metric_header_row}:I{last_metric_row}"
    widths = (30, 34, 17, 27, 18, 13, 13, 14, 14)
    for col, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.sheet_properties.tabColor = BLUE if mode != "baseline" else NAVY
    apply_base_sheet_style(ws)


def build_workbook(summary: dict[str, Any]) -> Workbook:
    grouped = experiments_by_mode(summary)
    modes = ordered_modes(summary, grouped)
    baseline = baseline_metrics(summary)
    wb = Workbook()
    add_summary_sheet(wb, summary, modes, grouped, baseline)

    used_names = {"summary"}
    for mode in modes:
        sheet_name = unique_sheet_name(display_name(mode), used_names)
        add_method_sheet(wb, sheet_name, summary, mode, grouped[mode], baseline)

    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.calculation.calcMode = "auto"
    return wb


def output_paths(
    input_path: Path,
    output: Path | None,
    summaries: Sequence[tuple[Path, dict[str, Any]]],
) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if len(summaries) == 1:
        count = int(summaries[0][1]["forget_count"])
        if output is None:
            directory = input_path if input_path.is_dir() else input_path.parent
            return [directory / f"unlearning_benchmark_forget_{count}.xlsx"]
        resolved = output.expanduser().resolve()
        if resolved.suffix.lower() == ".xlsx":
            return [resolved]
        return [resolved / f"unlearning_benchmark_forget_{count}.xlsx"]

    if output is not None and output.suffix.lower() == ".xlsx":
        raise ValueError(
            "Multiple summaries were found; --output must be a directory, not one .xlsx file"
        )
    directory = (
        output.expanduser().resolve()
        if output is not None
        else (input_path if input_path.is_dir() else input_path.parent) / "xlsx_reports"
    )
    return [
        directory / f"unlearning_benchmark_forget_{int(summary['forget_count'])}.xlsx"
        for _, summary in summaries
    ]


def verify_workbook(path: Path, expected_modes: Sequence[str]) -> None:
    workbook = load_workbook(path, data_only=False, read_only=False)
    expected_count = 1 + len(expected_modes)
    if len(workbook.sheetnames) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} sheets, found {len(workbook.sheetnames)}"
        )
    if workbook.sheetnames[0] != "Summary":
        raise ValueError(f"{path}: first sheet must be Summary")
    formula_count = 0
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    formula_count += 1
    expected_minimum = 2 * len(expected_modes) + 3 * len(METRICS) * len(expected_modes)
    if formula_count < expected_minimum:
        raise ValueError(
            f"{path}: expected at least {expected_minimum} formulas, found {formula_count}"
        )
    workbook.close()


def percent(value: float | None) -> float | None:
    return round(value * 100.0, 6) if value is not None else None


def change_percent(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return round(100.0 * (value - baseline) / abs(baseline), 6)


def write_flat_csv(path: Path, summary: dict[str, Any]) -> None:
    grouped = experiments_by_mode(summary)
    modes = ordered_modes(summary, grouped)
    baseline = baseline_metrics(summary)
    rate_columns = (
        ("Forget Success", "forget_success"),
        ("Paper Misclassification", "paper_misclassification"),
        ("Forget Unknown Probability", "forget_probability_unknown"),
        ("Test Accuracy", "accuracy"),
        ("Test Precision", "unknown_precision"),
        ("Test Recall", "unknown_recall"),
        ("Test F1", "unknown_f1"),
        ("Test Balanced Accuracy", "balanced_accuracy"),
        ("Test Known Recall", "known_recall"),
        ("Test Unknown Recall", "unknown_recall"),
        ("Same Label Known Recall", "same_label_known_recall"),
        ("Other Known Recall", "other_known_recall"),
        ("Unknown 100 Recall", "unknown_100_recall"),
    )
    delta_rate_columns = (
        ("Test Accuracy", "accuracy"),
        ("Test Precision", "unknown_precision"),
        ("Test Recall", "unknown_recall"),
        ("Test F1", "unknown_f1"),
        ("Test Balanced Accuracy", "balanced_accuracy"),
        ("Test Known Recall", "known_recall"),
        ("Test Unknown Recall", "unknown_recall"),
    )
    confusion_columns = (("TN", "tn"), ("FP", "fp"), ("FN", "fn"), ("TP", "tp"))
    headers = [
        "Method",
        "Mode",
        "Objective",
        "Trainable Scope",
        "Dataset",
        "Forget Count",
        "Test Samples",
        "Repeats",
        "Positive Class",
        "Threshold",
    ]
    for label, _ in rate_columns:
        headers.extend((f"{label} Mean (%)", f"{label} Std (%)"))
    for label, _ in delta_rate_columns:
        headers.append(f"{label} Delta vs Baseline (pp)")
    for label, _ in confusion_columns:
        headers.extend(
            (
                f"{label} Mean",
                f"{label} Std",
                f"{label} Delta vs Baseline",
                f"{label} Change vs Baseline (%)",
            )
        )
    headers.extend(
        (
            "Total Error Mean",
            "Total Error Std",
            "Total Error Delta vs Baseline",
            "Total Error Reduction (%) Compared to Baseline",
            "Runtime Mean (s)",
            "Runtime Std (s)",
        )
    )

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for mode in modes:
            experiments = grouped[mode]
            first = experiments[0]
            extracted = experiment_metric_rows(experiments)
            aggregates = {
                key: mean_std(item.get(key) for item in extracted)
                for key in baseline
            }
            test_samples = sum(
                baseline[key] or 0.0 for key in ("tn", "fp", "fn", "tp")
            )
            row: dict[str, Any] = {
                "Method": display_name(mode),
                "Mode": mode,
                "Objective": objective_for(summary, mode, first),
                "Trainable Scope": scope_for(summary, mode, first),
                "Dataset": "BKCS",
                "Forget Count": int(summary["forget_count"]),
                "Test Samples": int(test_samples),
                "Repeats": len(experiments),
                "Positive Class": "UNKNOWN",
                "Threshold": float(summary["threshold"]),
            }
            for label, key in rate_columns:
                mean, std = aggregates[key]
                row[f"{label} Mean (%)"] = percent(mean)
                row[f"{label} Std (%)"] = percent(std)
            for label, key in delta_rate_columns:
                mean, _ = aggregates[key]
                base = baseline[key]
                row[f"{label} Delta vs Baseline (pp)"] = (
                    round(100.0 * (mean - base), 6)
                    if mean is not None and base is not None
                    else None
                )
            for label, key in confusion_columns:
                mean, std = aggregates[key]
                base = baseline[key]
                row[f"{label} Mean"] = round(mean, 6) if mean is not None else None
                row[f"{label} Std"] = round(std, 6) if std is not None else None
                row[f"{label} Delta vs Baseline"] = (
                    round(mean - base, 6)
                    if mean is not None and base is not None
                    else None
                )
                row[f"{label} Change vs Baseline (%)"] = change_percent(mean, base)
            error_mean, error_std = aggregates["total_error"]
            baseline_error = baseline["total_error"]
            runtime_mean, runtime_std = aggregates["runtime_seconds"]
            row["Total Error Mean"] = error_mean
            row["Total Error Std"] = error_std
            row["Total Error Delta vs Baseline"] = (
                error_mean - baseline_error
                if error_mean is not None and baseline_error is not None
                else None
            )
            row["Total Error Reduction (%) Compared to Baseline"] = (
                round(100.0 * (baseline_error - error_mean) / baseline_error, 6)
                if error_mean is not None and baseline_error not in (None, 0)
                else None
            )
            row["Runtime Mean (s)"] = runtime_mean
            row["Runtime Std (s)"] = runtime_std
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    try:
        summaries = discover_summaries(args.input)
        if args.csv_output is not None and len(summaries) != 1:
            raise ValueError("--csv-output requires exactly one discovered summary.json")
        destinations = output_paths(args.input, args.output, summaries)
        for (summary_path, summary), destination in zip(summaries, destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            workbook = build_workbook(summary)
            workbook.save(destination)
            if args.verify:
                grouped = experiments_by_mode(summary)
                verify_workbook(destination, ordered_modes(summary, grouped))
            print(
                f"Created {destination} from {summary_path} "
                f"(forget_count={summary['forget_count']})"
            )
        if args.csv_output is not None:
            write_flat_csv(args.csv_output, summaries[0][1])
            print(f"Created {args.csv_output.expanduser().resolve()}")
        return 0
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
