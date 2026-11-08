#!/usr/bin/env python3
"""Build the hard-example correction workbook in the benchmark-report layout.

The layout follows ``outputs/linh_san_unlearning_benchmark_forget_1.xlsx``: one
``Summary`` sheet with a row per mode, then one sheet per mode comparing the
mode mean against the common baseline with live Delta / % / verdict formulas.

Usage:
    python generate_benchmark_report.py \
        outputs/son-tung-binary-hard-example/hard_example_correction/summary.json \
        --output outputs/son_tung_hard_example_benchmark.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR / "outputs/son-tung-binary-hard-example/hard_example_correction/summary.json"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs/son_tung_hard_example_benchmark.xlsx"

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "uniform_finetune": "Uniform Finetune",
    "hard_weighted_ce": "Hard Weighted CE",
    "hard_weighted_center": "Hard Weighted Center",
    "hard_weighted_mkmmd": "Hard Weighted MKMMD",
    "hard_aware_mkmmd": "Hard-aware MKMMD (λ=0.1)",
    "hard_aware_mkmmd_l05": "Hard-aware MKMMD (λ=0.5)",
    "hard_aware_mkmmd_l10": "Hard-aware MKMMD (λ=1.0)",
    "hard_aware_mkmmd_ramp_l002": "Hard-aware MKMMD ramp (λ=0.02)",
    "hard_aware_mkmmd_ramp_l005": "Hard-aware MKMMD ramp (λ=0.05)",
    "hard_aware_mkmmd_ramp_l010": "Hard-aware MKMMD ramp (λ=0.10)",
    "hard_aware_mkmmd_ramp_l020": "Hard-aware MKMMD ramp (λ=0.20)",
    "hard_aware_cdd_m05": "Hard-aware CDD (μ=0.5)",
    "hard_aware_cdd_m10": "Hard-aware CDD (μ=1.0)",
    "blindspot_kd": "Blindspot KD",
    "boundary_shrink": "Boundary Shrink",
}

# Styles copied from the reference workbook.
NAVY = "17365D"
FONT = "Arial"
TITLE_FONT = Font(name=FONT, size=16, bold=True, color=NAVY)
SUBTITLE_FONT = Font(name=FONT, size=10, italic=True, color="52606D")
HEADER_FONT = Font(name=FONT, size=10, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
LABEL_FONT = Font(name=FONT, size=10, bold=True, color="1F2937")
LABEL_FILL = PatternFill("solid", fgColor="F2F2F2")
BODY_FONT = Font(name=FONT, size=11)
STRIPE_FILLS = (PatternFill("solid", fgColor="EAF3F8"), PatternFill("solid", fgColor="FFFFFF"))
THIN = Side(style="thin", color="D9E2F3")
BODY_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
MIDDLE = Alignment(vertical="center")

PCT = "0.00%"
NUM = "#,##0.00"

HIGHER = "Cao hơn tốt"
LOWER = "Thấp hơn tốt"


MODE_ORDER = tuple(DISPLAY_NAMES)


def load_from_results(root: Path) -> dict[str, Any]:
    """Rebuild a summary from per-run ``result.json`` files in a run folder.

    Used when ``summary.json`` is missing or stale (for example a partial
    download).  Only results written by the current pipeline (history rows carry
    ``eligible_for_selection``) are kept, so runs from older protocols are never
    mixed in.  Modes without such results are omitted.
    """
    base = json.loads((root / "base_model/result.json").read_text(encoding="utf-8"))
    hard = json.loads((root / "initial_hard_examples.json").read_text(encoding="utf-8"))
    experiments: list[dict[str, Any]] = []
    skipped = 0
    for path in sorted((root / "hard_example_correction").glob("*/repeat_*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        history = result.get("history", [])
        if result["mode"] != "baseline" and not (
            history and "eligible_for_selection" in history[0]
        ):
            skipped += 1
            continue
        experiments.append(result)
    present = {e["mode"] for e in experiments}
    specs = {"baseline": {"objective": "no_update", "trainable_scope": "none"}}
    configs = {
        "baseline": {
            "rounds": 0, "epochs_per_round": 0, "learning_rate": 0.0,
            "weight_decay": 0.0, "hard_weight": 1.0, "center_weight": 0.0,
            "mmd_weight": 0.0, "target_size": 32,
        }
    }
    for e in experiments:
        specs[e["mode"]] = e["mode_spec"]
        configs[e["mode"]] = e["mode_hyperparameters"]
    print(f"Loaded {len(experiments)} current-protocol runs; skipped {skipped} stale runs")
    return {
        "threshold": base["threshold"],
        "initial_hard_manifest": {
            "count": hard["count"],
            "known_count": hard["known_count"],
            "unknown_count": hard["unknown_count"],
        },
        "before": base["before"],
        "modes": [m for m in MODE_ORDER if m == "baseline" or m in present],
        "mode_specs": specs,
        "mode_configs": configs,
        "experiments": experiments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "summary", nargs="?", type=Path, default=DEFAULT_INPUT,
        help="summary.json, or a run folder (son-tung-binary-hard-example) to rebuild from result.json files",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def std(values: Sequence[float]) -> float:
    """Population std, matching ``np.std`` used by ``aggregate_rows``."""
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def confusion(metrics: dict[str, Any]) -> dict[str, int]:
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "errors": fp + fn}


def run_metrics(
    correction: dict[str, Any],
    validation: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, float]:
    """Flatten one run (or the baseline) into the metrics shown in the report."""
    full_test = test["full"]
    return {
        "correction_rate": correction["correction_rate"],
        "retention_rate": correction["retention_rate"],
        "new_error_count": correction["new_error_count"],
        "current_hard_count": correction["current_hard_count"],
        "train_accuracy": correction["current_train_accuracy"],
        "val_bacc": validation["full"]["balanced_accuracy"],
        "test_accuracy": full_test["accuracy"],
        "test_precision": full_test["precision_unknown"],
        "test_recall": full_test["unknown_recall"],
        "test_f1": full_test["f1_unknown"],
        "test_bacc": full_test["balanced_accuracy"],
        "test_known_recall": full_test["known_recall"],
        "test_unknown_recall": full_test["unknown_recall"],
        **{f"test_{key}": value for key, value in confusion(full_test).items()},
    }


# (group, label, key, direction, number format)
METRIC_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    ("Correction", "Sửa lỗi (baseline-hard được sửa)", "correction_rate", HIGHER, PCT),
    ("Correction", "Giữ mẫu đúng (retention)", "retention_rate", HIGHER, PCT),
    ("Correction", "Lỗi mới (mẫu đúng bị làm sai)", "new_error_count", LOWER, NUM),
    ("Correction", "Lỗi train còn lại", "current_hard_count", LOWER, NUM),
    ("Correction", "Train accuracy", "train_accuracy", HIGHER, PCT),
    ("Validation", "Validation Balanced Accuracy", "val_bacc", HIGHER, PCT),
    ("Phân loại", "Accuracy", "test_accuracy", HIGHER, PCT),
    ("Phân loại", "Precision (UNKNOWN positive)", "test_precision", HIGHER, PCT),
    ("Phân loại", "Recall (UNKNOWN positive)", "test_recall", HIGHER, PCT),
    ("Phân loại", "F1 (UNKNOWN positive)", "test_f1", HIGHER, PCT),
    ("Phân loại", "Balanced Accuracy", "test_bacc", HIGHER, PCT),
    ("Retain", "Known Recall (Swizerland)", "test_known_recall", HIGHER, PCT),
    ("Retain", "Unknown Recall (Masque 500)", "test_unknown_recall", HIGHER, PCT),
    ("Confusion", "TN: KNOWN dự đoán KNOWN", "test_tn", HIGHER, NUM),
    ("Confusion", "FP: KNOWN dự đoán UNKNOWN", "test_fp", LOWER, NUM),
    ("Confusion", "FN: UNKNOWN dự đoán KNOWN", "test_fn", LOWER, NUM),
    ("Confusion", "TP: UNKNOWN dự đoán UNKNOWN", "test_tp", HIGHER, NUM),
    ("Confusion", "Tổng lỗi (FP + FN)", "test_errors", LOWER, NUM),
    ("Runtime", "Thời gian mỗi repeat (s)", "runtime_seconds", LOWER, NUM),
)


def collect(summary: dict[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    before = summary["before"]
    baseline = run_metrics(before["correction"], before["validation"], before["test"])
    per_mode: dict[str, dict[str, Any]] = {}
    for experiment in summary["experiments"]:
        mode = experiment["mode"]
        metrics = run_metrics(
            experiment["after_correction"],
            experiment["after_validation"],
            experiment["after_test"],
        )
        metrics["runtime_seconds"] = experiment["runtime_seconds"]
        history = experiment.get("history", [])
        metrics["rounds_run"] = len(history)
        metrics["selected_round"] = next(
            (row["round"] for row in history if row.get("selected_for_evaluation")), 0
        )
        entry = per_mode.setdefault(mode, {"runs": [], "experiment": experiment})
        entry["runs"].append(metrics)
    if "baseline" not in per_mode:
        # Baseline is the unchanged checkpoint: reuse its pre-correction metrics.
        synthetic = dict(baseline, runtime_seconds=0.0, rounds_run=0, selected_round=0)
        per_mode["baseline"] = {"runs": [synthetic], "experiment": None}
    baseline["runtime_seconds"] = per_mode["baseline"]["runs"][0]["runtime_seconds"]
    for entry in per_mode.values():
        runs = entry["runs"]
        entry["mean"] = {key: mean([run[key] for run in runs]) for key in runs[0]}
        entry["std"] = {key: std([run[key] for run in runs]) for key in runs[0]}
    return baseline, per_mode


def style_header(ws, row: int, first_col: int, values: Sequence[str]) -> None:
    for offset, value in enumerate(values):
        cell = ws.cell(row=row, column=first_col + offset, value=value)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = CENTER


def style_body(cell, stripe: int, number_format: str | None = None) -> None:
    cell.font = BODY_FONT
    cell.fill = STRIPE_FILLS[stripe % 2]
    cell.border = BODY_BORDER
    cell.alignment = MIDDLE
    if number_format:
        cell.number_format = number_format


def write_title(ws, title: str, subtitle: str) -> None:
    ws.sheet_view.showGridLines = False
    ws["A2"] = title
    ws["A2"].font = TITLE_FONT
    ws["A2"].alignment = MIDDLE
    ws["A3"] = subtitle
    ws["A3"].font = SUBTITLE_FONT
    ws["A3"].alignment = MIDDLE
    ws.row_dimensions[2].height = 24
    ws.row_dimensions[3].height = 18


def set_widths(ws, widths: Sequence[float]) -> None:
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width


SUMMARY_COLUMNS: tuple[tuple[str, str | None, str], ...] = (
    ("Mode", None, "General"),
    ("Scope", None, "General"),
    ("Repeats", None, "General"),
    ("Round được chọn", "selected_round", "0.00"),
    ("Sửa lỗi", "correction_rate", PCT),
    ("Giữ mẫu đúng", "retention_rate", PCT),
    ("Lỗi mới", "new_error_count", NUM),
    ("Lỗi train còn lại", "current_hard_count", NUM),
    ("Accuracy", "test_accuracy", PCT),
    ("Balanced Accuracy", "test_bacc", PCT),
    ("Known Recall", "test_known_recall", PCT),
    ("Unknown Recall", "test_unknown_recall", PCT),
    ("FP", "test_fp", NUM),
    ("FN", "test_fn", NUM),
    ("Tổng lỗi", "test_errors", NUM),
    ("Delta lỗi", None, NUM),
    ("Giảm lỗi so baseline", None, PCT),
    ("Runtime (s)", "runtime_seconds", NUM),
)


def build_summary_sheet(
    ws, summary: dict[str, Any], baseline: dict[str, float], per_mode: dict[str, dict[str, Any]]
) -> None:
    threshold = summary["threshold"]
    hard = summary["initial_hard_manifest"]
    write_title(
        ws,
        "Binary Hard-Example Correction Benchmark",
        (
            f"Positive class=UNKNOWN | Threshold={threshold:.2f} | "
            f"Baseline-hard={hard['count']} (KNOWN={hard['known_count']}, "
            f"UNKNOWN={hard['unknown_count']}) | Metric test = mean theo repeat"
        ),
    )
    header_row, first_row = 6, 7
    style_header(ws, header_row, 1, [column[0] for column in SUMMARY_COLUMNS])
    ws.row_dimensions[header_row].height = 30
    errors_col = get_column_letter(
        [column[0] for column in SUMMARY_COLUMNS].index("Tổng lỗi") + 1
    )
    delta_col = get_column_letter(
        [column[0] for column in SUMMARY_COLUMNS].index("Delta lỗi") + 1
    )
    base_ref = f"${errors_col}${first_row}"
    modes = [mode for mode in summary["modes"] if mode in per_mode]
    for stripe, mode in enumerate(modes):
        row = first_row + stripe
        entry = per_mode[mode]
        values = baseline if mode == "baseline" else entry["mean"]
        for col, (header, key, fmt) in enumerate(SUMMARY_COLUMNS, start=1):
            if header == "Mode":
                value: Any = DISPLAY_NAMES.get(mode, mode)
            elif header == "Scope":
                value = summary["mode_specs"][mode]["trainable_scope"]
            elif header == "Repeats":
                value = len(entry["runs"])
            elif header == "Delta lỗi":
                value = f"={errors_col}{row}-{base_ref}"
            elif header == "Giảm lỗi so baseline":
                value = f'=IF({base_ref}=0,"",-{delta_col}{row}/{base_ref})'
            elif key in ("selected_round", "runtime_seconds"):
                value = entry["mean"][key]
            else:
                value = values[key]
            style_body(ws.cell(row=row, column=col, value=value), stripe, fmt)
    last_row = first_row + len(modes) - 1
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(SUMMARY_COLUMNS))}{last_row}"
    ws.freeze_panes = f"A{first_row}"

    notes_row = last_row + 3
    style_header(ws, notes_row, 1, ["Ghi chú", "Ý nghĩa", "Chiều tốt"])
    notes = (
        ("Giá trị", "Method là mean theo repeat; baseline có 1 run.", ""),
        ("Sửa lỗi", "Tỷ lệ baseline-hard trên train được dự đoán đúng sau correction.", "Tăng"),
        ("Giữ mẫu đúng", "Tỷ lệ mẫu baseline đúng vẫn còn đúng.", "Tăng"),
        ("Lỗi mới", "Số mẫu baseline đúng bị correction làm sai.", "Giảm"),
        ("Metric dương", "UNKNOWN. Confusion matrix: [[TN, FP], [FN, TP]].", "Tăng"),
        ("FP", "KNOWN -> UNKNOWN", "Giảm"),
        ("FN", "UNKNOWN -> KNOWN", "Giảm"),
        ("TP", "UNKNOWN -> UNKNOWN", "Tăng"),
    )
    for offset, note in enumerate(notes, start=1):
        for col, value in enumerate(note, start=1):
            cell = ws.cell(row=notes_row + offset, column=col, value=value)
            cell.font = BODY_FONT
            cell.border = BODY_BORDER
            cell.alignment = MIDDLE
    set_widths(ws, (22, 14, 10, 15, 12, 14, 11, 15, 12, 17, 14, 15, 11, 11, 11, 11, 20, 13))
    ws.column_dimensions["B"].width = 24


def verdict_formula(row: int) -> str:
    return (
        f'=IF(ABS(G{row})<0.0000001,"Không đổi",'
        f'IF(C{row}="{HIGHER}",IF(G{row}>0,"Tốt hơn","Kém hơn"),'
        f'IF(C{row}="{LOWER}",IF(G{row}<0,"Tốt hơn","Kém hơn"),"Tham khảo")))'
    )


def write_context_block(
    ws, first_col: int, header: tuple[str, str], rows: Sequence[tuple[str, Any, str]]
) -> None:
    style_header(ws, 6, first_col, header)
    for offset, (label, value, fmt) in enumerate(rows, start=7):
        label_cell = ws.cell(row=offset, column=first_col, value=label)
        label_cell.font = LABEL_FONT
        label_cell.fill = LABEL_FILL
        label_cell.alignment = MIDDLE
        value_cell = ws.cell(row=offset, column=first_col + 1, value=value)
        value_cell.font = BODY_FONT
        value_cell.alignment = MIDDLE
        value_cell.number_format = fmt


def build_mode_sheet(
    ws, mode: str, summary: dict[str, Any], baseline: dict[str, float], entry: dict[str, Any]
) -> None:
    spec = summary["mode_specs"][mode]
    config = summary["mode_configs"][mode]
    runs = entry["runs"]
    write_title(
        ws,
        f"{DISPLAY_NAMES.get(mode, mode)}: đánh giá so với Baseline",
        f"Objective={spec['objective']} | Scope={spec['trainable_scope']} | {len(runs)} repeat",
    )
    hard = summary["initial_hard_manifest"]
    m, s = entry["mean"], entry["std"]
    write_context_block(
        ws,
        1,
        ("Run context", "Giá trị"),
        (
            ("Baseline-hard count", hard["count"], "0"),
            ("Threshold UNKNOWN", summary["threshold"], "0.00"),
            ("Positive class", "UNKNOWN", "General"),
            ("Repeats", len(runs), "0"),
            ("Round thực chạy (mean ± std)", f"{m['rounds_run']:.2f} ± {s['rounds_run']:.2f}", "General"),
            ("Round được chọn (mean ± std)", f"{m['selected_round']:.2f} ± {s['selected_round']:.2f}", "General"),
            # Recorded since the momentum / tolerance fix; older runs show N/A.
            ("Hard-weight momentum", summary.get("hard_weight_momentum", "N/A"), "0.00"),
            ("Selection BAcc tolerance", summary.get("selection_bacc_tolerance", "N/A"), "0.000"),
        ),
    )
    write_context_block(
        ws,
        4,
        ("Hyperparameter", "Giá trị"),
        (
            ("Rounds (tối đa)", config["rounds"], "0"),
            ("Epochs / round", config["epochs_per_round"], "0"),
            ("Learning rate", config["learning_rate"], "0.000000"),
            ("Weight decay", config["weight_decay"], "0.000000"),
            ("Hard weight", config["hard_weight"], "0.00"),
            ("Center weight", config["center_weight"], "0.000"),
            ("MMD weight", config["mmd_weight"], "0.000"),
            ("Reference target size", config["target_size"], "0"),
        ),
    )

    header_row, first_row = 21, 22
    style_header(
        ws,
        header_row,
        1,
        ("Nhóm", "Metric", "Chiều đánh giá", "Baseline", "Method mean", "Std", "Delta", "% thay đổi", "Kết luận"),
    )
    ws.row_dimensions[header_row].height = 28
    for stripe, (group, label, key, direction, fmt) in enumerate(METRIC_ROWS):
        row = first_row + stripe
        values = (
            group,
            label,
            direction,
            baseline[key],
            m[key],
            s[key],
            f"=E{row}-D{row}",
            f'=IF(D{row}=0,"",G{row}/ABS(D{row}))',
            verdict_formula(row),
        )
        formats = (None, None, None, fmt, fmt, fmt, fmt, PCT, None)
        for col, (value, number_format) in enumerate(zip(values, formats), start=1):
            style_body(ws.cell(row=row, column=col, value=value), stripe, number_format)
    last_row = first_row + len(METRIC_ROWS) - 1
    ws.auto_filter.ref = f"A{header_row}:I{last_row}"
    ws.freeze_panes = f"A{first_row}"
    verdict_range = f"I{first_row}:I{last_row}"
    ws.conditional_formatting.add(
        verdict_range,
        FormulaRule(
            formula=[f'$I{first_row}="Tốt hơn"'],
            font=Font(color="375623"),
            fill=PatternFill("solid", bgColor="E2F0D9"),
        ),
    )
    ws.conditional_formatting.add(
        verdict_range,
        FormulaRule(
            formula=[f'$I{first_row}="Kém hơn"'],
            font=Font(color="9C0006"),
            fill=PatternFill("solid", bgColor="FCE4D6"),
        ),
    )
    ws.conditional_formatting.add(
        f"G{first_row}:G{last_row}",
        CellIsRule(operator="greaterThan", formula=["0"], fill=PatternFill("solid", bgColor="D9EAF7")),
    )
    set_widths(ws, (30, 34, 17, 27, 18, 13, 13, 14, 14))


def sheet_title(mode: str) -> str:
    return DISPLAY_NAMES.get(mode, mode)[:31]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.summary.is_dir():
        summary = load_from_results(args.summary)
    else:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    baseline, per_mode = collect(summary)
    workbook = Workbook()
    build_summary_sheet(workbook.active, summary, baseline, per_mode)
    workbook.active.title = "Summary"
    for mode in summary["modes"]:
        if mode in per_mode:
            build_mode_sheet(
                workbook.create_sheet(sheet_title(mode)), mode, summary, baseline, per_mode[mode]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
