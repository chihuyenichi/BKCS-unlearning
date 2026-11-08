#!/usr/bin/env python3
"""Paired (same-seed) comparison of correction modes; optionally appends a 'Paired' sheet to the xlsx.

For every pair (mode, reference) the per-seed differences of validation / test balanced accuracy are
summarised by mean, standard deviation, a bootstrap 95% interval of the mean, win/loss counts and an
exact sign-flip permutation p-value (two-sided).  No scipy needed.

    python analyze_paired.py outputs/<run>/summary.json --xlsx outputs/<report>.xlsx
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

REFERENCES = ("baseline", "uniform_finetune", "hard_weighted_ce", "hard_weighted_mkmmd", "hard_aware_mkmmd", "hard_aware_mkmmd_ramp_l010")
METRICS = (("validation_balanced_accuracy", "Validation BAcc"), ("test_balanced_accuracy", "Test BAcc"))


def mean(values):
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5


def sign_flip_p(diffs):
    """Exact two-sided p-value of mean(diff) != 0 under random sign flips (n <= 16), else Monte Carlo."""
    observed = abs(mean(diffs))
    n = len(diffs)
    if n <= 16:
        hits = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            hits += abs(mean([s * d for s, d in zip(signs, diffs)])) >= observed - 1e-15
        return hits / total
    rng = random.Random(0)
    hits = sum(
        abs(mean([rng.choice((1, -1)) * d for d in diffs])) >= observed - 1e-15 for _ in range(20000)
    )
    return (hits + 1) / 20001


def bootstrap_ci(diffs, resamples=10000):
    rng = random.Random(0)
    means = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def collect(rows):
    table: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        table.setdefault(row["mode"], {})[int(row["repeat"])] = row
    return table


def compare(table, mode, ref, metric):
    runs, ref_runs = table[mode], table[ref]
    diffs = []
    for repeat, row in runs.items():
        ref_row = ref_runs.get(repeat, ref_runs.get(0) if len(ref_runs) == 1 else None)
        if ref_row is not None:
            diffs.append(float(row[metric]) - float(ref_row[metric]))
    if len(diffs) < 2:
        return None
    low, high = bootstrap_ci(diffs)
    return {
        "n": len(diffs), "mean": mean(diffs), "std": std(diffs), "ci_low": low, "ci_high": high,
        "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs), "p": sign_flip_p(diffs),
    }


def analyse(summary, references=REFERENCES):
    table = collect(summary["benchmark_rows"])
    out = []
    for mode in table:
        if mode == "baseline":
            continue
        for ref in references:
            if ref == mode or ref not in table:
                continue
            for metric, label in METRICS:
                result = compare(table, mode, ref, metric)
                if result:
                    out.append({"mode": mode, "reference": ref, "metric": label, **result})
    return out, table


def per_mode(table):
    rows = []
    for mode, runs in table.items():
        val = [float(r["validation_balanced_accuracy"]) for r in runs.values()]
        test = [float(r["test_balanced_accuracy"]) for r in runs.values()]
        rows.append((mode, len(runs), mean(val), std(val) if len(val) > 1 else 0.0, mean(test), std(test) if len(test) > 1 else 0.0))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary", type=Path)
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--refs", default=None, help="comma-separated reference modes (default: the built-in list)")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results, table = analyse(summary, tuple(args.refs.split(",")) if args.refs else REFERENCES)
    print(f"{'mode':24s} n  val_mean  val_std  test_mean  test_std")
    for mode, n, vm, vs, tm, ts in per_mode(table):
        print(f"{mode:24s} {n:<2d} {vm:.4f}   {vs:.4f}   {tm:.4f}    {ts:.4f}")
    print()
    for r in results:
        if r["metric"] != "Test BAcc":
            continue
        print(f"{r['mode']:24s} vs {r['reference']:20s} dTest {r['mean']:+.4f} (sd {r['std']:.4f}, 95%CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]) "
              f"W/L {r['wins']}/{r['losses']} p={r['p']:.3f}")
    if args.xlsx:
        from openpyxl import load_workbook
        from openpyxl.styles import Font

        wb = load_workbook(args.xlsx)
        if "Paired" in wb.sheetnames:
            del wb["Paired"]
        ws = wb.create_sheet("Paired", 1)
        bold = Font(bold=True, name="Arial")
        ws.append(["Mỗi mode"])
        ws["A1"].font = bold
        ws.append(["Mode", "Số lần chạy", "Val BAcc TB", "Val BAcc std", "Test BAcc TB", "Test BAcc std"])
        for cell in ws[2]:
            cell.font = bold
        for row in per_mode(table):
            ws.append(list(row))
        ws.append([])
        ws.append(["So sánh ghép cặp theo seed (hiệu mode - tham chiếu)"])
        ws.cell(ws.max_row, 1).font = bold
        ws.append(["Mode", "Tham chiếu", "Chỉ số", "n", "Hiệu TB", "Std hiệu", "CI95 dưới", "CI95 trên", "Thắng", "Thua", "p (sign-flip)"])
        for cell in ws[ws.max_row]:
            cell.font = bold
        for r in results:
            ws.append([r["mode"], r["reference"], r["metric"], r["n"], r["mean"], r["std"], r["ci_low"], r["ci_high"], r["wins"], r["losses"], r["p"]])
        ws.append([])
        ws.append(["CI95 là bootstrap phần trăm của hiệu trung bình; p là kiểm định hoán vị dấu chính xác hai phía. Baseline chỉ có 1 lần chạy nên hiệu so với baseline là mode_i - baseline."])
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 22
        ws.column_dimensions["C"].width = 18
        wb.save(args.xlsx)
        print("Appended sheet 'Paired' to", args.xlsx)


if __name__ == "__main__":
    main()
