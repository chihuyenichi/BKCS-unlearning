#!/usr/bin/env python3
"""Convert head-sweep summaries (rows only) into the standard summary layout, so that
generate_benchmark_report.py / analyze_paired.py produce the usual workbook.

Confusion counts come from the recalls (10,000 test rows per class); train accuracy is
1 - current_hard_count / TRAIN_ROWS.  The "Baseline" row is the mean of --baseline-mode.

    python head_to_standard.py OUT.json --baseline-mode h256__base SUMMARY.json [SUMMARY.json ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEST_PER_CLASS = 10_000
TRAIN_ROWS = 68_000
VAL_ROWS = 12_000


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def slice_metrics(known_recall: float, unknown_recall: float, bacc: float) -> dict:
    tn = round(known_recall * TEST_PER_CLASS)
    tp = round(unknown_recall * TEST_PER_CLASS)
    fp, fn = TEST_PER_CLASS - tn, TEST_PER_CLASS - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * unknown_recall / (precision + unknown_recall) if precision + unknown_recall else 0.0
    return {
        "accuracy": (tn + tp) / (2 * TEST_PER_CLASS),
        "precision_unknown": precision,
        "known_recall": known_recall,
        "unknown_recall": unknown_recall,
        "f1_unknown": f1,
        "balanced_accuracy": bacc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def run_record(row: dict) -> tuple[dict, dict, dict]:
    correction = {
        "correction_rate": row["correction_rate"],
        "retention_rate": row["retention_rate"],
        "new_error_count": row["new_error_count"],
        "current_hard_count": row["current_hard_count"],
        "current_train_accuracy": 1.0 - row["current_hard_count"] / TRAIN_ROWS,
    }
    validation = {"full": {"balanced_accuracy": row["validation_balanced_accuracy"]}}
    test = {"full": slice_metrics(row["test_known_recall"], row["test_unknown_recall"], row["test_balanced_accuracy"])}
    return correction, validation, test


def convert(summaries: list[dict], baseline_mode: str) -> dict:
    rows, details = [], {}
    for summary in summaries:
        rows += summary["benchmark_rows"]
        for d in summary["details"]:
            details[(d["mode"], d["repeat"])] = d
    config = summaries[0]["correction_config"]
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    base_rows = [r for r in rows if r["mode"] == baseline_mode]
    parts = [run_record(r) for r in base_rows]
    mean_row = {k: mean(r[k] for r in base_rows) for k in base_rows[0] if isinstance(base_rows[0][k], (int, float)) and k != "repeat"}
    before_c, before_v, before_t = run_record(mean_row)
    before_t["full"]["confusion_matrix"] = [[mean(p[2]["full"]["confusion_matrix"][i][j] for p in parts) for j in range(2)] for i in range(2)]
    experiments, specs, configs = [], {}, {}
    for mode in modes:
        specs[mode] = {
            "objective": "none (own base head, no update)" if mode.endswith("__base") else "uniform_cross_entropy_control",
            "trainable_scope": "binary_mlp" if not mode.endswith("__base") else "none",
        }
        configs[mode] = dict(config) if not mode.endswith("__base") else {**config, "rounds": 0, "epochs_per_round": 0, "learning_rate": 0.0, "weight_decay": 0.0}
    for r in rows:
        c, v, t = run_record(r)
        d = details.get((r["mode"], r["repeat"]), {})
        history = [{"round": d["selected_round"], "selected_for_evaluation": True}] if d.get("selected_round") else []
        history += [{"round": i + 1} for i in range(max(0, d.get("rounds_run", 0) - len(history)))]
        experiments.append({
            "mode": r["mode"], "repeat": r["repeat"], "runtime_seconds": r["runtime_seconds"],
            "after_correction": c, "after_validation": v, "after_test": t, "history": history,
        })
    hard_mean = round(mean(r["initial_hard_count"] for r in base_rows))
    return {
        "pipeline": "head-sweep (converted)",
        "threshold": mean(d["threshold"] for d in details.values() if "threshold" in d),
        "initial_hard_manifest": {"count": hard_mean, "known_count": "n/a", "unknown_count": "n/a"},
        "before": {"correction": before_c, "validation": before_v, "test": before_t},
        "modes": ["baseline"] + [m for m in modes],
        "mode_specs": {"baseline": {"objective": f"mean of {baseline_mode}", "trainable_scope": "none"}, **specs},
        "mode_configs": {"baseline": {**config, "rounds": 0, "epochs_per_round": 0, "learning_rate": 0.0, "weight_decay": 0.0}, **configs},
        "experiments": experiments,
        "benchmark_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--baseline-mode", default="h256__base")
    args = parser.parse_args()
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in args.summaries]
    args.output.write_text(json.dumps(convert(summaries, args.baseline_mode)), encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
