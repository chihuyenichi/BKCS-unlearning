# ====================================================================================
# Experiment extension: shared runner.  Appended to pipeline.py by build_experiments.py;
# everything below runs inside the pipeline namespace and may redefine its functions.
# ====================================================================================


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Optimizer of a correction branch (build_experiments.py routes run_correction_mode here)."""
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


@dataclass
class Family:
    """A model together with the bundles it consumes (rows aligned across families)."""

    base_model: nn.Module
    train: FeatureBundle
    validation: FeatureBundle
    test: FeatureBundle
    encoder_frozen: bool = True


def default_extractor(
    train_records: Sequence[CsvRecord],
    validation_records: Sequence[CsvRecord],
    test_records: Sequence[CsvRecord],
    encoder: RawPacketEncoder,
    device: torch.device,
    signature_root: str,
) -> tuple[FeatureBundle, FeatureBundle, FeatureBundle, dict[str, Any]]:
    cache_dir = CACHE_ROOT
    train_bundle = extract_features(
        train_records, encoder, device, cache_dir / "train.pt", signature_root + ":train"
    )
    validation_bundle = extract_features(
        validation_records, encoder, device, cache_dir / "validation.pt", signature_root + ":validation"
    )
    test_bundle = extract_features(
        test_records, encoder, device, cache_dir / "test.pt", signature_root + ":test"
    )
    return train_bundle, validation_bundle, test_bundle, {}


def prepare_experiment_data(
    device: torch.device,
    extractor: Any,
) -> dict[str, Any]:
    """Datasets, splits, frozen encoder and the three feature bundles."""
    known_dir = resolve_dataset_dir(*KNOWN_DATASET_NAMES)
    unknown_dir = resolve_dataset_dir(*UNKNOWN_DATASET_NAMES)
    checkpoint_path = resolve_unique_file(ENCODER_CHECKPOINT_NAME)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("KNOWN:", known_dir)
    print("UNKNOWN:", unknown_dir)
    print("Encoder:", checkpoint_path)
    source_train, source_test, dataset_signatures = build_records(
        known_dir, unknown_dir, TRAIN_CAP_PER_ORIGINAL_CLASS, SEED
    )
    train_records, validation_records, test_records = split_records(
        source_train, source_test, SEED, VALID_RATIO
    )
    encoder, encoder_info = load_pretrained_encoder(checkpoint_path, device)
    split_payload = {
        "seed": SEED,
        "valid_ratio": VALID_RATIO,
        "train_cap_per_original_class": TRAIN_CAP_PER_ORIGINAL_CLASS,
        "dataset_signatures": dataset_signatures,
        "train_ids": [row.instance_id for row in train_records],
        "validation_ids": [row.instance_id for row in validation_records],
        "test_ids": [row.instance_id for row in test_records],
    }
    write_json(OUTPUT_ROOT / "split_manifest.json", split_payload)
    signature_root = hashlib.sha256(
        json.dumps({"split": split_payload, "encoder": encoder_info["sha256"]}, sort_keys=True).encode()
    ).hexdigest()
    print("Extracting frozen encoder features...")
    train_bundle, validation_bundle, test_bundle, extras = extractor(
        train_records, validation_records, test_records, encoder, device, signature_root
    )
    return {
        "encoder": encoder,
        "encoder_info": encoder_info,
        "dataset_signatures": dataset_signatures,
        "split_payload": split_payload,
        "train": train_bundle,
        "validation": validation_bundle,
        "test": test_bundle,
        "extras": extras,
    }


def run_experiment(
    *,
    pipeline_name: str,
    modes: Sequence[str],
    mode_family: dict[str, str],
    repeats_by_mode: dict[str, int],
    budget_seconds: float,
    extractor: Any = None,
    build_families: Any = None,
    require_cuda: bool = True,
    notes: dict[str, Any] | None = None,
) -> None:
    """The pipeline.main() flow, repeat-major, with a wall-clock budget and incremental output.

    Seeds are SEED + repeat exactly as in pipeline.main(), so a mode that exists in both
    runs reproduces the earlier numbers.  Repeat-major order keeps the paired design intact
    when the budget stops the run early: every finished repeat contains the same modes."""
    global MODES
    started_at = time.perf_counter()
    modes = tuple(modes)
    MODES = modes
    set_seed(SEED)
    if not KAGGLE_INPUT_ROOT.is_dir():
        raise RuntimeError("This pipeline expects Kaggle inputs under /kaggle/input")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Enable a Kaggle GPU before running this experiment")
    print("device:", device, "| torch threads:", torch.get_num_threads(), "| budget_s:", budget_seconds)

    data = prepare_experiment_data(device, extractor or default_extractor)
    train_bundle, validation_bundle, test_bundle = data["train"], data["validation"], data["test"]
    encoder_info, dataset_signatures = data["encoder_info"], data["dataset_signatures"]
    print(f"data ready after {time.perf_counter() - started_at:.0f}s")

    set_seed(SEED)
    base_model, base_history = train_base(train_bundle, validation_bundle, device)
    threshold, threshold_sweep = select_threshold(base_model, validation_bundle, device)
    initial_hard_mask, initial_correct_mask, initial_hard_rows = describe_initial_hard_examples(
        base_model, train_bundle, device, threshold
    )
    print("Baseline hard examples:", int(initial_hard_mask.sum()), "/", len(train_bundle))
    before = {
        "correction": correction_metrics(
            base_model, train_bundle, initial_hard_mask, initial_correct_mask, device, threshold
        ),
        "validation": evaluate_slices(base_model, validation_bundle, device, threshold),
        "test": evaluate_slices(base_model, test_bundle, device, threshold),
    }
    base_dir = OUTPUT_ROOT / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_to_cpu(base_model.state_dict()),
            "model_config": {"in_dim": ENCODER_DIM, "hidden_dim": 256, "classes": 2},
            "unknown_threshold": threshold,
            "encoder": encoder_info,
            "split": data["split_payload"],
        },
        base_dir / "best_model.pt",
    )
    base_checkpoint_sha256 = sha256_file(base_dir / "best_model.pt")
    write_json(
        base_dir / "result.json",
        {
            "history": base_history,
            "threshold_sweep": threshold_sweep,
            "threshold": threshold,
            "before": before,
            "train_binary_counts": dict(Counter(train_bundle.labels.tolist())),
            "base_sampling": "balanced_binary_batches_1_to_1",
        },
    )
    manifest = {
        "threshold": threshold,
        "count": len(initial_hard_rows),
        "known_count": sum(row["binary_target"] == KNOWN for row in initial_hard_rows),
        "unknown_count": sum(row["binary_target"] == UNKNOWN for row in initial_hard_rows),
        "instances": initial_hard_rows,
    }
    write_json(OUTPUT_ROOT / "initial_hard_examples.json", manifest)

    families: dict[str, Family] = {
        "default": Family(base_model, train_bundle, validation_bundle, test_bundle, True)
    }
    if build_families is not None:
        families.update(
            build_families(
                {
                    "device": device,
                    "base_model": base_model,
                    "threshold": threshold,
                    "initial_hard_mask": initial_hard_mask,
                    "data": data,
                }
            )
        )
    for mode in modes:
        if mode_family.get(mode, "default") not in families:
            raise KeyError(f"mode {mode} needs family {mode_family.get(mode)}")

    output_dir = OUTPUT_ROOT / "hard_example_correction"
    experiments: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    runtimes: dict[str, list[float]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []

    def write_summary(status: str) -> None:
        finished = [mode for mode in modes if any(row["mode"] == mode for row in benchmark_rows)]
        summary = {
            "pipeline": pipeline_name,
            "reference": "reference/unlearning-kf.ipynb",
            "task": "binary KNOWN/UNKNOWN correction without relabeling",
            "threshold": threshold,
            "positive_class": "UNKNOWN",
            "encoder_frozen": all(families[mode_family.get(m, 'default')].encoder_frozen for m in finished),
            "hard_weight_momentum": HARD_WEIGHT_MOMENTUM,
            "selection_bacc_tolerance": SELECTION_BACC_TOLERANCE,
            "encoder": encoder_info,
            "dataset_signatures": dataset_signatures,
            "initial_hard_manifest": manifest,
            "before": before,
            "modes": finished,
            "mode_specs": {
                mode: {
                    **MODE_SPECS[mode],
                    "labels_changed": False,
                    "encoder_frozen": families[mode_family.get(mode, "default")].encoder_frozen,
                }
                for mode in finished
            },
            "mode_hyperparameters": MODE_CONFIGS,
            "mode_configs": MODE_CONFIGS,
            "experiments": experiments,
            "benchmark_rows": benchmark_rows,
            "benchmark_summary": aggregate_rows(benchmark_rows),
            "experiment": {
                "status": status,
                "budget_seconds": budget_seconds,
                "elapsed_seconds": time.perf_counter() - started_at,
                "repeats_by_mode": {mode: repeats_by_mode.get(mode, REPEATS) for mode in modes},
                "skipped_runs": skipped,
                **(notes or {}),
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "benchmark_runs.csv", benchmark_rows)
        write_csv(output_dir / "benchmark_summary.csv", summary["benchmark_summary"])

    max_repeats = max([1] + [repeats_by_mode.get(mode, REPEATS) for mode in modes if mode != "baseline"])
    for repeat in range(max_repeats):
        for mode in modes:
            if mode == "baseline" and repeat > 0:
                continue
            if mode != "baseline" and repeat >= repeats_by_mode.get(mode, REPEATS):
                continue
            remaining = budget_seconds - (time.perf_counter() - started_at)
            expected = float(np.mean(runtimes[mode])) if runtimes[mode] else 0.0
            if remaining <= 0 or (runtimes[mode] and expected > remaining):
                skipped.append(
                    {"mode": mode, "repeat": repeat, "remaining_seconds": remaining, "expected_seconds": expected}
                )
                print(f"[budget] skip {mode} repeat {repeat}: remaining {remaining:.0f}s, expected {expected:.0f}s")
                continue
            family = families[mode_family.get(mode, "default")]
            set_seed(SEED + repeat)
            started = time.perf_counter()
            branch, history = run_correction_mode(
                family.base_model,
                mode,
                family.train,
                initial_hard_mask,
                initial_correct_mask,
                family.validation,
                device,
                threshold,
                SEED + repeat,
            )
            after_correction = correction_metrics(
                branch, family.train, initial_hard_mask, initial_correct_mask, device, threshold
            )
            after_validation = evaluate_slices(branch, family.validation, device, threshold)
            after_test = evaluate_slices(branch, family.test, device, threshold)
            runtime_seconds = time.perf_counter() - started
            runtimes[mode].append(runtime_seconds)
            result = {
                "mode": mode,
                "mode_spec": {
                    **MODE_SPECS[mode],
                    "labels_changed": False,
                    "encoder_frozen": family.encoder_frozen,
                },
                "mode_hyperparameters": MODE_CONFIGS[mode],
                "repeat": repeat,
                "optimization_seed": SEED + repeat,
                "runtime_seconds": runtime_seconds,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "initial_hard_count": len(initial_hard_rows),
                "before_correction": before["correction"],
                "before_validation": before["validation"],
                "before_test": before["test"],
                "after_correction": after_correction,
                "after_validation": after_validation,
                "after_test": after_test,
                "history": history,
            }
            write_json(output_dir / mode / f"repeat_{repeat}" / "result.json", result)
            experiments.append(result)
            full_test = after_test["full"]
            full_validation = after_validation["full"]
            benchmark_rows.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "initial_hard_count": after_correction["initial_hard_count"],
                    "correction_rate": after_correction["correction_rate"],
                    "retention_rate": after_correction["retention_rate"],
                    "new_error_count": after_correction["new_error_count"],
                    "current_hard_count": after_correction["current_hard_count"],
                    "validation_balanced_accuracy": full_validation["balanced_accuracy"],
                    "test_accuracy": full_test["accuracy"],
                    "test_balanced_accuracy": full_test["balanced_accuracy"],
                    "test_known_recall": full_test["known_recall"],
                    "test_unknown_recall": full_test["unknown_recall"],
                    "runtime_seconds": runtime_seconds,
                }
            )
            print(
                f"[done] {mode} repeat {repeat}: test_bacc={full_test['balanced_accuracy']:.4f} "
                f"runtime={runtime_seconds:.0f}s elapsed={time.perf_counter() - started_at:.0f}s"
            )
            del branch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_summary("running")
    write_summary("stopped_by_budget" if skipped else "complete")
    print("Saved:", output_dir)
