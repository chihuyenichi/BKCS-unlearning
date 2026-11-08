# ====================================================================================
# Experiment A: head capacity / batch size / training length on the frozen embeddings.
# Every (configuration, seed) trains its own base head, so the spread of the baseline itself
# is measured instead of being taken from the single base model of the earlier runs.
# ====================================================================================


class FlexMLP(nn.Module):
    """BinaryMLP with any number of hidden layers.  hidden_dims=(256,) builds exactly BinaryMLP()."""

    def __init__(self, hidden_dims: Sequence[int] = (256,), in_dim: int = ENCODER_DIM, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(width, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            width = hidden_dim
        if not layers:  # linear probe on the raw embedding
            layers = [nn.Identity(), nn.Identity()]
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(width, NUM_CLASSES)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.feature_extractor[:-1](features)
        return self.fc(self.feature_extractor[-1](hidden)), hidden


# name -> (hidden_dims, base batch size, base max epochs, base patience)
HEAD_CONFIGS: dict[str, dict[str, Any]] = {
    "lin": {"hidden_dims": (), "batch": 128, "epochs": 100, "patience": 15},
    "h256": {"hidden_dims": (256,), "batch": 128, "epochs": 100, "patience": 15},  # the pipeline's head
    "h512": {"hidden_dims": (512,), "batch": 128, "epochs": 100, "patience": 15},
    "h256x2": {"hidden_dims": (256, 256), "batch": 128, "epochs": 100, "patience": 15},
    "h512x2": {"hidden_dims": (512, 512), "batch": 128, "epochs": 100, "patience": 15},
    "h256_b64": {"hidden_dims": (256,), "batch": 64, "epochs": 100, "patience": 15},
    "h256_b256": {"hidden_dims": (256,), "batch": 256, "epochs": 100, "patience": 15},
    "h256_e200": {"hidden_dims": (256,), "batch": 128, "epochs": 200, "patience": 30},
}
HEAD_REPEATS = 10
HEAD_BUDGET_HOURS = 8.5


def train_base_flex(
    config: dict[str, Any],
    seed: int,
    train_bundle: FeatureBundle,
    validation_bundle: FeatureBundle,
    device: torch.device,
) -> tuple[FlexMLP, dict[str, Any]]:
    """pipeline.train_base with the architecture, batch size and length of ``config`` and a seed.

    With the h256 configuration and seed == SEED it is the pipeline's own baseline."""
    set_seed(seed)
    model = FlexMLP(config["hidden_dims"]).to(device)
    sampler = BalancedBinaryBatchSampler(train_bundle.labels.tolist(), int(config["batch"]), seed)
    loader = DataLoader(TensorDataset(train_bundle.features, train_bundle.labels), batch_sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=BASE_WEIGHT_DECAY)
    best_state = state_to_cpu(model.state_dict())
    best_score, best_epoch, stale = -float("inf"), 0, 0
    epochs_run = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for features, labels in loader:
            if len(features) < 2:  # BatchNorm needs two rows
                continue
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            F.cross_entropy(logits, labels).backward()
            optimizer.step()
        epochs_run = epoch
        epoch_threshold, _ = select_threshold(model, validation_bundle, device)
        score = float(evaluate(model, validation_bundle, device, epoch_threshold)["balanced_accuracy"])
        if epoch % 25 == 0:
            print(f"  base epoch {epoch}/{config['epochs']}: val_bal_acc={score:.4f} (best {best_score:.4f} @ {best_epoch})")
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = state_to_cpu(model.state_dict())
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epochs_run, "best_validation_bacc": best_score}


def head_row(mode: str, repeat: int, correction: dict[str, Any], validation: dict[str, Any], test: dict[str, Any], runtime: float, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "repeat": repeat,
        "initial_hard_count": correction["initial_hard_count"],
        "correction_rate": correction["correction_rate"],
        "retention_rate": correction["retention_rate"],
        "new_error_count": correction["new_error_count"],
        "current_hard_count": correction["current_hard_count"],
        "validation_balanced_accuracy": validation["full"]["balanced_accuracy"],
        "test_accuracy": test["full"]["accuracy"],
        "test_balanced_accuracy": test["full"]["balanced_accuracy"],
        "test_known_recall": test["full"]["known_recall"],
        "test_unknown_recall": test["full"]["unknown_recall"],
        "runtime_seconds": runtime,
        **extra,
    }


def main_head() -> None:
    global MODES
    started_at = time.perf_counter()
    budget_seconds = HEAD_BUDGET_HOURS * 3600.0
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| torch threads:", torch.get_num_threads(), "| budget_s:", budget_seconds)
    data = prepare_experiment_data(device, default_extractor)
    train_bundle, validation_bundle, test_bundle = data["train"], data["validation"], data["test"]
    print(f"data ready after {time.perf_counter() - started_at:.0f}s")

    mode_names = [f"{name}__{kind}" for name in HEAD_CONFIGS for kind in ("base", "uniform")]
    MODES = tuple(mode_names)
    MODE_CONFIGS_LOCAL = {"uniform_finetune": MODE_CONFIGS["uniform_finetune"]}
    output_dir = OUTPUT_ROOT / "hard_example_correction"
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    runtimes: dict[str, list[float]] = defaultdict(list)

    def write_summary(status: str) -> None:
        summary = {
            "pipeline": "son-tung-binary-hard-example-head-sweep-v1",
            "encoder": data["encoder_info"],
            "dataset_signatures": data["dataset_signatures"],
            "head_configs": {name: {**cfg, "hidden_dims": list(cfg["hidden_dims"])} for name, cfg in HEAD_CONFIGS.items()},
            "correction_config": MODE_CONFIGS_LOCAL["uniform_finetune"],
            "modes": [mode for mode in mode_names if any(row["mode"] == mode for row in rows)],
            "benchmark_rows": rows,
            "benchmark_summary": aggregate_rows(rows),
            "details": details,
            "experiment": {
                "name": "head_sweep",
                "status": status,
                "budget_seconds": budget_seconds,
                "elapsed_seconds": time.perf_counter() - started_at,
                "repeats": HEAD_REPEATS,
                "skipped_runs": skipped,
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "benchmark_runs.csv", rows)
        write_csv(output_dir / "benchmark_summary.csv", summary["benchmark_summary"])

    for repeat in range(HEAD_REPEATS):
        for name, config in HEAD_CONFIGS.items():
            key = f"{name}__base"
            remaining = budget_seconds - (time.perf_counter() - started_at)
            expected = float(np.mean(runtimes[name])) if runtimes[name] else 0.0
            if remaining <= 0 or (runtimes[name] and expected > remaining):
                skipped.append({"config": name, "repeat": repeat, "remaining_seconds": remaining, "expected_seconds": expected})
                print(f"[budget] skip {name} repeat {repeat}")
                continue
            started = time.perf_counter()
            seed = SEED + repeat
            base, info = train_base_flex(config, seed, train_bundle, validation_bundle, device)
            threshold, _ = select_threshold(base, validation_bundle, device)
            hard_mask, correct_mask, _ = describe_initial_hard_examples(base, train_bundle, device, threshold)
            base_correction = correction_metrics(base, train_bundle, hard_mask, correct_mask, device, threshold)
            base_validation = evaluate_slices(base, validation_bundle, device, threshold)
            base_test = evaluate_slices(base, test_bundle, device, threshold)
            base_seconds = time.perf_counter() - started
            extra = {**info, "threshold": threshold, "parameters": int(sum(p.numel() for p in base.parameters()))}
            rows.append(head_row(key, repeat, base_correction, base_validation, base_test, base_seconds, {}))
            details.append({"mode": key, "repeat": repeat, **extra})

            set_seed(seed)
            started = time.perf_counter()
            branch, history = run_correction_mode(base, "uniform_finetune", train_bundle, hard_mask, correct_mask, validation_bundle, device, threshold, seed)
            correction = correction_metrics(branch, train_bundle, hard_mask, correct_mask, device, threshold)
            validation = evaluate_slices(branch, validation_bundle, device, threshold)
            test = evaluate_slices(branch, test_bundle, device, threshold)
            uniform_seconds = time.perf_counter() - started
            rows.append(head_row(f"{name}__uniform", repeat, correction, validation, test, uniform_seconds, {}))
            details.append(
                {
                    "mode": f"{name}__uniform",
                    "repeat": repeat,
                    "threshold": threshold,
                    "selected_round": next((h["round"] for h in history if h.get("selected_for_evaluation")), 0),
                    "rounds_run": len(history),
                }
            )
            runtimes[name].append(base_seconds + uniform_seconds)
            print(
                f"[done] {name} repeat {repeat}: base test_bacc={base_test['full']['balanced_accuracy']:.4f} "
                f"(best epoch {info['best_epoch']}/{info['epochs_run']}) -> uniform {test['full']['balanced_accuracy']:.4f} "
                f"| {base_seconds + uniform_seconds:.0f}s elapsed={time.perf_counter() - started_at:.0f}s"
            )
            del base, branch
            write_summary("running")
    write_summary("stopped_by_budget" if skipped else "complete")
    print("Saved:", output_dir)
