# ====================================================================================
# Experiment C: Sharpness-Aware Minimization (SAM) on top of the correction modes.
# ====================================================================================
import contextlib


@contextlib.contextmanager
def frozen_batchnorm_statistics(model: nn.Module) -> Iterator[None]:
    """BatchNorm uses batch statistics but leaves its running averages untouched."""
    saved = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            tracked = module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None
            saved.append((module, module.momentum, tracked))
            module.momentum = 0.0
    try:
        yield
    finally:
        for module, momentum, tracked in saved:
            module.momentum = momentum
            if tracked is not None:
                module.num_batches_tracked.copy_(tracked)


def rng_snapshot(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cpu": torch.get_rng_state(), "generator": generator.get_state()}
    if device.type == "cuda":
        snapshot["cuda"] = torch.cuda.get_rng_state(device)
    return snapshot


def rng_restore(snapshot: dict[str, Any], generator: torch.Generator, device: torch.device) -> None:
    torch.set_rng_state(snapshot["cpu"])
    generator.set_state(snapshot["generator"])
    if "cuda" in snapshot:
        torch.cuda.set_rng_state(snapshot["cuda"], device)


def train_correction_epoch(
    model: BinaryMLP,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    true_labels: Tensor,
    sample_weights: Tensor,
    hard_mask: Tensor,
    bank: dict[int, Tensor],
    mode: str,
    config: dict[str, Any],
    mmd: MultipleKernelMaximumMeanDiscrepancy,
    device: torch.device,
    seed: int,
    hard_pool: dict[int, Tensor] | None = None,
    mmd_scale: float = 1.0,
) -> dict[str, float]:
    """pipeline.train_correction_epoch plus an optional SAM step (config["sam_rho"] > 0).

    SAM (Foret et al. 2021): take the gradient g of the loss, move to w + rho * g / ||g||,
    take the gradient there and apply it at w.  Dropout masks, the hard-pool draws and the
    MMD reference draws of the second pass are replayed from the first pass, and BatchNorm
    running averages are updated only once per step.  With sam_rho == 0 this function is
    numerically identical to the original."""
    per_class = int(config.get("mmd_min_hard_per_class", 0))
    push_weight = float(config.get("mmd_push_weight", 0.0))
    sam_rho = float(config.get("sam_rho", 0.0))
    hard_generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(features, true_labels, sample_weights, hard_mask)
    sampler = BalancedBinaryBatchSampler(true_labels.tolist(), EMBEDDING_BATCH_SIZE, seed)
    loader = DataLoader(dataset, batch_sampler=sampler)
    set_mlp_train_mode(model)
    sums = {"total": 0.0, "ce": 0.0, "regularizer": 0.0}
    if per_class > 0:
        sums.update({"mmd_intra": 0.0, "mmd_inter": 0.0})
    batches = 0
    for feature_batch, label_batch, weight_batch, hard_batch in loader:
        # BatchNorm needs at least two rows; duplicate only a singleton tail batch.
        if len(feature_batch) == 1:
            feature_batch = torch.cat((feature_batch, feature_batch), dim=0)
            label_batch = torch.cat((label_batch, label_batch), dim=0)
            weight_batch = torch.cat((weight_batch, weight_batch), dim=0)
            hard_batch = torch.cat((hard_batch, hard_batch), dim=0)
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)
        weight_batch = weight_batch.to(device)
        hard_batch = hard_batch.to(device)

        def compute_loss() -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
            logits, hidden = model(feature_batch)
            per_sample_ce = F.cross_entropy(logits, label_batch, reduction="none")
            ce_loss = (per_sample_ce * weight_batch).sum() / weight_batch.sum().clamp_min(1e-12)
            regularizer = hidden.new_zeros(())
            mmd_stats: dict[str, float] = {}
            if per_class > 0 and mode in HARD_AWARE_MKMMD_MODES:
                if hard_pool and mmd_scale > 0.0:
                    regularizer = hard_aware_mkmmd_regularizer(
                        model,
                        features,
                        hard_pool,
                        per_class,
                        bank,
                        mmd,
                        int(config["target_size"]),
                        hard_generator,
                        device,
                        push_weight=push_weight,
                        stats=mmd_stats,
                    )
            elif bool(hard_batch.any()) and mode == "hard_weighted_center":
                regularizer = center_regularizer(hidden, label_batch, hard_batch, bank, device)
            elif bool(hard_batch.any()) and mode == "hard_weighted_mkmmd":
                regularizer = mkmmd_regularizer(
                    hidden, label_batch, hard_batch, bank, mmd, int(config["target_size"]), device
                )
            coefficient = (
                float(config["center_weight"])
                if mode == "hard_weighted_center"
                else float(config["mmd_weight"])
            )
            if mode in HARD_AWARE_MKMMD_MODES:
                coefficient *= mmd_scale
            return ce_loss + coefficient * regularizer, ce_loss, regularizer, mmd_stats

        optimizer.zero_grad(set_to_none=True)
        snapshot = rng_snapshot(hard_generator, device) if sam_rho > 0.0 else None
        loss, ce_loss, regularizer, mmd_stats = compute_loss()
        loss.backward()
        if sam_rho > 0.0:
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params))
            scale = sam_rho / (grad_norm + 1e-12)
            perturbations: list[Tensor] = []
            with torch.no_grad():
                for p in params:
                    step = p.grad * scale
                    p.add_(step)
                    perturbations.append(step)
            optimizer.zero_grad(set_to_none=True)
            rng_restore(snapshot, hard_generator, device)
            with frozen_batchnorm_statistics(model):
                perturbed_loss, _, _, _ = compute_loss()
                perturbed_loss.backward()
            with torch.no_grad():
                for p, step in zip(params, perturbations):
                    p.sub_(step)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        sums["total"] += loss.item()
        sums["ce"] += ce_loss.item()
        sums["regularizer"] += regularizer.item()
        if per_class > 0:
            sums["mmd_intra"] += mmd_stats.get("intra", 0.0)
            sums["mmd_inter"] += mmd_stats.get("inter", 0.0)
        batches += 1
    return {key: value / max(batches, 1) for key, value in sums.items()}


SAM_MODES = (
    "uniform_sam_r05",
    "uniform_sam_r10",
    "hard_aware_ramp_l010_sam_r05",
)
SAM_REFERENCE_MODES = ("baseline", "uniform_finetune", "hard_aware_mkmmd_ramp_l010")
SAM_RUN_MODES = SAM_REFERENCE_MODES + SAM_MODES
HARD_AWARE_MKMMD_MODES = HARD_AWARE_MKMMD_MODES + ("hard_aware_ramp_l010_sam_r05",)

MODE_CONFIGS["uniform_sam_r05"] = {**MODE_CONFIGS["uniform_finetune"], "sam_rho": 0.05}
MODE_CONFIGS["uniform_sam_r10"] = {**MODE_CONFIGS["uniform_finetune"], "sam_rho": 0.10}
MODE_CONFIGS["hard_aware_ramp_l010_sam_r05"] = {
    **MODE_CONFIGS["hard_aware_mkmmd_ramp_l010"],
    "sam_rho": 0.05,
}
MODE_SPECS["uniform_sam_r05"] = {
    "objective": "uniform_cross_entropy_control_with_sam",
    "trainable_scope": "binary_mlp",
    "adaptation": "uniform_finetune with sharpness-aware minimization, rho 0.05",
}
MODE_SPECS["uniform_sam_r10"] = {
    "objective": "uniform_cross_entropy_control_with_sam",
    "trainable_scope": "binary_mlp",
    "adaptation": "uniform_finetune with sharpness-aware minimization, rho 0.10",
}
MODE_SPECS["hard_aware_ramp_l010_sam_r05"] = {
    "objective": "hard_example_weighted_ce_plus_hard_aware_class_conditional_mkmmd_with_sam",
    "trainable_scope": "binary_mlp",
    "adaptation": "hard-aware MK-MMD (weight 0.10, 5-round warm-up, 64-row reference) with SAM, rho 0.05",
}

SAM_BUDGET_HOURS = 3.5


def main_sam() -> None:
    run_experiment(
        pipeline_name="son-tung-binary-hard-example-sam-v1",
        modes=SAM_RUN_MODES,
        mode_family={},
        repeats_by_mode={},
        budget_seconds=SAM_BUDGET_HOURS * 3600.0,
        notes={"name": "sam", "sam_modes": list(SAM_MODES)},
    )
