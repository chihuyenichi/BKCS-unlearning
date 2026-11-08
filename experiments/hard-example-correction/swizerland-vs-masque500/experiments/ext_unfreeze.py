# ====================================================================================
# Experiment B: unfreeze the last part of the encoder.
#   level "fc"     : the projection layer (flattened block-4 conv output -> 256-d embedding)
#   level "block4" : block 4 (conv4, conv4_4, batch_norm4, max_pool_4) plus the projection layer
# Blocks 1-3 stay frozen and run only once; their output is cached (fp16).
# ====================================================================================


def encoder_front(encoder: RawPacketEncoder, features: Tensor, mask: Tensor) -> Tensor:
    """Blocks 1-3 of RawPacketEncoder._forward_convs (eval mode: dropout is a no-op)."""
    x = (features * mask.unsqueeze(-1)).transpose(1, 2)
    x = F.elu(encoder.conv1(encoder._pad(x)))
    x = F.elu(encoder.batch_norm1(encoder.conv1_1(encoder._pad(x))))
    x = encoder.dropout1(encoder.max_pool_1(encoder._pad(x)))
    x = F.relu(encoder.conv2(encoder._pad(x)))
    x = F.relu(encoder.batch_norm2(encoder.conv2_2(encoder._pad(x))))
    x = encoder.dropout2(encoder.max_pool_2(encoder._pad(x)))
    x = F.relu(encoder.conv3(encoder._pad(x)))
    x = F.relu(encoder.batch_norm3(encoder.conv3_3(encoder._pad(x))))
    return encoder.dropout3(encoder.max_pool_3(encoder._pad(x)))


class EncoderTail(nn.Module):
    """Trainable copy of block 4 (optional) and the projection layer of a pretrained encoder."""

    def __init__(self, encoder: RawPacketEncoder, with_convs: bool):
        super().__init__()
        self.with_convs = with_convs
        if with_convs:
            self.conv4 = copy.deepcopy(encoder.conv4)
            self.conv4_4 = copy.deepcopy(encoder.conv4_4)
            self.batch_norm4 = copy.deepcopy(encoder.batch_norm4)
            self.max_pool_4 = copy.deepcopy(encoder.max_pool_4)
        self.fc = copy.deepcopy(encoder.fc)
        for parameter in self.parameters():
            parameter.requires_grad = True

    def convs(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv4(RawPacketEncoder._pad(x)))
        x = F.relu(self.batch_norm4(self.conv4_4(RawPacketEncoder._pad(x))))
        return self.max_pool_4(RawPacketEncoder._pad(x))  # dropout4 is a no-op in eval mode

    def project(self, flat: Tensor) -> Tensor:
        return self.fc(flat)


class UnfrozenModel(nn.Module):
    """EncoderTail followed by the BinaryMLP; consumes cached features, returns (logits, hidden).

    The tail always runs in eval mode (BatchNorm uses its running statistics, no dropout) and
    only its parameters are fine-tuned, so the embedding of a sample changes only through the
    weights and not through the composition of the balanced batch."""

    def __init__(self, tail: EncoderTail, head: BinaryMLP):
        super().__init__()
        self.tail = tail
        self.head = head

    def embed(self, x: Tensor) -> Tensor:
        x = x.float()
        if self.tail.with_convs:
            x = self.tail.convs(x).flatten(1)
        return self.tail.project(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.head(self.embed(x))

    def train(self, mode: bool = True) -> "UnfrozenModel":
        super().train(mode)
        self.tail.eval()
        return self


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:  # noqa: F811
    lr = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    if isinstance(model, UnfrozenModel):
        # No weight decay on the pretrained tail: it would pull SupCon weights toward zero.
        return torch.optim.Adam(
            [
                {"params": list(model.head.parameters()), "lr": lr, "weight_decay": weight_decay},
                {"params": list(model.tail.parameters()), "lr": float(config["encoder_lr"]), "weight_decay": 0.0},
            ]
        )
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def build_reference_bank(  # noqa: F811
    model: nn.Module,
    features: Tensor,
    true_labels: Tensor,
    device: torch.device,
    threshold: float,
) -> dict[int, Tensor]:
    """Same result as pipeline.build_reference_bank with one forward pass instead of two."""
    model.eval()
    probabilities: list[Tensor] = []
    hiddens: list[Tensor] = []
    for start in range(0, len(features), EMBEDDING_BATCH_SIZE):
        logits, hidden = model(features[start : start + EMBEDDING_BATCH_SIZE].to(device))
        probabilities.append(torch.softmax(logits, dim=1)[:, UNKNOWN].cpu())
        hiddens.append(hidden.cpu())
    predicted = (torch.cat(probabilities) >= threshold).long()
    hidden_all = torch.cat(hiddens)
    bank: dict[int, Tensor] = {}
    for label in (KNOWN, UNKNOWN):
        mask = (true_labels == label) & (predicted == label)
        if int(mask.sum()) >= 2:
            bank[label] = hidden_all[mask]
    return bank


@torch.no_grad()
def extract_block3_cache(
    records: Sequence[CsvRecord], encoder: RawPacketEncoder, device: torch.device
) -> FeatureBundle:
    """Run blocks 1-3 once over the raw rows and keep the output as fp16."""
    dataset = RawCsvDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=RAW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    encoder.eval()
    maps: list[Tensor] = []
    labels: list[Tensor] = []
    for batch_index, (features, mask, binary_labels, _) in enumerate(loader, start=1):
        front = encoder_front(encoder, features.to(device), mask.to(device))
        maps.append(front.half().cpu())
        labels.append(binary_labels.long().cpu())
        if batch_index % 100 == 0:
            print(f"  block-3 cache {min(batch_index * RAW_BATCH_SIZE, len(dataset))}/{len(dataset)}")
    return FeatureBundle(
        features=torch.cat(maps),
        labels=torch.cat(labels),
        instance_ids=[record.instance_id for record in records],
        original_labels=[record.original_label for record in records],
        sources=[record.source for record in records],
    )


@torch.no_grad()
def apply_in_chunks(function: Any, features: Tensor, device: torch.device, chunk: int = 256) -> Tensor:
    return torch.cat([function(features[start : start + chunk].to(device).float()).cpu() for start in range(0, len(features), chunk)])


def unfreeze_extractor(
    train_records: Sequence[CsvRecord],
    validation_records: Sequence[CsvRecord],
    test_records: Sequence[CsvRecord],
    encoder: RawPacketEncoder,
    device: torch.device,
    signature_root: str,
) -> tuple[FeatureBundle, FeatureBundle, FeatureBundle, dict[str, Any]]:
    """One pass over the raw rows -> block-3 caches; the 256-d embeddings come from the cache,
    so the frozen baseline and the unfrozen models start from exactly the same function."""
    cache = {
        "train": extract_block3_cache(train_records, encoder, device),
        "validation": extract_block3_cache(validation_records, encoder, device),
        "test": extract_block3_cache(test_records, encoder, device),
    }
    tail = EncoderTail(encoder, with_convs=True).to(device).eval()
    embedded: dict[str, FeatureBundle] = {}
    for name, bundle in cache.items():
        embedding = apply_in_chunks(lambda x: tail.project(tail.convs(x).flatten(1)), bundle.features, device)
        embedded[name] = FeatureBundle(
            features=embedding,
            labels=bundle.labels,
            instance_ids=bundle.instance_ids,
            original_labels=bundle.original_labels,
            sources=bundle.sources,
        )
    return embedded["train"], embedded["validation"], embedded["test"], {"cache3": cache, "encoder": encoder}


def build_unfreeze_families(context: dict[str, Any]) -> dict[str, Family]:
    device: torch.device = context["device"]
    base_model: BinaryMLP = context["base_model"]
    extras = context["data"]["extras"]
    encoder: RawPacketEncoder = extras["encoder"]
    cache: dict[str, FeatureBundle] = extras["cache3"]
    reference = context["data"]["train"]
    families: dict[str, Family] = {}
    for level in ("fc", "block4"):
        tail = EncoderTail(encoder, with_convs=(level == "block4")).to(device)
        model = UnfrozenModel(tail, copy.deepcopy(base_model)).to(device).eval()
        if level == "fc":
            tail_for_flat = EncoderTail(encoder, with_convs=True).to(device).eval()
            bundles = {}
            for name, bundle in cache.items():
                flat = apply_in_chunks(lambda x: tail_for_flat.convs(x).flatten(1), bundle.features, device).half()
                bundles[name] = FeatureBundle(flat, bundle.labels, bundle.instance_ids, bundle.original_labels, bundle.sources)
            del tail_for_flat
        else:
            bundles = cache
        # The unfrozen model must start from the baseline's function.
        probe = slice(0, min(2000, len(reference)))
        expected = predict_probability(base_model, reference.features[probe], device)
        actual = predict_probability(model, bundles["train"].features[probe], device)
        gap = float((expected - actual).abs().max())
        print(f"[unfreeze:{level}] max |p_base - p_unfrozen| on {probe.stop} rows = {gap:.2e}")
        if gap > 1e-3:
            raise RuntimeError(f"unfrozen model ({level}) does not reproduce the baseline: {gap}")
        families[level] = Family(model, bundles["train"], bundles["validation"], bundles["test"], encoder_frozen=False)
    return families


UNFREEZE_REFERENCE_MODES = ("baseline", "uniform_finetune", "hard_aware_mkmmd_ramp_l010")
UNFREEZE_MODE_FAMILY = {
    "fc_uniform_lr2e5": "fc",
    "fc_ramp_l010_lr2e5": "fc",
    "b4_uniform_lr5e6": "block4",
    "b4_uniform_lr2e5": "block4",
    "b4_ramp_l010_lr2e5": "block4",
}
UNFREEZE_RUN_MODES = UNFREEZE_REFERENCE_MODES + tuple(UNFREEZE_MODE_FAMILY)
HARD_AWARE_MKMMD_MODES = HARD_AWARE_MKMMD_MODES + ("fc_ramp_l010_lr2e5", "b4_ramp_l010_lr2e5")

for _name, _source, _lr, _label in (
    ("fc_uniform_lr2e5", "uniform_finetune", 2e-5, "projection layer"),
    ("fc_ramp_l010_lr2e5", "hard_aware_mkmmd_ramp_l010", 2e-5, "projection layer"),
    ("b4_uniform_lr5e6", "uniform_finetune", 5e-6, "block 4 + projection layer"),
    ("b4_uniform_lr2e5", "uniform_finetune", 2e-5, "block 4 + projection layer"),
    ("b4_ramp_l010_lr2e5", "hard_aware_mkmmd_ramp_l010", 2e-5, "block 4 + projection layer"),
):
    MODE_CONFIGS[_name] = {**MODE_CONFIGS[_source], "encoder_lr": _lr}
    MODE_SPECS[_name] = {
        "objective": MODE_SPECS[_source]["objective"] + "_with_unfrozen_encoder_tail",
        "trainable_scope": f"binary_mlp + encoder {_label}",
        "adaptation": f"{MODE_SPECS[_source]['adaptation']}; also fine-tunes the {_label} with lr {_lr:g} (eval-mode BatchNorm, no weight decay)",
    }

UNFREEZE_BUDGET_HOURS = 4.0


_pipeline_set_seed = set_seed


def set_seed(seed: int) -> None:  # noqa: F811
    """pipeline.set_seed, then let cuDNN pick the fastest convolution algorithms.

    The block-4 modes back-propagate through 1-D convolutions; deterministic algorithms are
    markedly slower and the run is budget-limited.  Only the weights of the unfrozen convolutions
    are affected by the non-determinism (last-bit differences), not the data order or splits."""
    _pipeline_set_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
# block-4 modes cost roughly 10-20 minutes per run on a T4, so they get fewer seeds; the budget decides.
UNFREEZE_REPEATS = {
    "fc_uniform_lr2e5": 10,
    "fc_ramp_l010_lr2e5": 10,
    "b4_uniform_lr5e6": 6,
    "b4_uniform_lr2e5": 6,
    "b4_ramp_l010_lr2e5": 6,
}


def main_unfreeze() -> None:
    run_experiment(
        pipeline_name="son-tung-binary-hard-example-unfreeze-v1",
        modes=UNFREEZE_RUN_MODES,
        mode_family=UNFREEZE_MODE_FAMILY,
        repeats_by_mode=UNFREEZE_REPEATS,
        budget_seconds=UNFREEZE_BUDGET_HOURS * 3600.0,
        extractor=unfreeze_extractor,
        build_families=build_unfreeze_families,
        notes={"name": "unfreeze", "levels": {"fc": "projection layer", "block4": "block 4 + projection layer"}},
    )
