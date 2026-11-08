"""Binary hard-example correction pipeline adapted from reference/unlearning-kf.ipynb.

A frozen packet encoder maps AOL and 100 Top Search traffic to embeddings.  A
binary MLP first learns KNOWN/UNKNOWN classification.  Correction branches then
repeatedly find training samples that are still predicted incorrectly and
fine-tune the MLP, either uniformly or with a larger loss weight for those hard
examples.  Ground-truth labels are never changed and the test split is never
used for training, threshold selection, or model selection.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_TEMP_ROOT = Path("/kaggle/temp")
OUTPUT_ROOT = KAGGLE_WORKING_ROOT / "linh-san-binary-hard-example"
CACHE_ROOT = KAGGLE_TEMP_ROOT / "linh-san-binary-hard-example-cache"

KNOWN = 0
UNKNOWN = 1
NUM_CLASSES = 2
MAX_PACKETS = 10_000
RAW_FIELDS_PER_PACKET = 4
ROW_WIDTH = MAX_PACKETS * RAW_FIELDS_PER_PACKET
ENCODER_DIM = 256

# Main experiment controls.
SEED = 42
REPEATS = 3
VALID_RATIO = 0.15
TRAIN_CAP_PER_ORIGINAL_CLASS: int | None = 150
RAW_BATCH_SIZE = 16
EMBEDDING_BATCH_SIZE = 128
NUM_WORKERS = 2
BASE_EPOCHS = 50
BASE_LR = 1e-3
BASE_WEIGHT_DECAY = 1e-4
BASE_PATIENCE = 15
CORRECTION_PATIENCE = 5
# Corrected samples keep part of their extra weight in later rounds so the
# decision boundary does not swing back and forth between the two classes.
HARD_WEIGHT_MOMENTUM = 0.5
# A correction round is selectable only while validation BAcc stays this close
# to the baseline; round 0 (the unchanged baseline) is always a candidate.
SELECTION_BACC_TOLERANCE = 0.002
# Wide enough that the selected threshold is not pinned to a grid edge.
THRESHOLD_GRID = tuple(round(0.10 + 0.05 * step, 2) for step in range(17))  # 0.10..0.90
SAVE_BRANCH_MODELS = False
RUN_PIPELINE = True

MMD_ALPHAS = (0.001, 0.01, 0.1, 1.0)
MODES = (
    "baseline",
    "uniform_finetune",
    "hard_weighted_ce",
    "hard_weighted_center",
    "hard_weighted_mkmmd",
)

MODE_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "objective": "no_update",
        "trainable_scope": "none",
        "adaptation": "common baseline checkpoint without parameter updates",
    },
    "uniform_finetune": {
        "objective": "uniform_cross_entropy_control",
        "trainable_scope": "binary_mlp",
        "adaptation": "same correction budget with weight 1 for every sample",
    },
    "hard_weighted_ce": {
        "objective": "iterative_hard_example_weighted_cross_entropy",
        "trainable_scope": "binary_mlp",
        "adaptation": "currently misclassified samples receive a larger CE weight",
    },
    "hard_weighted_center": {
        "objective": "hard_example_weighted_ce_plus_center",
        "trainable_scope": "binary_mlp",
        "adaptation": "weighted CE plus same-class center alignment for hard samples",
    },
    "hard_weighted_mkmmd": {
        "objective": "hard_example_weighted_ce_plus_class_conditional_mkmmd",
        "trainable_scope": "binary_mlp",
        "adaptation": "weighted CE plus same-class distribution alignment for hard samples",
    },
}

MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "rounds": 0,
        "epochs_per_round": 0,
        "learning_rate": 0.0,
        "weight_decay": 0.0,
        "hard_weight": 1.0,
        "center_weight": 0.0,
        "mmd_weight": 0.0,
        "target_size": 32,
    },
    "uniform_finetune": {
        "rounds": 25,
        "epochs_per_round": 2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "hard_weight": 1.0,
        "center_weight": 0.0,
        "mmd_weight": 0.0,
        "target_size": 32,
    },
    "hard_weighted_ce": {
        "rounds": 25,
        "epochs_per_round": 2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "hard_weight": 5.0,
        "center_weight": 0.0,
        "mmd_weight": 0.0,
        "target_size": 32,
    },
    "hard_weighted_center": {
        "rounds": 25,
        "epochs_per_round": 2,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "hard_weight": 5.0,
        "center_weight": 0.02,
        "mmd_weight": 0.0,
        "target_size": 32,
    },
    "hard_weighted_mkmmd": {
        "rounds": 25,
        "epochs_per_round": 2,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "hard_weight": 5.0,
        "center_weight": 0.0,
        "mmd_weight": 0.10,
        "target_size": 32,
    },
}


@dataclass(frozen=True)
class CsvRecord:
    instance_id: str
    csv_path: str
    byte_offset: int
    line_number: int
    row_sha256: str
    source: str
    split_file: str
    original_label_id: int
    original_label: str
    binary_label: int


@dataclass
class FeatureBundle:
    features: Tensor
    labels: Tensor
    instance_ids: list[str]
    original_labels: list[str]
    sources: list[str]

    def __len__(self) -> int:
        return int(self.labels.numel())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_to_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Detach checkpoint tensors so saved models are portable outside CUDA."""
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def resolve_dataset_dir(folder_name: str) -> Path:
    matches = [
        path.resolve()
        for path in KAGGLE_INPUT_ROOT.rglob(folder_name)
        if path.is_dir()
        and {"train_data.csv", "test_data.csv", "label_mapping.csv"}.issubset(
            child.name for child in path.iterdir()
        )
    ]
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one dataset folder {folder_name!r}; found {matches}"
        )
    return matches[0]


def resolve_unique_file(filename: str) -> Path:
    matches = sorted(
        path.resolve()
        for path in KAGGLE_INPUT_ROOT.rglob(filename)
        if path.is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename!r}; found {matches}")
    return matches[0]


def read_label_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[int, str] = {}
    for row in rows:
        if "label_id" not in row or "label_name" not in row:
            raise ValueError(f"Invalid label mapping header: {path}")
        mapping[int(row["label_id"])] = row["label_name"]
    return mapping


def index_csv(
    path: Path,
    source: str,
    split_file: str,
    binary_label: int,
    label_mapping: dict[int, str],
) -> list[CsvRecord]:
    records: list[CsvRecord] = []
    with path.open("rb") as handle:
        line_number = 0
        while True:
            byte_offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            stripped = raw.rstrip(b"\r\n")
            first = stripped.split(b",", 1)[0]
            original_label_id = int(float(first))
            row_sha256 = hashlib.sha256(stripped).hexdigest()
            records.append(
                CsvRecord(
                    instance_id=(
                        f"{source}:{split_file}:{line_number}:{row_sha256[:12]}"
                    ),
                    csv_path=str(path.resolve()),
                    byte_offset=byte_offset,
                    line_number=line_number,
                    row_sha256=row_sha256,
                    source=source,
                    split_file=split_file,
                    original_label_id=original_label_id,
                    original_label=label_mapping.get(
                        original_label_id, f"class_{original_label_id}"
                    ),
                    binary_label=binary_label,
                )
            )
    return records


def cap_by_original_class(
    records: Sequence[CsvRecord], cap: int | None, seed: int
) -> list[CsvRecord]:
    if cap is None:
        return list(records)
    groups: dict[tuple[str, int], list[CsvRecord]] = defaultdict(list)
    for record in records:
        groups[(record.source, record.original_label_id)].append(record)
    rng = random.Random(seed)
    selected: list[CsvRecord] = []
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        selected.extend(values[:cap])
    rng.shuffle(selected)
    return selected


def build_records(
    aol_dir: Path, unknown_dir: Path, cap: int | None, seed: int
) -> tuple[list[CsvRecord], list[CsvRecord], dict[str, Any]]:
    aol_mapping = read_label_mapping(aol_dir / "label_mapping.csv")
    unknown_mapping = read_label_mapping(unknown_dir / "label_mapping.csv")
    source_train = index_csv(
        aol_dir / "train_data.csv", "AOL", "train_data.csv", KNOWN, aol_mapping
    ) + index_csv(
        unknown_dir / "train_data.csv",
        "100_TOPSEARCH",
        "train_data.csv",
        UNKNOWN,
        unknown_mapping,
    )
    source_train = cap_by_original_class(source_train, cap, seed)
    test = index_csv(
        aol_dir / "test_data.csv", "AOL", "test_data.csv", KNOWN, aol_mapping
    ) + index_csv(
        unknown_dir / "test_data.csv",
        "100_TOPSEARCH",
        "test_data.csv",
        UNKNOWN,
        unknown_mapping,
    )
    signatures = {
        str(path.resolve()): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in (
            aol_dir / "train_data.csv",
            aol_dir / "test_data.csv",
            unknown_dir / "train_data.csv",
            unknown_dir / "test_data.csv",
        )
    }
    return source_train, test, signatures


def split_records(
    source_train: Sequence[CsvRecord], test: Sequence[CsvRecord], seed: int, valid_ratio: float
) -> tuple[list[CsvRecord], list[CsvRecord], list[CsvRecord]]:
    groups: dict[tuple[str, int], list[CsvRecord]] = defaultdict(list)
    for record in source_train:
        groups[(record.source, record.original_label_id)].append(record)
    rng = random.Random(seed)
    train: list[CsvRecord] = []
    validation: list[CsvRecord] = []
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        n_valid = max(1, round(len(values) * valid_ratio)) if len(values) >= 3 else 0
        validation.extend(values[:n_valid])
        train.extend(values[n_valid:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation, list(test)


class RawCsvDataset(Dataset):
    def __init__(self, records: Sequence[CsvRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, int]:
        record = self.records[index]
        with Path(record.csv_path).open("rb") as handle:
            handle.seek(record.byte_offset)
            raw = handle.readline().rstrip(b"\r\n")
        if hashlib.sha256(raw).hexdigest() != record.row_sha256:
            raise ValueError(f"CSV row changed: {record.instance_id}")
        values = np.fromstring(raw.decode("ascii"), sep=",", dtype=np.float32)
        if values.size != ROW_WIDTH:
            raise ValueError(
                f"Expected {ROW_WIDTH} values, got {values.size}: {record.instance_id}"
            )
        packets = values.reshape(MAX_PACKETS, RAW_FIELDS_PER_PACKET)
        features_np = packets[:, 1:]
        mask_np = features_np[:, 2] != 0
        features = torch.from_numpy(features_np.copy())
        mask = torch.from_numpy(mask_np.copy())
        return features, mask, torch.tensor(record.binary_label), index


class RawPacketEncoder(nn.Module):
    """DF-style encoder compatible with the reference AOL checkpoint."""

    def __init__(self, max_packets: int = MAX_PACKETS, hidden_size: int = ENCODER_DIM):
        super().__init__()
        self.max_packets = max_packets
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(3, 32, 8)
        self.conv1_1 = nn.Conv1d(32, 32, 8)
        self.conv2 = nn.Conv1d(32, 64, 8)
        self.conv2_2 = nn.Conv1d(64, 64, 8)
        self.conv3 = nn.Conv1d(64, 128, 8)
        self.conv3_3 = nn.Conv1d(128, 128, 8)
        self.conv4 = nn.Conv1d(128, 256, 8)
        self.conv4_4 = nn.Conv1d(256, 256, 8)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.max_pool_1 = nn.MaxPool1d(8, stride=4)
        self.max_pool_2 = nn.MaxPool1d(8, stride=4)
        self.max_pool_3 = nn.MaxPool1d(8, stride=4)
        self.max_pool_4 = nn.MaxPool1d(8, stride=4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        with torch.no_grad():
            flat_dim = self._forward_convs(torch.zeros(1, 3, max_packets)).flatten(1).shape[1]
        self.fc = nn.Linear(flat_dim, hidden_size)

    @staticmethod
    def _pad(x: Tensor) -> Tensor:
        return F.pad(x, (3, 4))

    def _forward_convs(self, x: Tensor) -> Tensor:
        x = F.elu(self.conv1(self._pad(x)))
        x = F.elu(self.batch_norm1(self.conv1_1(self._pad(x))))
        x = self.dropout1(self.max_pool_1(self._pad(x)))
        x = F.relu(self.conv2(self._pad(x)))
        x = F.relu(self.batch_norm2(self.conv2_2(self._pad(x))))
        x = self.dropout2(self.max_pool_2(self._pad(x)))
        x = F.relu(self.conv3(self._pad(x)))
        x = F.relu(self.batch_norm3(self.conv3_3(self._pad(x))))
        x = self.dropout3(self.max_pool_3(self._pad(x)))
        x = F.relu(self.conv4(self._pad(x)))
        x = F.relu(self.batch_norm4(self.conv4_4(self._pad(x))))
        return self.dropout4(self.max_pool_4(self._pad(x)))

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            features = features * mask.unsqueeze(-1)
        return self.fc(self._forward_convs(features.transpose(1, 2)).flatten(1))


def load_pretrained_encoder(
    path: Path, device: torch.device
) -> tuple[RawPacketEncoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    max_packets = int(config.get("max_packets", MAX_PACKETS))
    hidden_size = int(config.get("hidden_size", ENCODER_DIM))
    if max_packets != MAX_PACKETS or hidden_size != ENCODER_DIM:
        raise ValueError(
            f"Checkpoint config mismatch: max_packets={max_packets}, hidden_size={hidden_size}"
        )
    source_state = checkpoint.get("model_state_dict", checkpoint)
    encoder = RawPacketEncoder(max_packets=max_packets, hidden_size=hidden_size)
    target_names = set(encoder.state_dict())
    encoder_state: dict[str, Tensor] = {}
    for raw_name, value in source_state.items():
        name = raw_name.removeprefix("module.")
        if name.startswith("encoder."):
            name = name.removeprefix("encoder.")
        if name in target_names:
            encoder_state[name] = value
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Encoder checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": checkpoint.get("epoch"),
        "model_config": config,
    }


@torch.no_grad()
def extract_features(
    records: Sequence[CsvRecord],
    encoder: RawPacketEncoder,
    device: torch.device,
    cache_path: Path,
    cache_signature: str,
) -> FeatureBundle:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("signature") == cache_signature:
            return FeatureBundle(
                features=payload["features"],
                labels=payload["labels"],
                instance_ids=payload["instance_ids"],
                original_labels=payload["original_labels"],
                sources=payload["sources"],
            )
    dataset = RawCsvDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=RAW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    for batch_index, (features, mask, binary_labels, _) in enumerate(loader, start=1):
        embeddings.append(
            encoder(
                features.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            ).cpu()
        )
        labels.append(binary_labels.long().cpu())
        if batch_index % 50 == 0:
            print(f"  extracted {min(batch_index * RAW_BATCH_SIZE, len(dataset))}/{len(dataset)}")
    bundle = FeatureBundle(
        features=torch.cat(embeddings),
        labels=torch.cat(labels),
        instance_ids=[record.instance_id for record in records],
        original_labels=[record.original_label for record in records],
        sources=[record.source for record in records],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": cache_signature,
            "features": bundle.features,
            "labels": bundle.labels,
            "instance_ids": bundle.instance_ids,
            "original_labels": bundle.original_labels,
            "sources": bundle.sources,
        },
        cache_path,
    )
    return bundle


class BinaryMLP(nn.Module):
    """Reference one-hidden-layer classifier, changed to two output classes."""

    def __init__(self, in_dim: int = ENCODER_DIM, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        # Regularizers compare batch hidden features with an eval-mode bank, so
        # return the pre-dropout activation to keep both on the same scale.
        hidden = self.feature_extractor[:-1](features)
        return self.fc(self.feature_extractor[-1](hidden)), hidden


class BalancedBinaryBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: Sequence[int], batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced batches require an even batch size >= 2")
        self.indices = {
            label: [index for index, value in enumerate(labels) if value == label]
            for label in (KNOWN, UNKNOWN)
        }
        if any(not values for values in self.indices.values()):
            raise ValueError(f"Both binary classes are required: {self.indices}")
        self.pairs_per_batch = batch_size // 2
        self.samples_per_class = max(len(values) for values in self.indices.values())
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def _draw(values: Sequence[int], count: int, rng: random.Random) -> list[int]:
        result: list[int] = []
        while len(result) < count:
            cycle = list(values)
            rng.shuffle(cycle)
            result.extend(cycle)
        return result[:count]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        known = self._draw(self.indices[KNOWN], self.samples_per_class, rng)
        unknown = self._draw(self.indices[UNKNOWN], self.samples_per_class, rng)
        pairs = list(zip(known, unknown))
        rng.shuffle(pairs)
        for start in range(0, len(pairs), self.pairs_per_batch):
            selected_pairs = pairs[start : start + self.pairs_per_batch]
            batch = [index for pair in selected_pairs for index in pair]
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.samples_per_class / self.pairs_per_batch)


@torch.no_grad()
def predict_probability(
    model: BinaryMLP,
    features: Tensor,
    device: torch.device,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> Tensor:
    model.eval()
    values: list[Tensor] = []
    for start in range(0, len(features), batch_size):
        logits, _ = model(features[start : start + batch_size].to(device))
        values.append(torch.softmax(logits, dim=1)[:, UNKNOWN].cpu())
    return torch.cat(values)


def metrics_from_probability(
    labels: Tensor, probability: Tensor, threshold: float
) -> dict[str, Any]:
    predicted = (probability >= threshold).long()
    labels = labels.long().cpu()
    tn = int(((labels == KNOWN) & (predicted == KNOWN)).sum())
    fp = int(((labels == KNOWN) & (predicted == UNKNOWN)).sum())
    fn = int(((labels == UNKNOWN) & (predicted == KNOWN)).sum())
    tp = int(((labels == UNKNOWN) & (predicted == UNKNOWN)).sum())
    total = len(labels)
    known_recall = tn / (tn + fp) if tn + fp else float("nan")
    unknown_recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * unknown_recall / (precision + unknown_recall)
        if precision + unknown_recall
        else 0.0
    )
    valid_recalls = [x for x in (known_recall, unknown_recall) if not math.isnan(x)]
    return {
        "accuracy": (tn + tp) / total if total else float("nan"),
        "precision_unknown": precision,
        "known_recall": known_recall,
        "unknown_recall": unknown_recall,
        "f1_unknown": f1,
        "balanced_accuracy": (
            sum(valid_recalls) / len(valid_recalls)
            if valid_recalls
            else float("nan")
        ),
        "unknown_threshold": threshold,
        "total": total,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate(
    model: BinaryMLP,
    bundle: FeatureBundle,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    probability = predict_probability(model, bundle.features, device)
    result = metrics_from_probability(bundle.labels, probability, threshold)
    logits_loss = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bundle), EMBEDDING_BATCH_SIZE):
            labels = bundle.labels[start : start + EMBEDDING_BATCH_SIZE].to(device)
            logits, _ = model(bundle.features[start : start + EMBEDDING_BATCH_SIZE].to(device))
            logits_loss += F.cross_entropy(logits, labels, reduction="sum").item()
    result["loss"] = logits_loss / max(len(bundle), 1)
    return result


def subset_bundle(bundle: FeatureBundle, indices: Sequence[int]) -> FeatureBundle:
    index_tensor = torch.tensor(list(indices), dtype=torch.long)
    return FeatureBundle(
        features=bundle.features[index_tensor],
        labels=bundle.labels[index_tensor],
        instance_ids=[bundle.instance_ids[index] for index in indices],
        original_labels=[bundle.original_labels[index] for index in indices],
        sources=[bundle.sources[index] for index in indices],
    )


def evaluate_slices(
    model: BinaryMLP,
    bundle: FeatureBundle,
    device: torch.device,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Report the full binary split plus its two source-defined classes."""
    slices = {
        "full": list(range(len(bundle))),
        "known_aol": [i for i, source in enumerate(bundle.sources) if source == "AOL"],
        "unknown_100_topsearch": [
            i for i, source in enumerate(bundle.sources) if source == "100_TOPSEARCH"
        ],
    }
    return {
        name: evaluate(model, subset_bundle(bundle, indices), device, threshold)
        for name, indices in slices.items()
        if indices
    }


@torch.no_grad()
def prediction_state(
    model: BinaryMLP,
    bundle: FeatureBundle,
    device: torch.device,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    probability = predict_probability(model, bundle.features, device)
    predicted = (probability >= threshold).long()
    correct = predicted.eq(bundle.labels.cpu())
    return probability, predicted, correct


def correction_metrics(
    model: BinaryMLP,
    train_bundle: FeatureBundle,
    initial_hard_mask: Tensor,
    initial_correct_mask: Tensor,
    device: torch.device,
    threshold: float,
) -> dict[str, float | int]:
    """Measure corrections and regressions against the fixed baseline partition."""
    _, _, correct = prediction_state(model, train_bundle, device, threshold)
    initial_hard_count = int(initial_hard_mask.sum())
    initial_correct_count = int(initial_correct_mask.sum())
    corrected_count = int((correct & initial_hard_mask).sum())
    retained_correct_count = int((correct & initial_correct_mask).sum())
    current_hard_count = int((~correct).sum())
    return {
        "initial_hard_count": initial_hard_count,
        "corrected_count": corrected_count,
        "correction_rate": (
            corrected_count / initial_hard_count if initial_hard_count else float("nan")
        ),
        "initial_correct_count": initial_correct_count,
        "retained_correct_count": retained_correct_count,
        "retention_rate": (
            retained_correct_count / initial_correct_count
            if initial_correct_count
            else float("nan")
        ),
        "new_error_count": initial_correct_count - retained_correct_count,
        "current_hard_count": current_hard_count,
        "current_train_accuracy": 1.0 - current_hard_count / max(len(train_bundle), 1),
    }


def train_base(
    train_bundle: FeatureBundle,
    validation_bundle: FeatureBundle,
    device: torch.device,
) -> tuple[BinaryMLP, list[dict[str, float]]]:
    model = BinaryMLP().to(device)
    sampler = BalancedBinaryBatchSampler(train_bundle.labels.tolist(), EMBEDDING_BATCH_SIZE, SEED)
    loader = DataLoader(
        TensorDataset(train_bundle.features, train_bundle.labels), batch_sampler=sampler
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=BASE_LR, weight_decay=BASE_WEIGHT_DECAY
    )
    best_state = state_to_cpu(model.state_dict())
    best_score = -float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, BASE_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        epoch_threshold, _ = select_threshold(model, validation_bundle, device)
        validation = evaluate(model, validation_bundle, device, epoch_threshold)
        score = float(validation["balanced_accuracy"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(seen, 1),
                "validation_balanced_accuracy": score,
                "validation_known_recall": float(validation["known_recall"]),
                "validation_unknown_recall": float(validation["unknown_recall"]),
                "validation_threshold": epoch_threshold,
            }
        )
        print(f"base epoch {epoch}/{BASE_EPOCHS}: val_bal_acc={score:.4f}")
        if score > best_score:
            best_score = score
            best_state = state_to_cpu(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= BASE_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def select_threshold(
    model: BinaryMLP,
    validation: FeatureBundle,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    probability = predict_probability(model, validation.features, device)
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLD_GRID:
        metrics = metrics_from_probability(validation.labels, probability, threshold)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "known_recall": float(metrics["known_recall"]),
                "unknown_recall": float(metrics["unknown_recall"]),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["known_recall"],
            row["accuracy"],
        ),
    )
    return float(best["threshold"]), rows


def describe_initial_hard_examples(
    model: BinaryMLP,
    train_bundle: FeatureBundle,
    device: torch.device,
    threshold: float,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    """Freeze the baseline error set used for before/after correction metrics."""
    probability, predicted, correct = prediction_state(
        model, train_bundle, device, threshold
    )
    hard_mask = ~correct
    rows: list[dict[str, Any]] = []
    for index in hard_mask.nonzero(as_tuple=False).flatten().tolist():
        rows.append(
            {
                "train_index": index,
                "instance_id": train_bundle.instance_ids[index],
                "source": train_bundle.sources[index],
                "original_label": train_bundle.original_labels[index],
                "binary_target": int(train_bundle.labels[index]),
                "baseline_prediction": int(predicted[index]),
                "baseline_probability_unknown": float(probability[index]),
            }
        )
    return hard_mask, correct, rows


class GaussianKernel(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True)
        distance = (norm + norm.transpose(0, 1) - 2.0 * x @ x.transpose(0, 1)).clamp_min(0.0)
        return torch.exp(-self.alpha * distance)


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """Non-linear unequal-batch MK-MMD adapted from the reference notebook."""

    def __init__(self, alphas: Sequence[float]):
        super().__init__()
        self.kernels = nn.ModuleList([GaussianKernel(alpha) for alpha in alphas])
        self.index_cache: dict[tuple[int, int, str], Tensor] = {}

    def _index_matrix(self, n_source: int, n_target: int, device: torch.device) -> Tensor:
        key = (n_source, n_target, str(device))
        if key in self.index_cache:
            return self.index_cache[key]
        matrix = torch.zeros(n_source + n_target, n_source + n_target, device=device)
        if n_source > 1:
            source = 1.0 - torch.eye(n_source, device=device)
            matrix[:n_source, :n_source] = source / (n_source * (n_source - 1))
        if n_target > 1:
            target = 1.0 - torch.eye(n_target, device=device)
            matrix[n_source:, n_source:] = target / (n_target * (n_target - 1))
        matrix[:n_source, n_source:] = -1.0 / (n_source * n_target)
        matrix[n_source:, :n_source] = -1.0 / (n_source * n_target)
        self.index_cache[key] = matrix
        return matrix

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        features = torch.cat((source, target), dim=0)
        kernel = sum(component(features) for component in self.kernels)
        return (kernel * self._index_matrix(len(source), len(target), source.device)).sum()


@torch.no_grad()
def hidden_features(model: BinaryMLP, features: Tensor, device: torch.device) -> Tensor:
    model.eval()
    values: list[Tensor] = []
    for start in range(0, len(features), EMBEDDING_BATCH_SIZE):
        _, hidden = model(features[start : start + EMBEDDING_BATCH_SIZE].to(device))
        values.append(hidden.cpu())
    return torch.cat(values)


@torch.no_grad()
def build_reference_bank(
    model: BinaryMLP,
    features: Tensor,
    true_labels: Tensor,
    device: torch.device,
    threshold: float,
) -> dict[int, Tensor]:
    """Detached same-class hidden features from currently correct samples."""
    probability = predict_probability(model, features, device)
    predicted = (probability >= threshold).long()
    hidden = hidden_features(model, features, device)
    bank: dict[int, Tensor] = {}
    for label in (KNOWN, UNKNOWN):
        mask = (true_labels == label) & (predicted == label)
        if int(mask.sum()) >= 2:
            bank[label] = hidden[mask]
    return bank


def center_regularizer(
    hidden: Tensor,
    target_labels: Tensor,
    hard_mask: Tensor,
    bank: dict[int, Tensor],
    device: torch.device,
) -> Tensor:
    terms: list[Tensor] = []
    for label_tensor in target_labels[hard_mask].unique():
        label = int(label_tensor.item())
        if label not in bank:
            continue
        centroid = bank[label].mean(dim=0).to(device)
        selected = hidden[(target_labels == label_tensor) & hard_mask]
        terms.append((selected - centroid).pow(2).sum(dim=1).mean())
    return torch.stack(terms).mean() if terms else hidden.new_zeros(())


def mkmmd_regularizer(
    hidden: Tensor,
    target_labels: Tensor,
    hard_mask: Tensor,
    bank: dict[int, Tensor],
    mmd: MultipleKernelMaximumMeanDiscrepancy,
    target_size: int,
    device: torch.device,
) -> Tensor:
    terms: list[Tensor] = []
    for label_tensor in target_labels[hard_mask].unique():
        label = int(label_tensor.item())
        if label not in bank:
            continue
        target = bank[label]
        if len(target) > target_size:
            target = target[torch.randperm(len(target))[:target_size]]
        source = hidden[(target_labels == label_tensor) & hard_mask]
        terms.append(mmd(source, target.to(device)))
    return torch.stack(terms).mean() if terms else hidden.new_zeros(())


def endless_batches(loader: DataLoader) -> Iterator[tuple[Tensor, ...]]:
    while True:
        yield from loader


def set_mlp_train_mode(model: BinaryMLP) -> None:
    model.train()


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
) -> dict[str, float]:
    dataset = TensorDataset(features, true_labels, sample_weights, hard_mask)
    sampler = BalancedBinaryBatchSampler(
        true_labels.tolist(), EMBEDDING_BATCH_SIZE, seed
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    set_mlp_train_mode(model)
    sums = {"total": 0.0, "ce": 0.0, "regularizer": 0.0}
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
        optimizer.zero_grad(set_to_none=True)
        logits, hidden = model(feature_batch)
        per_sample_ce = F.cross_entropy(logits, label_batch, reduction="none")
        ce_loss = (per_sample_ce * weight_batch).sum() / weight_batch.sum().clamp_min(
            1e-12
        )
        regularizer = hidden.new_zeros(())
        if bool(hard_batch.any()) and mode == "hard_weighted_center":
            regularizer = center_regularizer(
                hidden, label_batch, hard_batch, bank, device
            )
        elif bool(hard_batch.any()) and mode == "hard_weighted_mkmmd":
            regularizer = mkmmd_regularizer(
                hidden,
                label_batch,
                hard_batch,
                bank,
                mmd,
                int(config["target_size"]),
                device,
            )
        coefficient = (
            float(config["center_weight"])
            if mode == "hard_weighted_center"
            else float(config["mmd_weight"])
        )
        loss = ce_loss + coefficient * regularizer
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        sums["total"] += loss.item()
        sums["ce"] += ce_loss.item()
        sums["regularizer"] += regularizer.item()
        batches += 1
    return {key: value / max(batches, 1) for key, value in sums.items()}


def run_correction_mode(
    base_model: BinaryMLP,
    mode: str,
    train_bundle: FeatureBundle,
    initial_hard_mask: Tensor,
    initial_correct_mask: Tensor,
    validation_bundle: FeatureBundle,
    device: torch.device,
    threshold: float,
    seed: int,
) -> tuple[BinaryMLP, list[dict[str, Any]]]:
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported correction mode: {mode}")
    branch = copy.deepcopy(base_model).to(device)
    if mode == "baseline":
        return branch.eval(), []
    config = MODE_CONFIGS[mode]
    optimizer = torch.optim.Adam(
        branch.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    true_labels = train_bundle.labels.clone()
    mmd = MultipleKernelMaximumMeanDiscrepancy(MMD_ALPHAS).to(device)
    history: list[dict[str, Any]] = []
    base_validation_bacc = float(
        evaluate(branch, validation_bundle, device, threshold)["balanced_accuracy"]
    )
    min_validation_bacc = base_validation_bacc - SELECTION_BACC_TOLERANCE
    best_state = state_to_cpu(branch.state_dict())
    best_score = (0.0, base_validation_bacc)
    best_round = 0
    stale_rounds = 0
    hardness = torch.zeros(len(train_bundle), dtype=torch.float32)

    for round_index in range(1, int(config["rounds"]) + 1):
        probability = predict_probability(branch, train_bundle.features, device)
        predicted = (probability >= threshold).long()
        hard_mask = predicted != true_labels
        current = hard_mask.float()
        hardness = (
            current
            if round_index == 1
            else HARD_WEIGHT_MOMENTUM * hardness + (1.0 - HARD_WEIGHT_MOMENTUM) * current
        )
        weights = 1.0 + (float(config["hard_weight"]) - 1.0) * hardness
        epoch_rows: list[dict[str, float]] = []
        for epoch_index in range(int(config["epochs_per_round"])):
            bank = build_reference_bank(
                branch,
                train_bundle.features,
                true_labels,
                device,
                threshold,
            )
            epoch_rows.append(
                train_correction_epoch(
                    branch,
                    optimizer,
                    train_bundle.features,
                    true_labels,
                    weights,
                    hard_mask,
                    bank,
                    mode,
                    config,
                    mmd,
                    device,
                    seed + round_index * 1_000 + epoch_index,
                )
            )
        correction = correction_metrics(
            branch,
            train_bundle,
            initial_hard_mask,
            initial_correct_mask,
            device,
            threshold,
        )
        validation = evaluate(branch, validation_bundle, device, threshold)
        correction_rate = float(correction["correction_rate"])
        if math.isnan(correction_rate):
            correction_rate = 1.0
        validation_bacc = float(validation["balanced_accuracy"])
        eligible = validation_bacc >= min_validation_bacc
        score = (correction_rate, validation_bacc)
        if eligible and score > best_score:
            best_score = score
            best_state = state_to_cpu(branch.state_dict())
            best_round = round_index
            stale_rounds = 0
        else:
            stale_rounds += 1
        history.append(
            {
                "round": round_index,
                "eligible_for_selection": eligible,
                "mean_sample_weight": float(weights.mean()),
                "epochs": int(config["epochs_per_round"]),
                "hard_examples_at_round_start": int(hard_mask.sum()),
                "hard_known_at_round_start": int(
                    (hard_mask & (true_labels == KNOWN)).sum()
                ),
                "hard_unknown_at_round_start": int(
                    (hard_mask & (true_labels == UNKNOWN)).sum()
                ),
                "loss_total": float(np.mean([row["total"] for row in epoch_rows])),
                "loss_ce": float(np.mean([row["ce"] for row in epoch_rows])),
                "loss_regularizer": float(
                    np.mean([row["regularizer"] for row in epoch_rows])
                ),
                "correction": correction,
                "validation_balanced_accuracy": float(
                    validation["balanced_accuracy"]
                ),
                "validation_known_recall": float(validation["known_recall"]),
                "validation_unknown_recall": float(validation["unknown_recall"]),
            }
        )
        print(
            f"[{mode}] round {round_index}/{config['rounds']} "
            f"hard={int(hard_mask.sum())} "
            f"corrected={correction_rate:.4f} "
            f"val_bal_acc={validation['balanced_accuracy']:.4f}"
        )
        if stale_rounds >= CORRECTION_PATIENCE:
            break
    for row in history:
        row["selected_for_evaluation"] = row["round"] == best_round
    branch.load_state_dict(best_state)
    branch.eval()
    return branch, history


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    result: list[dict[str, Any]] = []
    numeric = [
        "correction_rate",
        "retention_rate",
        "new_error_count",
        "current_hard_count",
        "validation_balanced_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_known_recall",
        "test_unknown_recall",
        "runtime_seconds",
    ]
    for mode in MODES:
        values = grouped.get(mode, [])
        if not values:
            continue
        summary: dict[str, Any] = {"mode": mode, "repeats": len(values)}
        for key in numeric:
            samples = [float(row[key]) for row in values]
            summary[f"{key}_mean"] = float(np.mean(samples))
            summary[f"{key}_std"] = float(np.std(samples))
        result.append(summary)
    return result


def main() -> None:
    if not RUN_PIPELINE:
        print("Pipeline disabled; set RUN_PIPELINE=True")
        return
    set_seed(SEED)
    if not KAGGLE_INPUT_ROOT.is_dir():
        raise RuntimeError("This pipeline expects Kaggle inputs under /kaggle/input")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Enable a Kaggle GPU before running the full pipeline")

    aol_dir = resolve_dataset_dir("icloud")
    unknown_dir = resolve_dataset_dir("icloud_100")
    checkpoint_path = resolve_unique_file("pretrain_AOL.pth")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("AOL:", aol_dir)
    print("UNKNOWN:", unknown_dir)
    print("Encoder:", checkpoint_path)

    source_train, source_test, dataset_signatures = build_records(
        aol_dir, unknown_dir, TRAIN_CAP_PER_ORIGINAL_CLASS, SEED
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
        json.dumps(
            {"split": split_payload, "encoder": encoder_info["sha256"]}, sort_keys=True
        ).encode()
    ).hexdigest()
    cache_dir = CACHE_ROOT
    print("Extracting frozen encoder features...")
    train_bundle = extract_features(
        train_records, encoder, device, cache_dir / "train.pt", signature_root + ":train"
    )
    validation_bundle = extract_features(
        validation_records,
        encoder,
        device,
        cache_dir / "validation.pt",
        signature_root + ":validation",
    )
    test_bundle = extract_features(
        test_records, encoder, device, cache_dir / "test.pt", signature_root + ":test"
    )

    # Keep baseline initialization identical whether embeddings came from cache or extraction.
    set_seed(SEED)
    base_model, base_history = train_base(train_bundle, validation_bundle, device)
    threshold, threshold_sweep = select_threshold(base_model, validation_bundle, device)
    initial_hard_mask, initial_correct_mask, initial_hard_rows = (
        describe_initial_hard_examples(
            base_model, train_bundle, device, threshold
        )
    )
    print(
        "Baseline hard examples:",
        int(initial_hard_mask.sum()),
        "/",
        len(train_bundle),
    )
    if not initial_hard_rows:
        print(
            "WARNING: baseline has no training errors; hard weighting will be "
            "equivalent to uniform fine-tuning until a new error appears."
        )
    before = {
        "correction": correction_metrics(
            base_model,
            train_bundle,
            initial_hard_mask,
            initial_correct_mask,
            device,
            threshold,
        ),
        "validation": evaluate_slices(
            base_model, validation_bundle, device, threshold
        ),
        "test": evaluate_slices(
            base_model, test_bundle, device, threshold
        ),
    }
    base_dir = OUTPUT_ROOT / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_to_cpu(base_model.state_dict()),
            "model_config": {"in_dim": ENCODER_DIM, "hidden_dim": 256, "classes": 2},
            "unknown_threshold": threshold,
            "encoder": encoder_info,
            "split": split_payload,
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
        "unknown_count": sum(
            row["binary_target"] == UNKNOWN for row in initial_hard_rows
        ),
        "instances": initial_hard_rows,
    }
    write_json(OUTPUT_ROOT / "initial_hard_examples.json", manifest)

    output_dir = OUTPUT_ROOT / "hard_example_correction"
    experiments: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for mode in MODES:
        repeat_count = 1 if mode == "baseline" else REPEATS
        for repeat in range(repeat_count):
            set_seed(SEED + repeat)
            started = time.perf_counter()
            branch, history = run_correction_mode(
                base_model,
                mode,
                train_bundle,
                initial_hard_mask,
                initial_correct_mask,
                validation_bundle,
                device,
                threshold,
                SEED + repeat,
            )
            after_correction = correction_metrics(
                branch,
                train_bundle,
                initial_hard_mask,
                initial_correct_mask,
                device,
                threshold,
            )
            after_validation = evaluate_slices(
                branch, validation_bundle, device, threshold
            )
            after_test = evaluate_slices(
                branch, test_bundle, device, threshold
            )
            runtime_seconds = time.perf_counter() - started
            result = {
                "mode": mode,
                "mode_spec": {
                    **MODE_SPECS[mode],
                    "labels_changed": False,
                    "encoder_frozen": True,
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
            result_dir = output_dir / mode / f"repeat_{repeat}"
            write_json(result_dir / "result.json", result)
            if SAVE_BRANCH_MODELS and mode != "baseline":
                result_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    state_to_cpu(branch.state_dict()),
                    result_dir / "corrected_mlp.pt",
                )
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
                    "validation_balanced_accuracy": full_validation[
                        "balanced_accuracy"
                    ],
                    "test_accuracy": full_test["accuracy"],
                    "test_balanced_accuracy": full_test["balanced_accuracy"],
                    "test_known_recall": full_test["known_recall"],
                    "test_unknown_recall": full_test["unknown_recall"],
                    "runtime_seconds": runtime_seconds,
                }
            )
            del branch
            torch.cuda.empty_cache()

    benchmark_summary = aggregate_rows(benchmark_rows)
    summary = {
        "pipeline": "linh-san-binary-hard-example-correction-v1",
        "reference": "reference/unlearning-kf.ipynb",
        "task": "binary KNOWN/UNKNOWN correction without relabeling",
        "threshold": threshold,
        "positive_class": "UNKNOWN",
        "encoder_frozen": True,
        "hard_weight_momentum": HARD_WEIGHT_MOMENTUM,
        "selection_bacc_tolerance": SELECTION_BACC_TOLERANCE,
        "encoder": encoder_info,
        "dataset_signatures": dataset_signatures,
        "initial_hard_manifest": manifest,
        "before": before,
        "modes": list(MODES),
        "mode_specs": {
            mode: {
                **MODE_SPECS[mode],
                "labels_changed": False,
                "encoder_frozen": True,
            }
            for mode in MODES
        },
        "mode_hyperparameters": MODE_CONFIGS,
        "mode_configs": MODE_CONFIGS,
        "experiments": experiments,
        "benchmark_rows": benchmark_rows,
        "benchmark_summary": benchmark_summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "benchmark_runs.csv", benchmark_rows)
    write_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
