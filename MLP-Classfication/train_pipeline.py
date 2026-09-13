"""Train an encoder + MLP for network-flow known/unknown classification.

Expected raw-flow format
------------------------
Each .txt/.csv file represents one flow. Each non-empty line contains at least:

    timestamp,src_ip,src_port,dst_ip,dst_port,packet_size

The pipeline also reads .pcap/.cap files directly. One pcap file is treated as
one sample and packets are converted to the same five features. If one pcap
contains multiple independent flows, split it into flow-level files first.

The encoder converts a flow into a fixed-size embedding of 256 dimensions:

    raw flow -> packet features -> 1D CNN -> embedding[256] -> MLP -> 2 logits

Labels are inferred from either the parent directory or the filename. Examples:

    data/facebook/flow_001.txt       -> facebook
    data/traffic_facebook_001.txt    -> facebook

Example: train a binary classifier
-----------------------------------
python train_pipeline.py train \
    --input-dir ./data \
    --known-labels facebook,google \
    --unknown-labels tiktok,youtube \
    --output-dir ./artifacts

Example: hold out unknown classes for a stricter open-world test
------------------------------------------------------------------
python train_pipeline.py train \
    --input-dir ./data \
    --known-labels facebook,google \
    --unknown-labels tiktok \
    --holdout-unknown-labels youtube \
    --output-dir ./artifacts

Example: forget selected instances
-----------------------------------
python train_pipeline.py unlearn \
    --input-dir ./data \
    --checkpoint ./artifacts/best_model.pt \
    --forget-list ./forget.txt \
    --output-dir ./artifacts/unlearned

The unlearning objective uses targeted relabeling for binary classification:
the original binary label of each forget instance is flipped, while a retain
loss and a weight-importance anchor preserve the retain set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


PACKET_FEATURES = 5
NUM_CLASSES = 2

# Dataset notes from the Colab/Drive inspection. These constants are advisory;
# CLI users still choose actual inputs through --input-dir.
COLAB_DRIVE_ROOT = "/content/drive/MyDrive/Traffic FingerPrinting "
BASELINE_DATASET_DIR = f"{COLAB_DRIVE_ROOT}/Data/273 (200samples key)"
CANDIDATE_DATASET_DIRS = (
    f"{COLAB_DRIVE_ROOT}/Data/273 (200samples key)",
    f"{COLAB_DRIVE_ROOT}/Data/273 (lan 1)",
    f"{COLAB_DRIVE_ROOT}/Data/AOL (lan 1)",
    f"{COLAB_DRIVE_ROOT}/Data/Data iPad/Data iPad new",
    f"{COLAB_DRIVE_ROOT}/Data/Data iPhone/Data iPhone new",
)
DATASET_PROFILE_NOTES = {
    "273 (200samples key)": (
        "Baseline candidate: <label>/*.pcap, 9005 PCAP files observed in Colab."
    ),
    "273 (lan 1)": "Same 273 family; add after baseline to test capture-session stability.",
    "AOL (lan 1)": "Usable for open-world/generalization; query-phrase labels may be noisier.",
    "Data iPad/Data iPad new": "Cross-device iPad data; use after the baseline pipeline is stable.",
    "Data iPhone/Data iPhone new": "Cross-device iPhone data; use after the baseline pipeline is stable.",
}


@dataclass(frozen=True)
class FlowRecord:
    path: Path
    original_label: str
    binary_label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def infer_label(path: Path, root: Path) -> str:
    """Infer a class label from a label directory or a traffic_*.txt name."""
    if path.parent.resolve() != root.resolve():
        return path.parent.name

    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    label = "_".join(parts)
    if label.startswith("traffic_"):
        label = label[len("traffic_"):]
    return label or stem


def scan_flow_files(root: Path | Sequence[Path]) -> list[tuple[Path, str]]:
    roots = [root] if isinstance(root, Path) else list(root)
    if not roots:
        raise ValueError("Cần ít nhất một folder dữ liệu")
    extensions = {".txt", ".csv", ".pcap", ".cap"}
    files: list[tuple[Path, str]] = []
    for source_root in roots:
        if not source_root.exists() or not source_root.is_dir():
            raise FileNotFoundError(f"Không tìm thấy folder dữ liệu: {source_root}")
        files.extend(
            (path, infer_label(path, source_root))
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
    files.sort()
    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file .txt/.csv/.pcap/.cap trong {[str(path) for path in roots]}"
        )
    return files


def read_pcap_packets(path: Path) -> list[tuple[float, str, int, str, int, float]]:
    """Read IPv4/IPv6 packets from a pcap and expose the CSV-like fields."""
    try:
        from scapy.all import PcapReader
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.inet6 import IPv6
    except ImportError as error:
        raise RuntimeError(
            "Đọc .pcap cần Scapy. Trong Colab hãy chạy cell dependency đầu notebook, "
            "hoặc cài thủ công bằng: pip install scapy"
        ) from error

    packets: list[tuple[float, str, int, str, int, float]] = []
    with PcapReader(str(path)) as capture:
        for packet in capture:
            ip_layer = packet.getlayer(IP)
            if ip_layer is None:
                ip_layer = packet.getlayer(IPv6)
            if ip_layer is None:
                continue

            transport_layer = packet.getlayer(TCP)
            if transport_layer is None:
                transport_layer = packet.getlayer(UDP)

            src_port = int(getattr(transport_layer, "sport", 0) or 0)
            dst_port = int(getattr(transport_layer, "dport", 0) or 0)
            timestamp = float(packet.time)
            packets.append(
                (
                    timestamp,
                    str(ip_layer.src),
                    max(0, min(src_port, 65535)),
                    str(ip_layer.dst),
                    max(0, min(dst_port, 65535)),
                    float(len(packet)),
                )
            )
    return packets


def parse_flow_file(path: Path, max_packets: int) -> tuple[Tensor, Tensor]:
    """Convert one packet-flow file into (features, valid_mask).

    Feature order per packet:
        log(1 + relative_time), direction, source_port, destination_port,
        log(1 + packet_size)
    """
    packets: list[tuple[float, str, int, str, int, float]] = []
    if path.suffix.lower() in {".pcap", ".cap"}:
        packets = read_pcap_packets(path)
    else:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 6:
                    continue
                try:
                    timestamp = float(row[0].strip())
                    src_ip = row[1].strip()
                    src_port = int(float(row[2].strip()))
                    dst_ip = row[3].strip()
                    dst_port = int(float(row[4].strip()))
                    packet_size = float(row[5].strip())
                except (ValueError, TypeError):
                    # Skips headers and malformed packet rows.
                    continue
                if not src_ip or not dst_ip:
                    continue
                packets.append(
                    (
                        timestamp,
                        src_ip,
                        max(0, min(src_port, 65535)),
                        dst_ip,
                        max(0, min(dst_port, 65535)),
                        max(0.0, packet_size),
                    )
                )

    features = torch.zeros(max_packets, PACKET_FEATURES, dtype=torch.float32)
    mask = torch.zeros(max_packets, dtype=torch.float32)
    if not packets:
        return features, mask

    local_ip = Counter(
        ip for _, src_ip, _, dst_ip, _, _ in packets for ip in (src_ip, dst_ip)
    ).most_common(1)[0][0]
    start_time = packets[0][0]

    for index, (timestamp, src_ip, src_port, _, dst_port, packet_size) in enumerate(
        packets[:max_packets]
    ):
        relative_time = max(0.0, timestamp - start_time)
        direction = 0.0 if src_ip == local_ip else 1.0
        features[index] = torch.tensor(
            [
                math.log1p(relative_time),
                direction,
                src_port / 65535.0,
                dst_port / 65535.0,
                math.log1p(packet_size) / math.log1p(65535.0),
            ],
            dtype=torch.float32,
        )
        mask[index] = 1.0

    return features, mask


def build_records(
    root: Path | Sequence[Path],
    known_labels: set[str],
    unknown_labels: set[str],
    holdout_unknown_labels: set[str],
) -> list[FlowRecord]:
    if not known_labels:
        raise ValueError("Cần chỉ định ít nhất một nhãn trong --known-labels")

    overlap = known_labels & (unknown_labels | holdout_unknown_labels)
    if overlap:
        raise ValueError(f"Nhãn xuất hiện ở nhiều nhóm: {sorted(overlap)}")

    files = scan_flow_files(root)
    discovered_labels = {label for _, label in files}
    unknown_labels = set(unknown_labels)
    if not unknown_labels:
        unknown_labels = discovered_labels - known_labels - holdout_unknown_labels
        print(
            "Cảnh báo: --unknown-labels chưa được chỉ định; "
            "các nhãn còn lại sẽ được dùng làm unknown train."
        )

    allowed = known_labels | unknown_labels | holdout_unknown_labels
    records: list[FlowRecord] = []
    for path, label in files:
        if label not in allowed:
            continue
        binary_label = 0 if label in known_labels else 1
        records.append(FlowRecord(path, label, binary_label))

    if not records:
        raise ValueError(
            "Không có flow nào khớp với known/unknown labels. "
            f"Labels tìm thấy: {sorted(discovered_labels)}"
        )
    return records


def split_records(
    records: Sequence[FlowRecord],
    holdout_unknown_labels: set[str],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_validation_ratio: float = 0.25,
) -> tuple[list[FlowRecord], list[FlowRecord], list[FlowRecord]]:
    """Split records; held-out unknown labels are excluded from train and split into validation/test."""
    rng = random.Random(seed)
    train: list[FlowRecord] = []
    validation: list[FlowRecord] = []
    test: list[FlowRecord] = []
    holdout_unknown: list[FlowRecord] = []

    grouped: dict[int, list[FlowRecord]] = {0: [], 1: []}
    for record in records:
        if record.original_label in holdout_unknown_labels:
            holdout_unknown.append(record)
        else:
            grouped[record.binary_label].append(record)

    for label, group in grouped.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else max(0, n - 1)
        n_val = max(1, int(n * val_ratio)) if n >= 5 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train.extend(group[:n_train])
        validation.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(holdout_unknown)
    if holdout_unknown:
        n_holdout_val = int(len(holdout_unknown) * holdout_validation_ratio)
        n_holdout_val = min(max(1, n_holdout_val), max(len(holdout_unknown) - 1, 1))
        validation.extend(holdout_unknown[:n_holdout_val])
        test.extend(holdout_unknown[n_holdout_val:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    if not train or len({record.binary_label for record in train}) < NUM_CLASSES:
        raise ValueError(
            "Tập train phải có cả known và unknown. "
            "Hãy chỉ định --unknown-labels có dữ liệu train."
        )
    return train, validation, test


class FlowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        records: Sequence[FlowRecord],
        max_packets: int,
        cache_dir: Path | None = None,
    ):
        self.records = list(records)
        self.max_packets = max_packets
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.labels = torch.tensor(
            [record.binary_label for record in self.records], dtype=torch.long
        )

    def cache_path(self, record: FlowRecord) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(str(record.path).encode("utf-8")).hexdigest()[:16]
        safe_label = record.original_label.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_label}_{digest}.pt"

    def load_or_parse(self, record: FlowRecord) -> tuple[Tensor, Tensor]:
        cache_path = self.cache_path(record)
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            return cached["features"], cached["mask"]

        features, mask = parse_flow_file(record.path, self.max_packets)
        if cache_path is not None:
            torch.save({"features": features, "mask": mask}, cache_path)
        return features, mask

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        features, mask = self.load_or_parse(self.records[index])
        return features, mask, self.labels[index]


class FlowEncoder(nn.Module):
    """Encode a padded packet sequence into a 256-dimensional flow vector."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(PACKET_FEATURES, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        # features: [batch, packets, packet_features]
        hidden = self.network(features.transpose(1, 2)).transpose(1, 2)
        valid = mask.unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Sequence[int] = (512, 256, 128, 64, 32),
        dropout: float = 0.20,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, NUM_CLASSES))
        self.network = nn.Sequential(*layers)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.network(embedding)


class FlowModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dims: Sequence[int] = (512, 256, 128, 64, 32),
        dropout: float = 0.20,
    ):
        super().__init__()
        self.encoder = FlowEncoder(embedding_dim)
        self.classifier = MLPClassifier(embedding_dim, hidden_dims, dropout)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.encoder(features, mask)
        logits = self.classifier(embedding)
        return logits, embedding


def make_loader(
    records: Sequence[FlowRecord],
    max_packets: int,
    batch_size: int,
    shuffle: bool,
    cache_dir: Path | None = None,
) -> DataLoader:
    return DataLoader(
        FlowDataset(records, max_packets, cache_dir=cache_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(
    batch: tuple[Tensor, Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    features, mask, labels = batch
    return features.to(device), mask.to(device), labels.to(device)


def evaluate(
    model: FlowModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    unknown_threshold: float = 0.50,
) -> dict[str, object]:
    if len(loader.dataset) == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "known_recall": float("nan"),
            "unknown_recall": float("nan"),
            "unknown_precision": float("nan"),
            "balanced_accuracy": float("nan"),
            "unknown_threshold": unknown_threshold,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    known_total = 0
    known_correct = 0
    unknown_total = 0
    unknown_correct = 0
    predicted_unknown_total = 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            features, mask, labels = move_batch(batch, device)
            logits, _ = model(features, mask)
            total_loss += criterion(logits, labels).item() * labels.numel()
            probabilities = torch.softmax(logits, dim=1)
            predictions = (probabilities[:, 1] >= unknown_threshold).long()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            known = labels == 0
            unknown = labels == 1
            predicted_unknown = predictions == 1
            known_total += known.sum().item()
            known_correct += ((predictions == labels) & known).sum().item()
            unknown_total += unknown.sum().item()
            unknown_correct += ((predictions == labels) & unknown).sum().item()
            predicted_unknown_total += predicted_unknown.sum().item()
            for true_label, predicted_label in zip(labels.detach().cpu(), predictions.detach().cpu()):
                confusion[int(true_label), int(predicted_label)] += 1

    accuracy = correct / max(total, 1)
    known_recall = known_correct / max(known_total, 1)
    unknown_recall = unknown_correct / max(unknown_total, 1)
    unknown_precision = unknown_correct / max(predicted_unknown_total, 1)
    balanced_accuracy = 0.5 * (known_recall + unknown_recall)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": accuracy,
        "known_recall": known_recall,
        "unknown_recall": unknown_recall,
        "unknown_precision": unknown_precision,
        "balanced_accuracy": balanced_accuracy,
        "unknown_threshold": unknown_threshold,
        "known_total": known_total,
        "unknown_total": unknown_total,
        "predicted_unknown_total": predicted_unknown_total,
        "confusion_matrix": confusion.tolist(),
    }


def build_class_weights(loader: DataLoader, device: torch.device, enabled: bool) -> Tensor | None:
    if not enabled or not hasattr(loader.dataset, "labels"):
        return None
    labels = loader.dataset.labels.detach().cpu()
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    if (counts == 0).any():
        print(f"Không dùng class weights vì thiếu class trong train counts={counts.tolist()}")
        return None
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights = weights / weights.mean()
    print(f"Train label counts={counts.int().tolist()}, class_weights={weights.tolist()}")
    return weights.to(device)


def metric_value(metrics: dict[str, object], name: str) -> float:
    if name == "open_world_score":
        accuracy = float(metrics.get("accuracy", float("nan")))
        unknown_recall = float(metrics.get("unknown_recall", float("nan")))
        return 0.5 * accuracy + 0.5 * unknown_recall
    value = metrics.get(name, float("nan"))
    return float(value) if isinstance(value, (int, float)) else float("nan")


def train_model(
    model: FlowModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    max_train_batches_per_epoch: int | None = None,
    max_eval_batches: int | None = None,
    log_every_n_batches: int = 5,
    use_class_weights: bool = True,
    checkpoint_score_metric: str = "balanced_accuracy",
    unknown_threshold: float = 0.50,
) -> dict[str, object]:
    model.to(device)
    class_weights = build_class_weights(train_loader, device, use_class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    history: list[dict[str, float]] = []
    run_start = time.perf_counter()

    print(
        f"Train config: epochs={epochs}, batch_size={train_loader.batch_size}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"max_train_batches_per_epoch={max_train_batches_per_epoch}, "
        f"max_eval_batches={max_eval_batches}, log_every_n_batches={log_every_n_batches}, "
        f"use_class_weights={use_class_weights}, checkpoint_score_metric={checkpoint_score_metric}, "
        f"unknown_threshold={unknown_threshold}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total = 0
        batches_seen = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if max_train_batches_per_epoch is not None and batch_index > max_train_batches_per_epoch:
                break
            features, mask, labels = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features, mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total += labels.numel()
            batches_seen += 1
            should_log = batch_index == 1 or batch_index % max(log_every_n_batches, 1) == 0
            if should_log:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch:03d}/{epochs} | batch {batch_index:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} | elapsed={elapsed:.1f}s"
                )

        val_start = time.perf_counter()
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=max_eval_batches,
            unknown_threshold=unknown_threshold,
        )
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = total_loss / max(total, 1)
        score = metric_value(val_metrics, checkpoint_score_metric)
        if math.isnan(score):
            score = -float("inf")
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_batches": float(batches_seen),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_known_recall": float(val_metrics["known_recall"]),
            "val_unknown_recall": float(val_metrics["unknown_recall"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "checkpoint_score": score,
            "epoch_seconds": epoch_seconds,
            "val_seconds": val_seconds,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} done | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_known_recall={row['val_known_recall']:.4f} | "
            f"val_unknown_recall={row['val_unknown_recall']:.4f} | "
            f"val_balanced_acc={row['val_balanced_accuracy']:.4f} | "
            f"score={row['checkpoint_score']:.4f} | "
            f"epoch_time={epoch_seconds:.1f}s | val_time={val_seconds:.1f}s | "
            f"confusion={val_metrics['confusion_matrix']}"
        )

    model.load_state_dict(best_state)
    total_seconds = time.perf_counter() - run_start
    print(f"Train finished in {total_seconds:.1f}s")
    return {"history": history, "best_score": best_score, "total_seconds": total_seconds}


def estimate_weight_importance(
    model: FlowModel,
    retain_loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> dict[str, Tensor]:
    """Estimate diagonal Fisher-style importance from retain samples."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    importance = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in model.named_parameters()
    }
    for batch_index, batch in enumerate(retain_loader):
        if batch_index >= max_batches:
            break
        features, mask, labels = move_batch(batch, device)
        model.zero_grad(set_to_none=True)
        logits, _ = model(features, mask)
        criterion(logits, labels).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                importance[name] += parameter.grad.detach().pow(2)

    for name, values in importance.items():
        mean = values.mean().clamp_min(1e-12)
        importance[name] = values / mean
    model.zero_grad(set_to_none=True)
    return importance


def parameter_anchor_loss(
    model: FlowModel,
    reference_state: dict[str, Tensor],
    importance: dict[str, Tensor],
) -> Tensor:
    loss = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        loss = loss + (importance[name] * (parameter - reference_state[name]).pow(2)).mean()
    return loss


def run_unlearning(
    model: FlowModel,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    retain_weight: float,
    regularization_weight: float,
) -> list[dict[str, float]]:
    """Perform targeted instance-wise relabeling with retain regularization."""
    model.to(device)
    model.eval()
    reference_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    importance = estimate_weight_importance(model, retain_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    steps = max(len(retain_loader), len(forget_loader))
    retain_batches = cycle(retain_loader)
    forget_batches = cycle(forget_loader)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(steps):
            retain_features, retain_mask, retain_labels = move_batch(next(retain_batches), device)
            forget_features, forget_mask, forget_labels = move_batch(next(forget_batches), device)
            optimizer.zero_grad(set_to_none=True)

            retain_logits, _ = model(retain_features, retain_mask)
            forget_logits, _ = model(forget_features, forget_mask)
            forget_target = 1 - forget_labels
            forget_loss = criterion(forget_logits, forget_target)
            retain_loss = criterion(retain_logits, retain_labels)
            reg_loss = parameter_anchor_loss(model, reference_state, importance)
            loss = forget_loss + retain_weight * retain_loss + regularization_weight * reg_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        row = {"epoch": float(epoch), "loss": total_loss / max(steps, 1)}
        history.append(row)
        print(f"Unlearning epoch {epoch:03d}/{epochs} | loss={row['loss']:.6f}")
    return history


def extract_embeddings(
    model: FlowModel,
    records: Sequence[FlowRecord],
    max_packets: int,
    batch_size: int,
    device: torch.device,
    cache_dir: Path | None = None,
) -> Tensor:
    loader = make_loader(records, max_packets, batch_size, shuffle=False, cache_dir=cache_dir)
    model.eval()
    embeddings: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features, mask, _ = move_batch(batch, device)
            _, batch_embeddings = model(features, mask)
            embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 256)


def save_embeddings(
    model: FlowModel,
    records: Sequence[FlowRecord],
    output_path: Path,
    max_packets: int,
    batch_size: int,
    device: torch.device,
    cache_dir: Path | None = None,
) -> None:
    embeddings = extract_embeddings(
        model,
        records,
        max_packets,
        batch_size,
        device,
        cache_dir=cache_dir,
    )
    torch.save(
        {
            "embeddings": embeddings,
            "binary_labels": torch.tensor([r.binary_label for r in records]),
            "original_labels": [r.original_label for r in records],
            "paths": [str(r.path) for r in records],
        },
        output_path,
    )
    print(f"Đã lưu {len(records)} embedding vào {output_path}")


def save_checkpoint(
    path: Path,
    model: FlowModel,
    embedding_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    max_packets: int,
    known_labels: set[str],
    unknown_labels: set[str],
    holdout_unknown_labels: set[str],
    unknown_threshold: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embedding_dim": embedding_dim,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "max_packets": max_packets,
            },
            "known_labels": sorted(known_labels),
            "unknown_labels": sorted(unknown_labels),
            "holdout_unknown_labels": sorted(holdout_unknown_labels),
            "unknown_threshold": unknown_threshold,
        },
        path,
    )
    print(f"Đã lưu model vào {path}")


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FlowModel, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = FlowModel(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def select_forget_records(
    records: Sequence[FlowRecord],
    root: Path,
    forget_list: Path | None,
    forget_labels: set[str],
    forget_fraction: float,
    seed: int,
) -> tuple[list[FlowRecord], list[FlowRecord]]:
    if forget_list:
        requested: set[Path] = set()
        requested_names: set[str] = set()
        for line in forget_list.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            candidate = Path(value)
            resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
            requested.add(resolved)
            requested_names.add(candidate.name)
        forget = [
            record
            for record in records
            if record.path.resolve() in requested or record.path.name in requested_names
        ]
    elif forget_labels:
        candidates = [record for record in records if record.original_label in forget_labels]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        count = max(1, int(len(candidates) * forget_fraction))
        forget = candidates[:count]
    else:
        raise ValueError("Cần chỉ định --forget-list hoặc --forget-labels")

    forget_paths = {record.path.resolve() for record in forget}
    retain = [record for record in records if record.path.resolve() not in forget_paths]
    if not forget:
        raise ValueError("Không tìm thấy mẫu nào trong forget set")
    if not retain:
        raise ValueError("Retain set không được rỗng")
    return retain, forget


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_threshold_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def sweep_unknown_threshold(
    model: FlowModel,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        metrics = evaluate(model, loader, device, unknown_threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(metrics["accuracy"]),
                "known_recall": float(metrics["known_recall"]),
                "unknown_recall": float(metrics["unknown_recall"]),
                "unknown_precision": float(metrics["unknown_precision"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    best = max(
        rows,
        key=lambda row: (row["balanced_accuracy"], row["unknown_recall"], row["accuracy"]),
    )
    return {"best": best, "rows": rows}


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    known_labels = parse_csv_list(args.known_labels)
    unknown_labels = parse_csv_list(args.unknown_labels)
    holdout_labels = parse_csv_list(args.holdout_unknown_labels)

    records = build_records(root, known_labels, unknown_labels, holdout_labels)
    train_records, val_records, test_records = split_records(
        records,
        holdout_labels,
        args.seed,
        holdout_validation_ratio=args.holdout_validation_ratio,
    )
    print(
        f"Records: train={len(train_records)}, val={len(val_records)}, "
        f"test={len(test_records)}, device={device}"
    )
    packet_cache_dir = output_dir / "packet-feature-cache"
    train_loader = make_loader(
        train_records,
        args.max_packets,
        args.batch_size,
        True,
        cache_dir=packet_cache_dir / "train",
    )
    val_loader = make_loader(
        val_records,
        args.max_packets,
        args.batch_size,
        False,
        cache_dir=packet_cache_dir / "validation",
    )
    test_loader = make_loader(
        test_records,
        args.max_packets,
        args.batch_size,
        False,
        cache_dir=packet_cache_dir / "test",
    )

    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(","))
    model = FlowModel(args.embedding_dim, hidden_dims, args.dropout)
    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.learning_rate,
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
        max_eval_batches=args.max_eval_batches,
        log_every_n_batches=args.log_every_n_batches,
        use_class_weights=not args.disable_class_weights,
        checkpoint_score_metric=args.checkpoint_score_metric,
        unknown_threshold=args.unknown_threshold,
    )
    if args.disable_threshold_sweep:
        threshold_result = {"best": {"threshold": args.unknown_threshold}, "rows": []}
        best_unknown_threshold = args.unknown_threshold
    else:
        threshold_result = sweep_unknown_threshold(
            model,
            val_loader,
            device,
            parse_threshold_grid(args.threshold_grid),
        )
        best_unknown_threshold = float(threshold_result["best"]["threshold"])
        print(f"Best unknown threshold: {best_unknown_threshold:.2f}")

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        max_batches=args.max_test_batches,
        unknown_threshold=best_unknown_threshold,
    )
    print(f"Test: {test_metrics}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "best_model.pt",
        model,
        args.embedding_dim,
        hidden_dims,
        args.dropout,
        args.max_packets,
        known_labels,
        unknown_labels,
        holdout_labels,
        unknown_threshold=best_unknown_threshold,
    )
    write_json(output_dir / "training_history.json", result)
    write_json(
        output_dir / "metrics.json",
        {
            "test": test_metrics,
            "threshold_sweep": threshold_result,
            "num_records": len(records),
        },
    )
    save_embeddings(
        model,
        records,
        output_dir / "embeddings.pt",
        args.max_packets,
        args.batch_size,
        device,
        cache_dir=packet_cache_dir / "all",
    )


def unlearn_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint).resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["model_config"]
    max_packets = int(config["max_packets"])
    unknown_threshold = float(checkpoint.get("unknown_threshold") or args.unknown_threshold)
    known_labels = parse_csv_list(args.known_labels) or set(checkpoint["known_labels"])
    unknown_labels = parse_csv_list(args.unknown_labels) or set(checkpoint["unknown_labels"])
    holdout_labels = parse_csv_list(args.holdout_unknown_labels) or set(
        checkpoint.get("holdout_unknown_labels", [])
    )
    root = Path(args.input_dir).resolve()
    records = build_records(root, known_labels, unknown_labels, holdout_labels)
    train_records, _, _ = split_records(records, holdout_labels, args.seed)
    retain_records, forget_records = select_forget_records(
        train_records,
        root,
        Path(args.forget_list).resolve() if args.forget_list else None,
        parse_csv_list(args.forget_labels),
        args.forget_fraction,
        args.seed,
    )
    print(f"Retain instances: {len(retain_records)} | Forget instances: {len(forget_records)}")

    output_dir = Path(args.output_dir).resolve()
    packet_cache_dir = output_dir / "packet-feature-cache"
    retain_loader = make_loader(
        retain_records,
        max_packets,
        args.batch_size,
        True,
        cache_dir=packet_cache_dir / "retain",
    )
    forget_loader = make_loader(
        forget_records,
        max_packets,
        args.batch_size,
        True,
        cache_dir=packet_cache_dir / "forget",
    )
    before_forget = evaluate(model, forget_loader, device, unknown_threshold=unknown_threshold)
    before_retain = evaluate(model, retain_loader, device, unknown_threshold=unknown_threshold)
    history = run_unlearning(
        model,
        retain_loader,
        forget_loader,
        device,
        args.unlearning_epochs,
        args.unlearning_learning_rate,
        args.retain_weight,
        args.regularization_weight,
    )
    after_forget = evaluate(model, forget_loader, device, unknown_threshold=unknown_threshold)
    after_retain = evaluate(model, retain_loader, device, unknown_threshold=unknown_threshold)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "unlearned_model.pt",
        model,
        int(config["embedding_dim"]),
        tuple(int(value) for value in config["hidden_dims"]),
        float(config["dropout"]),
        max_packets,
        known_labels,
        unknown_labels,
        holdout_labels,
        unknown_threshold=unknown_threshold,
    )
    write_json(
        output_dir / "unlearning_metrics.json",
        {
            "before_forget": before_forget,
            "after_forget": after_forget,
            "before_retain": before_retain,
            "after_retain": after_retain,
            "forget_count": len(forget_records),
            "retain_count": len(retain_records),
        },
    )
    write_json(output_dir / "unlearning_history.json", history)
    save_embeddings(
        model,
        records,
        output_dir / "unlearned_embeddings.pt",
        max_packets,
        args.batch_size,
        device,
    )
    print("Before forget:", before_forget)
    print("After forget:", after_forget)
    print("Before retain:", before_retain)
    print("After retain:", after_retain)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--input-dir", required=True, help="Folder chứa raw flow files")
        subparser.add_argument("--output-dir", default="./artifacts")
        subparser.add_argument("--known-labels", default="", help="Danh sách label, phân tách bằng dấu phẩy")
        subparser.add_argument("--unknown-labels", default="", help="Unknown labels dùng khi train")
        subparser.add_argument("--holdout-unknown-labels", default="", help="Unknown labels chỉ dùng để test")
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--device", default="", help="cuda, cpu hoặc để trống để tự chọn")
        subparser.add_argument("--unknown-threshold", type=float, default=0.50)

    train_parser = subparsers.add_parser("train", help="Train encoder và MLP classifier")
    add_common(train_parser)
    train_parser.add_argument("--max-packets", type=int, default=256)
    train_parser.add_argument("--embedding-dim", type=int, default=256)
    train_parser.add_argument("--hidden-dims", default="512,256,128,64,32")
    train_parser.add_argument("--dropout", type=float, default=0.20)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--max-train-batches-per-epoch", type=int, default=None)
    train_parser.add_argument("--max-eval-batches", type=int, default=None)
    train_parser.add_argument("--max-test-batches", type=int, default=None)
    train_parser.add_argument("--log-every-n-batches", type=int, default=2)
    train_parser.add_argument("--holdout-validation-ratio", type=float, default=0.25)
    train_parser.add_argument(
        "--threshold-grid",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )
    train_parser.add_argument("--disable-threshold-sweep", action="store_true")
    train_parser.add_argument("--disable-class-weights", action="store_true")
    train_parser.add_argument(
        "--checkpoint-score-metric",
        default="balanced_accuracy",
        choices=["accuracy", "balanced_accuracy", "unknown_recall", "open_world_score"],
    )

    unlearn_parser = subparsers.add_parser("unlearn", help="Instance-wise unlearning trên Df")
    add_common(unlearn_parser)
    unlearn_parser.add_argument("--checkpoint", required=True)
    unlearn_parser.add_argument("--forget-list", default="", help="File, mỗi dòng một path cần quên")
    unlearn_parser.add_argument("--forget-labels", default="", help="Chọn instance theo label")
    unlearn_parser.add_argument("--forget-fraction", type=float, default=1.0)
    unlearn_parser.add_argument("--unlearning-epochs", type=int, default=10)
    unlearn_parser.add_argument("--unlearning-learning-rate", type=float, default=1e-4)
    unlearn_parser.add_argument("--retain-weight", type=float, default=1.0)
    unlearn_parser.add_argument("--regularization-weight", type=float, default=1e-3)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "unlearn":
        unlearn_command(args)


if __name__ == "__main__":
    main()
