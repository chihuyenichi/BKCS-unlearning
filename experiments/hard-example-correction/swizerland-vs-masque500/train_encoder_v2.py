# OVERVIEW: paths, encoder contract, and the (Kaggle-sized) training hyper-parameters.
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

KAGGLE_INPUT_ROOT = Path(os.environ.get("BKCS_INPUT_ROOT", "/kaggle/input"))
OUTPUT_DIR = Path(os.environ.get("BKCS_OUTPUT_DIR", "/kaggle/working"))
KNOWN_DATASET_NAMES = ("swizerland-extracted", "swizerland")   # dataset root folder or sub-folder holding the 3 csv files
CHECKPOINT_NAME = "pretrain_swizerland.pth"                      # pipeline.ipynb looks for exactly this file name

# Encoder contract (must match RawPacketEncoder in pipeline.py).
MAX_PACKETS = 10_000
FIELDS_PER_PACKET = 4          # [label_id, relative_time, direction, packet_size]
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128

# Data volume: rows kept per class from train_data.csv (the reference notebook used 150 per class).
N_PER_CLASS = 70
VAL_PER_CLASS = 10             # extra held-out train rows per class, used only for the 1-NN check
SEED = 42

# Optimisation (from reference/unlearning-kf.ipynb; epochs reduced to fit a Kaggle session).
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
TEMPERATURE = 0.1
COSINE_SCHEDULE = True
LR_DECAY_RATE = 0.1
WARMUP = True
WARMUP_EPOCHS = 10
WARMUP_FROM = 0.01
NUM_WORKERS = 2

# Session safety.
TIME_BUDGET_HOURS = 10.5       # stop at an epoch boundary after this much training time
CHECKPOINT_EVERY = 10          # rewrite the checkpoint every N epochs
EVAL_EVERY = 20                # 1-NN sanity check every N epochs (and at the end)
# The reference augmentation also adds noise to padded steps, which stay exactly 0 at inference;
# keeping padded steps at 0 removes that train/inference mismatch.
KEEP_PADDING_ZERO = True

assert N_PER_CLASS >= 2

# OVERVIEW: SupConLoss (Khosla et al., as in reference/unlearning-kf.ipynb), AverageMeter, LR schedule and warm-up.
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf"""

    def __init__(self, temperature=0.1, contrast_mode="all", base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Anchors without a positive pair would divide by zero; count them as 1 (their loss term is 0).
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def _cosine_lr(epoch):
    eta_min = LEARNING_RATE * (LR_DECAY_RATE ** 3)
    return eta_min + (LEARNING_RATE - eta_min) * (1 + math.cos(math.pi * epoch / EPOCHS)) / 2


def adjust_learning_rate(optimizer, epoch):
    lr = _cosine_lr(epoch) if COSINE_SCHEDULE else LEARNING_RATE
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def warmup_learning_rate(optimizer, epoch, batch_id, total_batches):
    if not WARMUP or epoch > WARMUP_EPOCHS:
        return
    progress = (batch_id + (epoch - 1) * total_batches) / (WARMUP_EPOCHS * total_batches)
    warmup_to = _cosine_lr(WARMUP_EPOCHS) if COSINE_SCHEDULE else LEARNING_RATE
    lr = WARMUP_FROM + progress * (warmup_to - WARMUP_FROM)
    for group in optimizer.param_groups:
        group["lr"] = lr

# OVERVIEW: find the known dataset folder and stream train_data.csv, keeping N_PER_CLASS (+ VAL_PER_CLASS) rows per class.
def resolve_dataset_dir(*folder_names):
    required = {"train_data.csv", "test_data.csv", "label_mapping.csv"}
    matches = sorted({
        path.resolve()
        for name in folder_names
        for path in KAGGLE_INPUT_ROOT.rglob(name)
        if path.is_dir() and required.issubset(child.name for child in path.iterdir())
    })
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one dataset folder named one of {folder_names}; found {matches}")
    return matches[0]


def count_classes(label_mapping_path):
    import csv
    with Path(label_mapping_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if row["label_name"] != "")


def load_known_rows(train_csv, n_classes, n_per_class, val_per_class):
    """Stream the csv once. Returns train/val feature arrays [N, MAX_PACKETS, 3] and their label ids.

    Only a row's first field is parsed until it is known to be kept; reading stops when every class is full.
    """
    quota = n_per_class + val_per_class
    seen = Counter()
    full_classes = 0
    train_x, train_y, val_x, val_y = [], [], [], []
    started = time.time()
    with Path(train_csv).open("rb") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.rstrip(b"\r\n")
            if not raw:
                continue
            label = int(float(raw.split(b",", 1)[0]))
            if seen[label] >= quota:
                continue
            values = np.fromstring(raw.decode("ascii"), sep=",", dtype=np.float32)
            if values.size != MAX_PACKETS * FIELDS_PER_PACKET:
                raise ValueError(f"{train_csv}:{line_no}: expected {MAX_PACKETS * FIELDS_PER_PACKET} values, got {values.size}")
            features = values.reshape(MAX_PACKETS, FIELDS_PER_PACKET)[:, 1:]    # [time, direction, size]
            (train_x if seen[label] < n_per_class else val_x).append(features.copy())
            (train_y if seen[label] < n_per_class else val_y).append(label)
            seen[label] += 1
            if seen[label] == quota:
                full_classes += 1
                if full_classes >= n_classes:
                    break
            if line_no % 5000 == 0:
                print(f"  scanned {line_no} rows, kept {len(train_x) + len(val_x)}, {time.time() - started:.0f}s", flush=True)
    stack = lambda rows: np.stack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, MAX_PACKETS, 3), np.float32)
    return stack(train_x), np.asarray(train_y, dtype=np.int64), stack(val_x), np.asarray(val_y, dtype=np.int64)

# OVERVIEW: RawPacketEncoder (identical layout to pipeline.py) and the SupCon wrapper with its projection head.
class RawPacketEncoder(nn.Module):
    """DF-style encoder. Input (batch, max_packets, 3) = [time, direction, size]; output (batch, hidden_size)."""

    def __init__(self, max_packets, hidden_size=256, embedding_size=128):
        super().__init__()
        self.max_packets = max_packets
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        kernel_size, conv_stride, pool_stride, pool_size = 8, 1, 4, 8
        self.conv1 = nn.Conv1d(3, 32, kernel_size, stride=conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride=conv_stride)
        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride=conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride=conv_stride)
        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride=conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride=conv_stride)
        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride=conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride=conv_stride)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        with torch.no_grad():
            flat_dim = self._forward_convs(torch.zeros(1, 3, max_packets)).view(1, -1).shape[1]
        print(f"  flat dim after convs: {flat_dim}")
        self.fc = nn.Linear(flat_dim, hidden_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _forward_convs(self, x):
        x = F.pad(x, (3, 4)); x = F.elu(self.conv1(x))
        x = F.pad(x, (3, 4)); x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4)); x = self.max_pool_1(x)
        x = self.dropout1(x)
        x = F.pad(x, (3, 4)); x = F.relu(self.conv2(x))
        x = F.pad(x, (3, 4)); x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3, 4)); x = self.max_pool_2(x)
        x = self.dropout2(x)
        x = F.pad(x, (3, 4)); x = F.relu(self.conv3(x))
        x = F.pad(x, (3, 4)); x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3, 4)); x = self.max_pool_3(x)
        x = self.dropout3(x)
        x = F.pad(x, (3, 4)); x = F.relu(self.conv4(x))
        x = F.pad(x, (3, 4)); x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3, 4)); x = self.max_pool_4(x)
        return self.dropout4(x)

    def forward(self, x):
        x = self._forward_convs(x.transpose(1, 2))      # (batch, 3, max_packets) -> conv stack
        return self.fc(x.view(x.size(0), -1))            # (batch, hidden_size)


class SupConPacketNet(nn.Module):
    """DF-style backbone + projection head for Supervised Contrastive Learning."""

    def __init__(self, max_packets, hidden_size=256, embedding_size=128):
        super().__init__()
        self.max_packets = max_packets
        self.encoder = RawPacketEncoder(max_packets, hidden_size, embedding_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, x):
        return F.normalize(self.head(self.encoder(x)), dim=1)

# OVERVIEW: packet augmentation (time jitter, size scale, packet dropout, noise) and a dataset returning two views.
def packet_augmentation(x):
    """Random combination of augmentations on one sample of shape (max_packets, 3) = [time, direction, size]."""
    augmented = x.clone()
    if torch.rand(1).item() > 0.2:
        jitter = (torch.rand_like(augmented[:, 0]) - 0.5) * 0.2          # +-0.1 s
        augmented[:, 0] = torch.clamp(augmented[:, 0] + jitter, min=0)
    if torch.rand(1).item() > 0.2:
        augmented[:, 2] = augmented[:, 2] * (0.9 + torch.rand(1).item() * 0.2)   # size x [0.9, 1.1]
    if torch.rand(1).item() > 0.5:
        keep = (torch.rand(x.shape[0]) > 0.1).unsqueeze(1).expand_as(x)
        augmented = augmented * keep.float()
    if torch.rand(1).item() > 0.5:
        augmented = augmented + torch.randn_like(augmented) * 0.05
    augmented = torch.nan_to_num(augmented, nan=0.0, posinf=0.0, neginf=0.0)
    if KEEP_PADDING_ZERO:
        augmented = augmented * (x[:, 2:3] != 0).float()                  # padded steps (size 0) stay [0, 0, 0]
    return augmented


class TwoViewDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.features[index]
        return [packet_augmentation(sample), packet_augmentation(sample)], self.labels[index]

# OVERVIEW: one SupCon epoch, checkpoint writer, a 1-NN check on held-out known rows, and the time-budgeted training loop.
def save_checkpoint(model, optimizer, epoch, path, extra):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "model_config": {"max_packets": MAX_PACKETS, "hidden_size": HIDDEN_SIZE, "embedding_size": EMBEDDING_SIZE},
        **extra,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)           # never leaves a half-written checkpoint behind
    print(f"saved {path} (epoch {epoch})", flush=True)


def train_one_epoch(loader, model, criterion, optimizer, epoch, device):
    model.train()
    meter = AverageMeter()
    for idx, (views, labels) in enumerate(loader):
        images = torch.cat([views[0], views[1]], dim=0).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch = labels.shape[0]
        warmup_learning_rate(optimizer, epoch, idx, len(loader))
        features = model(images)
        f1, f2 = torch.split(features, [batch, batch], dim=0)
        loss = criterion(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), labels)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at batch {idx}; batch skipped")
            continue
        meter.update(loss.item(), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return meter.avg


@torch.no_grad()
def embed(encoder, features, device, batch_size=64):
    encoder.eval()
    out = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device)
        out.append(encoder(batch).float())
    return torch.cat(out)


@torch.no_grad()
def one_nn_accuracy(model, train_x, train_y, val_x, val_y, device):
    """Cosine 1-NN accuracy of held-out known rows against the training rows (class identity, not known/unknown)."""
    if len(val_y) == 0:
        return None
    reference = F.normalize(embed(model.encoder, train_x, device), dim=1)
    queries = F.normalize(embed(model.encoder, val_x, device), dim=1)
    reference_labels = torch.from_numpy(train_y).to(device)
    truth = torch.from_numpy(val_y).to(device)
    correct = 0
    for start in range(0, len(queries), 1024):
        nearest = (queries[start:start + 1024] @ reference.T).argmax(dim=1)
        correct += int((reference_labels[nearest] == truth[start:start + 1024]).sum())
    return correct / len(val_y)


def pretrain_encoder(train_x, train_y, val_x, val_y, device):
    train_dataset = TwoViewDataset(train_x, train_y)
    batch_size = min(BATCH_SIZE, len(train_dataset))
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda", persistent_workers=NUM_WORKERS > 0,
        drop_last=len(train_dataset) > batch_size,
    )
    model = SupConPacketNet(MAX_PACKETS, HIDDEN_SIZE, EMBEDDING_SIZE).to(device)
    criterion = SupConLoss(temperature=TEMPERATURE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,} | train rows {len(train_dataset)} | "
          f"batches/epoch {len(loader)} | epochs {EPOCHS} | batch {batch_size}")

    checkpoint_path = OUTPUT_DIR / CHECKPOINT_NAME
    extra = {"known_dataset": "swizerland-extracted", "n_per_class": N_PER_CLASS, "seed": SEED, "epochs_planned": EPOCHS}
    history = []
    started = time.time()
    last_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        lr = adjust_learning_rate(optimizer, epoch)
        epoch_started = time.time()
        loss = train_one_epoch(loader, model, criterion, optimizer, epoch, device)
        row = {"epoch": epoch, "loss": loss, "lr": lr, "epoch_seconds": time.time() - epoch_started}
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            row["one_nn_accuracy"] = one_nn_accuracy(model, train_x, train_y, val_x, val_y, device)
        history.append(row)
        last_epoch = epoch
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path, extra)
        if (time.time() - started) / 3600 >= TIME_BUDGET_HOURS and epoch < EPOCHS:
            print(f"time budget of {TIME_BUDGET_HOURS} h reached after epoch {epoch}; stopping early")
            break
    save_checkpoint(model, optimizer, last_epoch, checkpoint_path, {**extra, "stopped_early": last_epoch < EPOCHS})
    (OUTPUT_DIR / "train_log.json").write_text(json.dumps({
        "config": {"n_per_class": N_PER_CLASS, "val_per_class": VAL_PER_CLASS, "epochs_planned": EPOCHS,
                   "epochs_done": last_epoch, "batch_size": batch_size, "lr": LEARNING_RATE,
                   "temperature": TEMPERATURE, "keep_padding_zero": KEEP_PADDING_ZERO},
        "train_rows": int(len(train_y)), "val_rows": int(len(val_y)),
        "train_seconds": time.time() - started, "history": history,
    }, indent=2), encoding="utf-8")
    return model

# OVERVIEW: load the known rows, train the encoder, save pretrain_swizerland.pth.
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Enable a Kaggle GPU: SupCon on 10,000-packet rows is far too slow on CPU")
    torch.backends.cudnn.benchmark = True

    known_dir = resolve_dataset_dir(*KNOWN_DATASET_NAMES)
    n_classes = count_classes(known_dir / "label_mapping.csv")
    print("known dataset:", known_dir, "| classes:", n_classes)
    train_x, train_y, val_x, val_y = load_known_rows(known_dir / "train_data.csv", n_classes, N_PER_CLASS, VAL_PER_CLASS)
    print(f"train {train_x.shape} ({len(set(train_y.tolist()))} classes) | held-out {val_x.shape}")
    if len(set(train_y.tolist())) < 2:
        raise RuntimeError("Need at least 2 classes for contrastive learning")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pretrain_encoder(train_x, train_y, val_x, val_y, device)


if __name__ == "__main__":
    main()
