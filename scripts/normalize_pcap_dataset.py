#!/usr/bin/env python3
"""Normalize a labelled PCAP dataset into reusable PyTorch shards.

The extractor deliberately implements the same packet representation as
``MLP-Classfication/code/train_pipeline.ipynb``:

    IPv4/IPv6 PCAP -> [MAX_PACKETS, 3] float32 + [MAX_PACKETS] bool mask

Each source PCAP remains one sample.  The canonical original label is the
name of its parent directory; binary known/unknown labels are intentionally
not written because they belong to a particular experiment split.

Typical Colab invocation
-------------------------
python scripts/normalize_pcap_dataset.py --mount-drive

The defaults target ``273 (lan 1)`` and write to the approved Drive artifact
location.  Re-running with ``--resume`` safely skips samples already recorded
in the manifest.  Use ``--overwrite`` only when an entire artifact must be
rebuilt from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
except ImportError:  # Allows --help to work before the Colab dependencies are installed.
    torch = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
EXTRACTION_VERSION = "pcap3_log_normalized_v1"
PCAP_SUFFIXES = {".pcap", ".cap", ".pcapng"}
PACKET_FEATURES = 3
DEFAULT_MAX_PACKETS = 256
DEFAULT_SHARD_SIZE = 512
DEFAULT_INPUT_DIR = (
    "/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (lan 1)"
)
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/unlearning-artifacts/normalized/273_lan_1"
DEFAULT_DATASET_ID = "273_lan_1"


@dataclass(frozen=True)
class SourceSample:
    path: Path
    relative_path: str
    original_label: str
    sample_id: str


@dataclass
class ParsedSample:
    source: SourceSample
    features: torch.Tensor
    mask: torch.Tensor
    ip_packet_count: int
    stored_packet_count: int
    source_size: int
    source_mtime_ns: int
    source_sha256: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("Cần cài PyTorch: pip install torch")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sample_id(dataset_id: str, relative_path: str) -> str:
    value = f"{dataset_id}\0{relative_path}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def scan_sources(input_dir: Path, dataset_id: str) -> list[SourceSample]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy input directory: {input_dir}")

    samples: list[SourceSample] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PCAP_SUFFIXES:
            continue
        relative_path = path.relative_to(input_dir).as_posix()
        # This exactly follows the current notebook's scan_pcap_files().
        original_label = path.parent.name
        samples.append(
            SourceSample(
                path=path,
                relative_path=relative_path,
                original_label=original_label,
                sample_id=source_sample_id(dataset_id, relative_path),
            )
        )
    if not samples:
        suffixes = ", ".join(sorted(PCAP_SUFFIXES))
        raise FileNotFoundError(f"Không tìm thấy file {suffixes} trong {input_dir}")
    return samples


def pcap_to_features(path: Path, max_packets: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mirror the notebook's packet_rows(), infer_local_ip(), and transform."""
    try:
        from scapy.all import PcapReader
        from scapy.layers.inet import IP
        from scapy.layers.inet6 import IPv6
    except ImportError as error:
        raise RuntimeError("Cần cài Scapy: pip install scapy") from error

    rows: list[tuple[float, str, str, int]] = []
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if IP in packet:
                ip_layer = packet[IP]
            elif IPv6 in packet:
                ip_layer = packet[IPv6]
            else:
                continue
            rows.append(
                (float(packet.time), str(ip_layer.src), str(ip_layer.dst), int(len(packet)))
            )

    features = torch.zeros((max_packets, PACKET_FEATURES), dtype=torch.float32)
    mask = torch.zeros((max_packets,), dtype=torch.bool)
    ip_packet_count = len(rows)
    if not rows:
        return features, mask, ip_packet_count

    # The notebook defines local IP as the most frequent source IP, not the
    # most frequent endpoint.  Keep that definition for representation parity.
    local_ip = Counter(src for _, src, _, _ in rows).most_common(1)[0][0]
    start_time = rows[0][0]
    for index, (timestamp, src, _dst, packet_size) in enumerate(rows[:max_packets]):
        relative_time = max(0.0, timestamp - start_time)
        direction = 0.0 if src == local_ip else 1.0
        features[index] = torch.tensor(
            (
                math.log1p(relative_time),
                direction,
                math.log1p(max(packet_size, 0)) / math.log1p(65535.0),
            ),
            dtype=torch.float32,
        )
        mask[index] = True
    return features, mask, ip_packet_count


def parse_source(source: SourceSample, max_packets: int, source_checksum: bool) -> ParsedSample:
    stat = source.path.stat()
    features, mask, ip_packet_count = pcap_to_features(source.path, max_packets)
    return ParsedSample(
        source=source,
        features=features,
        mask=mask,
        ip_packet_count=ip_packet_count,
        stored_packet_count=min(ip_packet_count, max_packets),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        source_sha256=sha256_file(source.path) if source_checksum else None,
    )


def source_metadata(source: SourceSample) -> dict[str, Any]:
    stat = source.path.stat()
    return {
        "sample_id": source.sample_id,
        "relative_path": source.relative_path,
        "original_label": source.original_label,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def load_completed_sample_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    completed: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(str(row["sample_id"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Manifest lỗi ở dòng {line_number}: {manifest_path}"
                ) from error
    return completed


def next_shard_number(shard_dir: Path) -> int:
    numbers: list[int] = []
    for path in shard_dir.glob("shard-*.pt"):
        try:
            numbers.append(int(path.stem.removeprefix("shard-")))
        except ValueError:
            continue
    return max(numbers, default=-1) + 1


def save_shard(
    shard_dir: Path,
    shard_number: int,
    parsed_samples: Sequence[ParsedSample],
    label_to_id: dict[str, int],
    max_packets: int,
) -> tuple[str, list[dict[str, Any]]]:
    shard_name = f"shard-{shard_number:05d}.pt"
    shard_path = shard_dir / shard_name
    temporary = shard_path.with_suffix(".pt.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "features": torch.stack([sample.features for sample in parsed_samples]),
        "masks": torch.stack([sample.mask for sample in parsed_samples]),
        "label_ids": torch.tensor(
            [label_to_id[sample.source.original_label] for sample in parsed_samples],
            dtype=torch.int64,
        ),
        "sample_ids": [sample.source.sample_id for sample in parsed_samples],
    }
    torch.save(payload, temporary)
    temporary.replace(shard_path)

    manifest_rows: list[dict[str, Any]] = []
    for offset, sample in enumerate(parsed_samples):
        manifest_rows.append(
            {
                "status": "ok",
                "sample_id": sample.source.sample_id,
                "dataset_id": None,  # Filled by the caller; keeps this helper reusable.
                "source_root": None,
                "relative_path": sample.source.relative_path,
                "original_label": sample.source.original_label,
                "label_id": label_to_id[sample.source.original_label],
                "ip_packet_count": sample.ip_packet_count,
                "stored_packet_count": sample.stored_packet_count,
                "truncated": sample.ip_packet_count > max_packets,
                "source_size": sample.source_size,
                "source_mtime_ns": sample.source_mtime_ns,
                "source_sha256": sample.source_sha256,
                "shard": f"shards/{shard_name}",
                "offset": offset,
            }
        )
    return shard_name, manifest_rows


def prepare_output(output_dir: Path, input_dir: Path, overwrite: bool, resume: bool) -> None:
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("--output-dir không được trùng với --input-dir")
    if output_dir.exists() and overwrite:
        if not output_dir.is_dir():
            raise ValueError(f"Output tồn tại nhưng không phải directory: {output_dir}")
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not resume:
        raise FileExistsError(
            f"Output đã tồn tại: {output_dir}. Dùng --resume hoặc --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)


def assert_compatible_resume(
    metadata_path: Path, input_dir: Path, dataset_id: str, max_packets: int
) -> None:
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_id": dataset_id,
        "source_root": str(input_dir.resolve()),
        "max_packets": max_packets,
        "packet_features": PACKET_FEATURES,
        "extraction_version": EXTRACTION_VERSION,
    }
    found = {key: metadata.get(key) for key in expected}
    if found != expected:
        raise ValueError(
            "Artifact hiện có không cùng cấu hình. Dùng --overwrite để tạo lại. "
            f"Expected={expected}; found={found}"
        )


def mount_google_drive(force_remount: bool) -> None:
    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError("--mount-drive chỉ chạy được trong Google Colab") from error
    drive.mount("/content/drive", force_remount=force_remount)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--max-packets", type=int, default=DEFAULT_MAX_PACKETS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--source-checksum",
        action="store_true",
        help="Tính SHA-256 từng PCAP để kiểm tra nguồn; an toàn hơn nhưng chậm hơn.",
    )
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--force-remount", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    if args.max_packets <= 0 or args.shard_size <= 0:
        raise ValueError("--max-packets và --shard-size phải lớn hơn 0")
    if args.force_remount and not args.mount_drive:
        raise ValueError("--force-remount yêu cầu --mount-drive")
    if args.mount_drive:
        mount_google_drive(args.force_remount)

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    prepare_output(output_dir, input_dir, args.overwrite, args.resume)
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.jsonl"
    label_map_path = output_dir / "label_map.json"
    shard_dir = output_dir / "shards"
    assert_compatible_resume(metadata_path, input_dir, args.dataset_id, args.max_packets)

    sources = scan_sources(input_dir, args.dataset_id)
    labels = sorted({source.original_label for source in sources})
    label_to_id = {label: index for index, label in enumerate(labels)}
    if label_map_path.exists() and args.resume:
        existing_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        if existing_map.get("label_to_id") != label_to_id:
            raise ValueError("label_map.json không khớp input hiện tại; dùng --overwrite.")

    write_json(
        label_map_path,
        {
            "dataset_id": args.dataset_id,
            "label_to_id": label_to_id,
            "id_to_label": labels,
        },
    )
    initial_metadata = {
        "schema_version": SCHEMA_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "status": "running",
        "dataset_id": args.dataset_id,
        "source_root": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "created_at_utc": utc_now(),
        "max_packets": args.max_packets,
        "packet_features": PACKET_FEATURES,
        "feature_names": [
            "log1p_relative_time",
            "direction_local_source_zero",
            "normalized_log1p_packet_size",
        ],
        "direction_rule": "most_frequent_source_ip_is_local; local->0, other->1",
        "packet_filter": "IPv4 and IPv6 packets only; one PCAP/CAP/PCAPNG file is one sample",
        "source_checksum": "sha256" if args.source_checksum else "disabled",
        "shard_size": args.shard_size,
        "source_files_discovered": len(sources),
        "labels_discovered": len(labels),
    }
    write_json(metadata_path, initial_metadata)

    completed = load_completed_sample_ids(manifest_path) if args.resume else set()
    remaining = [source for source in sources if source.sample_id not in completed]
    print(f"Discovered {len(sources)} PCAP/CAP/PCAPNG files across {len(labels)} labels.")
    print(f"Already recorded: {len(completed)}; remaining: {len(remaining)}")

    shard_number = next_shard_number(shard_dir)
    parsed_batch: list[ParsedSample] = []
    error_rows: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0

    def flush_batch() -> None:
        nonlocal shard_number, success_count
        if not parsed_batch:
            return
        _shard_name, rows = save_shard(
            shard_dir, shard_number, parsed_batch, label_to_id, args.max_packets
        )
        for row in rows:
            row["dataset_id"] = args.dataset_id
            row["source_root"] = str(input_dir.resolve())
        append_jsonl(manifest_path, rows)
        success_count += len(rows)
        print(f"Saved {_shard_name}: {len(rows)} samples")
        parsed_batch.clear()
        shard_number += 1

    for number, source in enumerate(remaining, start=1):
        try:
            parsed_batch.append(parse_source(source, args.max_packets, args.source_checksum))
        except Exception as error:  # Keep a durable record and continue the dataset run.
            row = source_metadata(source)
            row.update(
                {
                    "status": "error",
                    "dataset_id": args.dataset_id,
                    "source_root": str(input_dir.resolve()),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            error_rows.append(row)
            error_count += 1
            print(f"[{number}/{len(remaining)}] ERROR {source.relative_path}: {type(error).__name__}: {error}")
        if len(parsed_batch) >= args.shard_size:
            flush_batch()
        if len(error_rows) >= 32:
            append_jsonl(manifest_path, error_rows)
            error_rows.clear()
        if number % 100 == 0:
            print(f"Progress {number}/{len(remaining)}; successful this run={success_count}; errors={error_count}")

    flush_batch()
    if error_rows:
        append_jsonl(manifest_path, error_rows)

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    status_counts = Counter(row["status"] for row in rows)
    metadata = {
        **initial_metadata,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "manifest_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "shards_written": len(list(shard_dir.glob("shard-*.pt"))),
    }
    write_json(metadata_path, metadata)

    checksum_paths = [metadata_path, label_map_path, manifest_path, *sorted(shard_dir.glob("shard-*.pt"))]
    checksums = {
        str(path.relative_to(output_dir)): sha256_file(path) for path in checksum_paths
    }
    write_json(
        output_dir / "checksums.json",
        {"algorithm": "sha256", "generated_at_utc": utc_now(), "files": checksums},
    )
    print(f"Complete. OK={status_counts.get('ok', 0)}, errors={status_counts.get('error', 0)}")
    print(f"Artifact: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
