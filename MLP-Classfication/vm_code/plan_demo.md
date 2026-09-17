# Plan — Train PCAP trên VM với Google Drive là workspace

## Mục đích

Tạo một implementation chạy trên một VM có GPU, nhưng **toàn bộ dữ liệu và artifact bền vững nằm trên Google Drive đã mount**. VM chỉ cung cấp CPU/GPU/RAM để parse và train; khi VM bị xóa hoặc đổi máy, experiment vẫn tiếp tục được từ Drive.

Plan này đã được triển khai thành `train_pipeline_vm.ipynb`. Nó là checklist để chạy và đánh giá baseline VM trước khi mở rộng sang normalized shard hoặc unlearning.

## Nguồn logic bắt buộc

| Nguồn | Vai trò trong bản VM |
| --- | --- |
| `code/train_pipeline.ipynb` | Nguồn logic chính: config runtime, parser `[256,3]`, split nhãn, SupCon, MLP binary, checkpoint, threshold sweep. |
| `code/supcon-model.ipynb` | Tham khảo DF-style `RawPacketEncoder`, SupCon projection head và checkpoint encoder. Không lấy lại các parser/feature cũ. |
| `plan.md` | Đặc tả thí nghiệm: one-PCAP-one-sample, original label, open-world split, metric và unlearning. |

`train_pipeline.ipynb` là source of truth cho feature hiện tại:

```text
[log1p(relative_time), direction, log1p(packet_size) / log1p(65535)]
features: [256, 3] float32
mask:     [256] bool
```

## Topology Drive-first đã chốt

Google Drive được mount bằng:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Mount layout được chốt có tầng `MyDrive`:

```text
/content/drive/MyDrive/
├── Traffic FingerPrinting /Data/273 (lan 1)/     # PCAP baseline, chỉ đọc
└── unlearning-artifacts/
    └── vm-training/
        ├── normalized/                            # tùy chọn: shard dataset tái sử dụng
        ├── experiments/
        │   ├── vm-a_seed42/
        │   │   ├── feature_cache/
        │   │   ├── label_inventory.json
        │   │   ├── label_split_*.json
        │   │   ├── supcon_encoder_*.pt
        │   │   ├── binary_training_*.pt
        │   │   ├── best_model.pt
        │   │   ├── run_summary.json
        │   │   └── logs/
        │   └── vm-b_seed43/
        └── forget-lists/
```

- Raw PCAP và normalized shards dùng chung ở chế độ read-only.
- Mỗi VM/run bắt buộc có `OUTPUT_DIR` riêng; không để nhiều VM cùng ghi một `feature_cache`, checkpoint hoặc summary.
- Drive mount có thể chậm với hàng nghìn file nhỏ, nhưng vẫn là workspace hợp lệ theo yêu cầu. Cache/shard tồn tại trên Drive để lần sau không parse lại.

## Runtime configuration của notebook hiện tại

Notebook hiện đã hỗ trợ mode local. Trước cell cấu hình runtime, thêm một cell setup trên VM:

```python
import os

os.environ['RUN_CONTEXT'] = 'local'
os.environ['DATA_DIR'] = '/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (lan 1)'
os.environ['OUTPUT_DIR'] = '/content/drive/MyDrive/unlearning-artifacts/vm-training/experiments/vm-a_seed42'
os.environ['DEVICE_NAME'] = 'cuda:0'
```

Lý do đặt rõ `RUN_CONTEXT='local'`:

- không gọi `google.colab.drive.mount()`;
- không dùng hard-coded `/content/drive/...`;
- GPU vẫn được chọn từ VM qua `torch.cuda.is_available()`;
- `REQUIRE_CUDA=True` phải giữ nguyên trong run chính thức để dừng sớm nếu PyTorch/CUDA không nhận GPU.

Không chỉ đặt `RUN_CONTEXT=local`: phải đặt `DATA_DIR`, vì default local hiện trỏ tới `<project-root>/data/273 (200samples key)`.

## Deliverable code đã tạo

Folder `vm_code/` hiện có:

```text
vm_code/
├── plan_demo.md
├── train_pipeline_vm.ipynb
└── README.md                         # cách mount, chạy, resume và layout Drive
```

`train_pipeline_vm.ipynb` là entrypoint code chính và chứa trực tiếp preflight, parser, cache, SupCon, MLP, checkpoint và evaluation. Không phụ thuộc CLI ngoài notebook trong bản demo này.

## Các phase triển khai

### Phase 0 — Preflight VM và Drive

Cell preflight trong notebook kiểm tra, chỉ đọc/ghi một file test nhỏ trong output run:

1. Drive mount tồn tại và có quyền đọc PCAP/ghi artifact.
2. `scapy`, `torch`, CUDA driver và GPU VM tương thích.
3. `torch.cuda.is_available() == True`; in tên GPU, VRAM và PyTorch CUDA version.
4. Quét vài PCAP mẫu để kiểm tra có IPv4/IPv6 packet.
5. In đường dẫn tuyệt đối data/output, tránh ghi nhầm run khác.

Không train khi bất kỳ điều kiện bắt buộc nào thất bại.

### Phase 1 — Mirror pipeline hiện tại trên VM với cache PCAP trên Drive

Giữ đúng logic `train_pipeline.ipynb`:

1. Scan `<label>/*.pcap`; `original_label = parent.name`.
2. Lưu `label_inventory.json` vào `OUTPUT_DIR`.
3. Split class theo seed thành known / unknown / optional holdout unknown.
4. Parse PCAP lazy và cache `{features, mask}` cho từng PCAP vào `OUTPUT_DIR/feature_cache/` trên Drive. Đây là lựa chọn đã chốt cho baseline VM.
5. SupCon pretrain chỉ known train records; checkpoint định kỳ lên Drive.
6. Freeze encoder, train MLP binary known/unknown.
7. Sweep threshold trên validation; evaluate test; lưu metrics và checkpoint tốt nhất.

Kết quả Phase 1 phải tương thích checkpoint/config với pipeline hiện tại.

### Phase 2 — Normalized dataset shared trên Drive (mở rộng sau baseline)

Sau một run baseline ổn định, thay cache hàng nghìn file bằng artifact chuẩn đã thảo luận:

```text
normalized/<dataset-id>/
├── metadata.json
├── label_map.json
├── manifest.jsonl
└── shards/shard-*.pt
```

`original_label`, `sample_id`, relative path, `features`, mask và `label_id` được giữ. `binary_label` luôn tạo động theo label split. Phase này giảm parsing lặp lại giữa VM/run, nhưng không thay đổi model hay split protocol.

### Phase 3 — Unlearning demo

Chỉ triển khai sau khi Phase 1 metric ổn định:

1. `forget_list` nhận `sample_id`/relative PCAP path hoặc original label.
2. Tạo `Df` và `Dr` chỉ từ train records.
3. Load `best_model.pt`, dùng targeted relabeling/flip target trên `Df`, retain loss + weight-importance anchor trên `Dr`.
4. Lưu checkpoint mới, `unlearning_result.json` và metric forget/retain/test trong folder experiment riêng.

## GPU/CPU và nhiều VM

- Một run dùng một GPU VM (`cuda:0`) trước; parser/cache chạy CPU, model chạy GPU.
- Current pipeline chưa dùng distributed data parallel. Nhiều VM nên chạy experiment độc lập theo seed, split, hyperparameter hoặc dataset.
- Không để hai VM resume cùng một checkpoint hoặc dùng cùng `OUTPUT_DIR`.

## Safety và resume

- Checkpoint lưu atomic: ghi file tạm trong cùng folder Drive rồi rename.
- Resume chỉ được phép khi `feature_config`, label split, model config và dataset path khớp metadata của run trước.
- Nếu Drive mount ngắt, dừng run; không tiếp tục ghi artifact nửa chừng. Khi mount ổn định lại, chạy resume từ checkpoint/cache còn nguyên.
- `RUN_UNLEARNING=False` mặc định; không chạy unlearning vô tình trong classification baseline.

## Acceptance criteria demo

Trước full train, code VM phải chứng minh:

1. Preflight nhận GPU VM và Drive mount.
2. Một PCAP cho tensor `[256,3]` và mask `[256]` giống notebook hiện tại.
3. Inventory và label split được lưu trên Drive.
4. Quick run (giới hạn label/file, 1 SupCon epoch, 1 MLP epoch) ghi được checkpoint + summary lên Drive.
5. Resume quick run không đổi label split và không parse lại cache hợp lệ.
6. Hai VM dùng `OUTPUT_DIR` khác nhau không xung đột file.

## Các quyết định đã chốt

1. Mount point là `/content/drive`, với Drive data ở `/content/drive/MyDrive/`.
2. Dataset baseline là `273 (lan 1)`.
3. Demo code được viết trực tiếp trong `train_pipeline_vm.ipynb`.
4. Phase 1 cache từng PCAP trên Drive; normalized shards là Phase 2 sau baseline.
