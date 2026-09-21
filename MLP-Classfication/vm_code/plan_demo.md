# Plan — Train và unlearning PCAP trên VM-local

## Mục tiêu

Chạy toàn bộ pipeline trực tiếp trên ổ đĩa local của máy ảo: đọc PCAP từ hai đường dẫn local, dùng CPU/GPU của VM để extract/train, và ghi cache/checkpoint/kết quả vào `./artifacts/`. Pipeline không mount Google Drive và không phụ thuộc Colab.

Hai biến cần người chạy sửa trong cell đầu của `train_pipeline_vm.ipynb` là:

```python
AOL_DATASET_DIR = Path('/CHANGE_ME/AOL')
UNKNOWN_273_DATASET_DIR = Path('/CHANGE_ME/273')
```

`AOL` là `known` (binary label `0`); `273` là `unknown` (binary label `1`). Cả hai nguồn đều tham gia train/validation/test MLP.

## Input và encoder đã train

Checkpoint `weight_trained/pretrain_AOL.pth` là encoder SupCon cũ, không có MLP classifier. Notebook nạp strict phần `encoder.*` từ checkpoint này và phải dùng đúng contract sau:

```text
one PCAP -> [10000, 3] float32
row       = [relative_time_seconds, direction, raw_packet_size]
direction = 0 nếu src là local IP suy ra từ PCAP, ngược lại 1
```

PCAP ngắn được zero-pad; `mask [10000]` được dùng để mask phần padding. MLP binary mới được khởi tạo và train trên embedding 256 chiều của encoder.

Không được nạp checkpoint này vào encoder `[256,3]` của pipeline cũ hoặc dùng `strict=False` để bỏ qua mismatch kiến trúc.

## Split và đánh giá base model

- Split xác định theo `(source, parent-folder label)`: xấp xỉ 70% train, 15% validation, 15% test; một folder có mặt ở cả ba split khi đủ sample.
- `split_manifest.json` lưu source, original label, binary label, path và split cho từng PCAP.
- Mặc định encoder được freeze, chỉ MLP classifier được train. Có thể bật `FREEZE_ENCODER_FOR_BASE_TRAINING=False` để fine-tune toàn bộ sau khi có baseline.
- Threshold của output `unknown` được chọn trên validation, sau đó test chỉ chạy một lần với threshold đã chọn.

Lưu ý khoa học: AOL-vs-273 có thể đo phân biệt nguồn/capture provenance hơn là khái niệm open-world tổng quát. Kết quả cần được diễn giải là baseline hai nguồn; nên đánh giá thêm unknown source thứ ba về sau.

## Layout artifact local

```text
./artifacts/vm-training/experiments/<RUN_ID>/
├── feature_cache/             # một cache tensor cho mỗi PCAP
├── preflight/local_write_probe.txt
├── label_inventory.json
├── split_manifest.json
├── base_model/
│   ├── README.md
│   ├── label_map.json
│   ├── binary_training_latest.pt
│   ├── binary_training_best.pt
│   └── best_model.pt
├── run_summary.json
├── run_summary.md
└── unlearning/                # chỉ có khi RUN_UNLEARNING=True
    ├── summary.json
    ├── head_only/
    ├── last_encoder_block/
    └── full_encoder_and_head/
```

Mỗi VM/run dùng `RUN_ID` khác nhau để không ghi đè cache hay checkpoint. Raw PCAP không bị sửa.

## Phase 3 — class-level unlearning baseline

`FORGET_LABEL` là tên một folder con của AOL. Khi để `None`, notebook chọn cố định theo `SEED` một AOL label có đủ dữ liệu train/validation/test.

Với label cần quên `B`:

```text
Df = AOL records có parent-folder label B
Dr = AOL records có label khác B + toàn bộ 273 records
```

Mục tiêu baseline có policy rõ ràng là chuyển Df thành unknown:

```text
CE(model(Df), unknown=1) + λ · CE(model(Dr), binary_label_gốc)
```

Ba baseline chạy từ cùng `base_model/best_model.pt` trong bộ nhớ:

1. `head_only`: chỉ MLP classifier cập nhật.
2. `last_encoder_block`: block encoder cuối, FC encoder và MLP cập nhật.
3. `full_encoder_and_head`: toàn bộ encoder và MLP cập nhật.

Validation chọn epoch tốt nhất theo trung bình của `Df -> unknown rate` và retain balanced accuracy. Test báo cáo hai phần riêng: tỷ lệ forget thành unknown trên Df-test, và binary metrics trên Dr-test. Mỗi baseline ghi `unlearned_model.pt`, `result.json`, `README.md` vào thư mục riêng.

Đây là relabel-to-unknown baseline có kiểm soát; không phải chứng minh certified machine unlearning. ADV/MAS và các baseline từ paper sẽ là bước mở rộng sau khi baseline này ổn định.

## Cách chạy

1. Mở notebook trên Jupyter của VM, sửa đúng hai đường dẫn dataset và kiểm tra `PRETRAIN_AOL_PATH`.
2. Chọn GPU cho kernel. Nếu chỉ debug CPU, đặt `REQUIRE_CUDA=False` và `DEVICE_NAME=''`.
3. Run All với `RUN_UNLEARNING=False` để tạo base model.
4. Xem `run_summary.md`, `base_model/README.md` và split manifest.
5. Đặt `RUN_UNLEARNING=True`; tùy chọn đặt `FORGET_LABEL='ten_folder_AOL'`; chạy lại từ đầu để tạo ba artifact unlearning.
