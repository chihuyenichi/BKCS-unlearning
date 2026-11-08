# Experiments

Mọi thí nghiệm của dự án nằm ở đây, chia theo bài toán:

```text
experiments/
├── README.md                           # file này
├── instance-wise-unlearning/           # benchmark quên instance AOL (known) vs 100 Top Search (unknown)
│   ├── kaggle/                         # bản chạy Kaggle (có thêm 3 mode l2ul_*)
│   │   ├── pipeline.ipynb
│   │   ├── PLAN.md
│   │   ├── generate_report.py
│   │   ├── results/                    # báo cáo đã chọn lọc (xlsx/csv) - được commit
│   │   └── outputs/                    # output thô của run - bị gitignore
│   └── virtual-machine/                # bản chạy VM, dataset local
│       ├── pipeline.ipynb
│       ├── PLAN.md
│       ├── requirements.txt
│       └── weights/                    # pretrain_AOL.pth (track Git), pretrain_273.pth (bỏ qua)
└── hard-example-correction/            # sửa mẫu bị đoán sai bằng fine-tune có trọng số (không phải unlearning)
    ├── aol-vs-top100/                  # AOL (known) vs 100 Top Search (unknown)
    │   ├── pipeline.py / pipeline.ipynb
    │   ├── generate_report.py, generate_benchmark_report.py
    │   ├── PLAN.md, requirements.txt
    │   └── results/
    └── swizerland-vs-masque500/        # Swizerland (known) vs masque_500 (unknown), có train encoder riêng
        ├── train_encoder*.py/.ipynb, pipeline.py/.ipynb, extract_dataset.ipynb
        ├── experiments/                # biến thể: head sweep, SAM, unfreeze
        ├── analyze_paired.py, head_to_standard.py, generate_*report.py
        ├── PLAN.md, REPORT_encoder_v2_fc.md
        ├── sample-input/               # mẫu nhỏ để kiểm tra schema
        └── results/
```

Dataset lớn (CSV đã trích xuất, file zip tải từ Drive) không nằm trong Git mà ở
`data/` (gitignore) ở repo root.

## Instance-wise unlearning

Hai pipeline `kaggle/` và `virtual-machine/` dùng chung data contract, model và
nhóm metric cốt lõi. Bản Kaggle hiện có thêm ba mode instance-wise L2UL theo
paper; vì vậy danh sách mode và hyperparameter không còn hoàn toàn giống bản
máy ảo.

| Nhánh | Dùng khi | Input mặc định | Output mặc định |
|---|---|---|---|
| `kaggle/` | Chạy bằng Kaggle Notebook + GPU | `/kaggle/input` | `/kaggle/working/bkcs-unlearning` |
| `virtual-machine/` | Chạy trên VM có dataset local | `/home/ubuntu/Documents/KF/data_processed` | `<repo>/artifacts/vm-training/instance-wise` |

Các đường dẫn bên dưới tính từ `experiments/instance-wise-unlearning/`.

## Quy trình chung

Mỗi notebook chạy bốn phase:

1. `validate-schema`: xác minh dataset, CSV row và checkpoint.
2. `train-base`: train encoder + binary MLP và chọn fixed UNKNOWN threshold.
3. `make-manifests`: tạo nested forget set `1,4,10,50,100`.
4. `unlearn`: chạy các mode benchmark từ cùng base checkpoint.

Mode hiện hành:

```text
baseline
only_head
full_mlp
last_encoder_block
full_model
l2ul_neggrad
l2ul_adv
l2ul_adv_imp
amnesiac
blindspot
bnd
mkmmd
```

`only_head`, `full_mlp`, `last_encoder_block` và `full_model` là các scope
baseline độc lập. Ba mode `l2ul_*` là implementation riêng trên Kaggle, dùng
full model và không được tự tích hợp vào `amnesiac`, `blindspot`, `bnd`,
`mkmmd` hoặc các scope baseline. Bản VM chưa được port ba mode này.

Mặc định hiện tại:

```python
FORGET_COUNT = 1
repeats = 3
positive_class = UNKNOWN
```

Mỗi method có preset optimizer/loss riêng trong `DEFAULT_MODE_HYPERPARAMETERS`.
Không dùng một epoch hoặc learning rate chung cho tất cả method.

## Chạy trên Kaggle

1. Attach dataset AOL, dataset `icloud_100` và dataset chứa
   `pretrain_AOL.pth`.
2. Bật GPU.
3. Mở `kaggle/pipeline.ipynb`.
4. Kiểm tra các path được resolve ở cell cấu hình.
5. Chạy `validate-schema` trước full pipeline.

Chi tiết data contract, loss, mode và evaluation nằm trong
[`kaggle/PLAN.md`](instance-wise-unlearning/kaggle/PLAN.md).

## Chạy trên máy ảo

Từ repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/instance-wise-unlearning/virtual-machine/requirements.txt
python -m notebook
```

Mở `virtual-machine/pipeline.ipynb`. Các path mặc định có thể override bằng
`BKCS_REPO_ROOT`, `BKCS_VM_INPUT_ROOT`, `BKCS_VM_WORKING_ROOT`,
`AOL_DATASET_DIR`, `UNKNOWN_100_DATASET_DIR` và `PRETRAIN_AOL_PATH`.

Chi tiết vận hành nằm trong
[`virtual-machine/PLAN.md`](instance-wise-unlearning/virtual-machine/PLAN.md).

## Output và báo cáo

Mỗi `forget_<count>` có `summary.json`, `benchmark_runs.csv`,
`benchmark_summary.csv` và `result.json` của từng repeat. `summary.json` là
nguồn chuẩn để sinh báo cáo.

Tạo báo cáo XLSX cho một run:

```bash
python experiments/instance-wise-unlearning/kaggle/generate_report.py \
  experiments/instance-wise-unlearning/kaggle/outputs/bkcs-unlearning/base_run/instance_unlearning/forget_1 \
  --output experiments/instance-wise-unlearning/kaggle/outputs/unlearning_benchmark_forget_1.xlsx \
  --csv-output experiments/instance-wise-unlearning/kaggle/outputs/unlearning_report.csv \
  --verify
```

Trỏ input vào folder cha `instance_unlearning` để tự tạo một workbook riêng
cho mỗi forget count. XLSX gồm sheet `Summary` và một sheet cho từng mode.

`feature_cache`, notebook output, checkpoint branch và file tạm không phải source
code. Không đưa chúng vào Git; xóa cache trước khi lưu Kaggle Version nếu cần
giảm dung lượng.

## Quy tắc đồng bộ

- `kaggle/pipeline.ipynb` và `virtual-machine/pipeline.ipynb` phải giữ cùng
  implementation của model, method, metric và schema output.
- Thay đổi protocol phải cập nhật cả hai `PLAN.md`.
- Chỉ nhánh VM giữ checkpoint local trong `virtual-machine/weights/`.
- Báo cáo hiện có trong `kaggle/results/` là kết quả Kaggle, không phải kết quả
  VM.

## Hard-example correction

Bài toán khác với unlearning: **không đổi ground-truth**, chỉ tìm các mẫu train đang bị đoán sai và
fine-tune MLP để sửa chúng mà hạn chế làm hỏng các mẫu vốn đã đúng. Test split không bao giờ được dùng
để train, chọn ngưỡng hay chọn model.

| Thư mục | Known (`0`) / Unknown (`1`) | Ghi chú |
|---|---|---|
| `hard-example-correction/aol-vs-top100/` | AOL / `icloud_100` | Pipeline gốc: baseline, uniform fine-tune, hard-weighted CE, + Center, + MK-MMD. Đặc tả trong `PLAN.md`. |
| `hard-example-correction/swizerland-vs-masque500/` | Swizerland / masque_500 | Cùng pipeline, thêm `train_encoder*.py` (SupCon) và các biến thể trong `experiments/` (head sweep, SAM, unfreeze). Kết quả tốt nhất hiện có và so sánh ghép cặp theo seed nằm trong `REPORT_encoder_v2_fc.md`. |

Mỗi thư mục có `pipeline.py` và `pipeline.ipynb` giữ đồng bộ nội dung; `.py` là nguồn chuẩn. Báo cáo XLSX đã chọn lọc
nằm ở `results/`; output thô (`outputs/`) bị gitignore. Tên thư mục output trong code (ví dụ
`son-tung-binary-hard-example`) là tên thư mục trên Kaggle và được giữ nguyên để khớp với artifact đã chạy.

