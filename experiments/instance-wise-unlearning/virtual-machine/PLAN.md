# Kế hoạch chạy Instance-Wise Unlearning trên máy ảo

## 1. Mục tiêu

`pipeline.ipynb` là bản chạy máy ảo của benchmark hiện hành. Logic model,
objective, mode, hyperparameter mặc định, metric và schema output phải đồng bộ
với `../kaggle/pipeline.ipynb`. Hai notebook chỉ khác cách resolve filesystem và
môi trường vận hành.

Bài toán là phân loại nhị phân traffic:

- `KNOWN = 0`: instance từ AOL.
- `UNKNOWN = 1`: instance từ `icloud_100`.
- Mỗi instance là một dòng CSV vật lý gồm 10.000 packet record.
- Mỗi packet có `[label_id, relative_time, direction, packet_size]`.
- Encoder nhận ba feature cuối và tạo embedding 256 chiều.
- MLP có kiến trúc `256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 2`.

Instance-wise unlearning chuyển đúng các AOL row trong `Df` sang hành vi
`UNKNOWN`, đồng thời cố giữ chất lượng trên phần retain `Dr`.

## 2. Cấu trúc runtime

Đường dẫn mặc định:

```text
known dataset:  /home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud
unknown dataset:/home/ubuntu/Documents/KF/data_processed/icloud_100
encoder:        <repo>/experiments/instance-wise-unlearning/virtual-machine/weights/pretrain_AOL.pth
output:         <repo>/artifacts/vm-training/instance-wise
```

Có thể override mà không sửa notebook:

| Biến môi trường | Vai trò |
|---|---|
| `BKCS_REPO_ROOT` | Repo root khi kernel không chạy trong repo |
| `BKCS_VM_INPUT_ROOT` | Root chung chứa hai dataset CSV |
| `BKCS_VM_WORKING_ROOT` | Root ghi artifact |
| `AOL_DATASET_DIR` | Đường dẫn chính xác tới dataset known |
| `UNKNOWN_100_DATASET_DIR` | Đường dẫn chính xác tới dataset unknown |
| `PRETRAIN_AOL_PATH` | Checkpoint encoder AOL |

Dataset phải có `train_data.csv`, `test_data.csv` và `label_mapping.csv`.
Notebook xác minh path input nằm dưới `BKCS_VM_INPUT_ROOT` và output nằm dưới
`BKCS_VM_WORKING_ROOT` để tránh ghi nhầm vị trí.

## 3. Bốn phase

1. `validate-schema`: kiểm tra schema hai dataset và checkpoint.
2. `train-base`: train binary MLP từ encoder pretrained, chọn threshold trên
   validation và lưu base checkpoint.
3. `make-manifests`: tạo nested forget manifest cho count `1,4,10,50,100`.
4. `unlearn`: chạy tuần tự các mode và repeat từ cùng base checkpoint.

Cell cấu hình cuối có:

```python
RUN_PIPELINE = True
FORGET_COUNT = 1
NOTEBOOK_COMMANDS = [
    "validate-schema",
    "train-base",
    "make-manifests",
    "unlearn",
]
```

Khi base checkpoint và manifest đã tồn tại, bỏ `train-base` và
`make-manifests` khỏi `NOTEBOOK_COMMANDS` để chỉ chạy lại benchmark. Không dùng
manifest cũ với base checkpoint mới vì notebook kiểm tra SHA-256 và dataset
signature.

## 4. Benchmark mode

| Mode | Objective | Scope |
|---|---|---|
| `baseline` | Không cập nhật | none |
| `only_head` | Targeted rehearsal | linear head cuối |
| `full_mlp` | Targeted rehearsal | toàn MLP |
| `last_encoder_block` | Targeted rehearsal | block encoder cuối + MLP |
| `full_model` | Targeted rehearsal | toàn model |
| `amnesiac` | Negative CE trên `Df` + retain rehearsal | full MLP |
| `blindspot` | Soft UNKNOWN target trên `Df` + teacher distillation trên `Dr` | full model |
| `bnd` | UNKNOWN/KNOWN logit margin + retain rehearsal | full MLP |
| `mkmmd` | Căn chỉnh embedding `Df` với unknown và giữ embedding `Dr` | full model |

Các scope baseline là mode độc lập, không tự gắn vào bốn method unlearning.
Mỗi mode lấy preset riêng từ `DEFAULT_MODE_HYPERPARAMETERS`, gồm epoch, learning
rate, weight decay, max step, loss weight và early-stop target. Preset hiện tại
được chọn để chạy được trên GPU phổ thông; thay đổi phải được ghi vào
`summary.json` và từng `result.json`.

## 5. Evaluation

Threshold UNKNOWN được chọn một lần từ validation của base model và giữ cố định
sau unlearning. Positive class là `UNKNOWN`:

```text
TN = KNOWN -> KNOWN
FP = KNOWN -> UNKNOWN
FN = UNKNOWN -> KNOWN
TP = UNKNOWN -> UNKNOWN
```

Metric bắt buộc:

- Forget success và mean probability UNKNOWN trên `Df`.
- Accuracy, UNKNOWN precision/recall/F1 và balanced accuracy.
- Known recall, unknown recall và các slice `same_original_label`,
  `other_known`, `unknown_100`.
- TN, FP, FN, TP, total error và thay đổi so với baseline.
- Mean, population standard deviation, runtime và hyperparameter thực tế.

Forget success riêng lẻ chỉ cho biết `Df` đã chuyển sang UNKNOWN. Một method tốt
còn phải hạn chế FP và giữ known recall; vì vậy không xếp hạng method chỉ bằng
forget success.

## 6. Output

```text
artifacts/vm-training/instance-wise/
├── csv_index.json
├── feature_cache/
├── forget_manifests/forget_<count>.json
└── base_run/
    ├── split_manifest.json
    ├── base_model/
    │   ├── best_model.pt
    │   └── result.json
    └── instance_unlearning/forget_<count>/
        ├── summary.json
        ├── benchmark_runs.csv
        ├── benchmark_summary.csv
        └── <mode>/repeat_<r>/result.json
```

Mặc định không thêm `--save-models`, nên các branch không lưu checkpoint riêng.
`feature_cache` chỉ là cache tăng tốc, có thể xóa sau khi benchmark hoàn tất.

Schema output giống Kaggle, do đó có thể tạo báo cáo bằng:

```bash
python experiments/instance-wise-unlearning/kaggle/generate_report.py \
  artifacts/vm-training/instance-wise/base_run/instance_unlearning/forget_1 \
  --output artifacts/vm-training/instance-wise/unlearning_benchmark_forget_1.xlsx \
  --verify
```

## 7. Kiểm tra trước full run

1. `torch.cuda.is_available()` trả về `True` và đúng GPU dự kiến.
2. Hai dataset resolve đúng, không có folder trùng tên dưới input root.
3. Checkpoint `pretrain_AOL.pth` tồn tại và hash được ghi vào base result.
4. Smoke run với `baseline,only_head`, một repeat, một step hoàn tất.
5. Mọi non-baseline branch bắt đầu từ cùng base checkpoint.
6. `Df ∪ Dr` bằng base train và `Df ∩ Dr` rỗng.
7. `summary.json`, benchmark CSV và từng `result.json` được tạo.

## 8. Quan hệ với nhánh Kaggle

Khi thay đổi model, objective, metric hoặc schema output, cập nhật cả hai notebook
trong cùng thay đổi. Các khác biệt hợp lệ chỉ gồm:

- Cách tìm input/checkpoint.
- Root ghi output.
- Thông báo và hướng dẫn vận hành theo nền tảng.

`../kaggle/PLAN.md` là đặc tả đầy đủ cho công thức loss và chi tiết từng method;
tài liệu này tập trung vào contract và thao tác riêng của VM.
