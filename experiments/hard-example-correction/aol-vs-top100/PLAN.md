# Binary hard-example correction

## 1. Mục tiêu

Thí nghiệm này minh họa bài toán **sửa các mẫu đang bị dự đoán sai** bằng
fine-tune có trọng số trong bài toán phân loại nhị phân:

| Nguồn dữ liệu | Nhãn nhị phân | Ý nghĩa |
|---|---:|---|
| AOL (`icloud`) | `KNOWN = 0` | traffic thuộc miền đã biết |
| 100 Top Search (`icloud_100`) | `UNKNOWN = 1` | traffic ngoài miền đã biết |

Đây không còn là machine unlearning. Pipeline không chọn mẫu đang đúng để đổi
thành `UNKNOWN`, không tạo forget set và không thay đổi ground-truth. Mục tiêu
mới là:

1. Train một binary classifier ban đầu.
2. Tìm các mẫu train mà classifier dự đoán sai.
3. Tăng ảnh hưởng của các mẫu sai trong loss.
4. Fine-tune MLP để sửa chúng nhưng hạn chế làm hỏng các mẫu vốn đã đúng.
5. Đánh giá trên validation và test độc lập.

## 2. Dữ liệu Kaggle được giữ nguyên

Notebook tiếp tục tự tìm các artifact dưới `/kaggle/input`:

```text
icloud/
├── train_data.csv
├── test_data.csv
└── label_mapping.csv

icloud_100/
├── train_data.csv
├── test_data.csv
└── label_mapping.csv

pretrain_AOL.pth
```

Quy ước dữ liệu:

- Mỗi dòng CSV là một instance.
- Mỗi instance có `10,000` packet record, mỗi record gồm bốn trường; trường
  đầu mang original label và ba trường còn lại là input của encoder.
- `train_data.csv` của hai nguồn được gộp rồi chia train/validation theo cặp
  `(source, original_label)` với `VALID_RATIO = 0.15`, seed `42`.
- Mỗi original class được giới hạn mặc định ở `150` dòng train để tránh tràn
  RAM. Đặt `TRAIN_CAP_PER_ORIGINAL_CLASS = None` nếu muốn dùng toàn bộ.
- Hai file `test_data.csv` chỉ được dùng cho đánh giá cuối; tuyệt đối không dùng
  để tìm hard examples, chọn threshold hoặc chọn checkpoint.

## 3. Model

Encoder DF-style được tải từ `pretrain_AOL.pth` và đóng băng hoàn toàn:

```text
raw packets [10,000, 3]
        -> frozen DF-style encoder
        -> embedding 256 chiều
        -> binary MLP
        -> logits [KNOWN, UNKNOWN]
```

Binary MLP:

```text
Linear(256 -> 256)
BatchNorm1d(256)
ReLU
Dropout(0.3)
Linear(256 -> 2)
```

Chỉ MLP được cập nhật trong giai đoạn correction. Embedding của train,
validation và test được trích xuất một lần rồi cache dưới `/kaggle/temp`.

## 4. Baseline

Baseline được train bằng cross-entropy trên balanced mini-batch
`KNOWN:UNKNOWN = 1:1`. Checkpoint được chọn theo validation balanced accuracy.

Sau khi baseline được chọn, threshold cho xác suất `UNKNOWN` được sweep trên
validation:

```text
0.10, 0.15, ..., 0.90
```

Threshold tốt nhất được cố định cho tất cả correction mode. Nhờ vậy chênh lệch
kết quả đến từ fine-tune, không phải do mỗi mode được chọn threshold riêng.

## 5. Xác định hard examples

Với nhãn thật `y_i`, xác suất `UNKNOWN` là `p_i` và threshold `t`:

```text
prediction_i = UNKNOWN nếu p_i >= t, ngược lại là KNOWN
hard_i       = prediction_i != y_i
```

Tập hard examples của baseline được lưu cố định để đo tỷ lệ sửa lỗi trước/sau.
Ngoài ra, ở đầu mỗi correction round, model dự đoán lại toàn bộ train set để
cập nhật dynamic hard mask dùng cho round tiếp theo.

Nếu baseline không còn lỗi trên train set, pipeline vẫn chạy và phát cảnh báo;
khi đó hard weighting tương đương uniform fine-tuning cho tới khi xuất hiện lỗi
mới. Không được lấy lỗi trên test để thay thế.

## 6. Objective có trọng số

Cross-entropy theo từng mẫu được tính như sau:

```text
wrong_i(r) = 1 nếu mẫu i đang bị dự đoán sai ở đầu round r, ngược lại 0
h_i(1)     = wrong_i(1)
h_i(r)     = m * h_i(r-1) + (1 - m) * wrong_i(r),   m = HARD_WEIGHT_MOMENTUM = 0.5
w_i        = 1 + (hard_weight - 1) * h_i(r)

L_weighted_CE = sum_i(w_i * CE_i) / sum_i(w_i)
```

Mẫu vừa được sửa không mất ngay trọng số mà giảm dần (ví dụ `5 -> 3 -> 2`).
Nếu đặt lại trọng số về `1` ngay lập tức, decision boundary dao động giữa hai
lớp: round này sửa lỗi UNKNOWN thì round sau lại phát sinh lỗi KNOWN và ngược
lại. Center Loss và MK-MMD vẫn chỉ áp dụng cho mẫu đang sai ở round hiện tại.

Chuẩn hóa theo `sum(w_i)` giữ thang gradient ổn định khi số hard examples thay
đổi giữa các round. Tăng `w_i` làm gradient của mẫu sai đóng góp mạnh hơn; nó
không trực tiếp nhân hay sửa thủ công ma trận trọng số của MLP.

## 7. Các mode so sánh

Mọi correction mode bắt đầu từ cùng một baseline checkpoint và dùng cùng số
round/epoch. Các mode không chạy nối tiếp nhau.

### `baseline`

Không cập nhật thêm. Đây là mốc trước correction.

### `uniform_finetune`

Fine-tune thêm với trọng số bằng `1` cho mọi mẫu. Đây là control quan trọng để
phân biệt lợi ích của hard weighting với lợi ích đơn thuần do train thêm epoch.

### `hard_weighted_ce`

Fine-tune bằng weighted cross-entropy, mặc định:

```text
normal_weight = 1
hard_weight   = 5
```

### `hard_weighted_center`

Dùng weighted cross-entropy và Center Loss. Trong hidden space của MLP, hard
example được kéo về centroid của các mẫu đang dự đoán đúng có cùng nhãn thật:

```text
L = L_weighted_CE + center_weight * L_center
```

### `hard_weighted_mkmmd`

Dùng weighted cross-entropy và class-conditional MK-MMD. Hidden features của
hard examples được căn chỉnh với feature bank của các mẫu đúng cùng lớp:

```text
L = L_weighted_CE + mmd_weight * MK-MMD(hard_c, correct_bank_c)
```

MK-MMD dùng nhiều Gaussian kernel với alpha:

```text
0.001, 0.01, 0.1, 1.0
```

## 8. Chọn checkpoint correction

Sau mỗi round, pipeline đo metric trên validation bằng threshold cố định.
Round 0, tức baseline chưa cập nhật, luôn là một ứng viên. Một round chỉ được
chọn nếu:

```text
validation_BAcc(round) >= validation_BAcc(baseline) - SELECTION_BACC_TOLERANCE
SELECTION_BACC_TOLERANCE = 0.002   # khoảng 3 lỗi KNOWN trên validation
```

Trong các round hợp lệ, checkpoint được chọn theo:

1. correction rate trên fixed baseline-hard set cao nhất;
2. nếu hòa, validation balanced accuracy cao hơn.

Nếu không có round nào hợp lệ, mode trả về baseline. Khi đó correction rate
bằng `0` và `selected_for_evaluation` không đánh dấu round nào. Đây là kết quả
trung thực: correction không cải thiện được mà không làm giảm validation.

Early stopping dùng `CORRECTION_PATIENCE = 5`. Test set chỉ được đọc để báo cáo
sau khi checkpoint đã được chọn.

## 9. Metric

### Sửa lỗi trên train

- `initial_hard_count`: số lỗi của baseline.
- `corrected_count`: số baseline-hard examples đã được sửa đúng.
- `correction_rate = corrected_count / initial_hard_count`.
- `retention_rate`: tỷ lệ mẫu baseline vốn đúng vẫn còn đúng.
- `new_error_count`: số mẫu vốn đúng nhưng bị correction làm sai.
- `current_hard_count`: tổng lỗi train sau correction.

### Validation và test

- Accuracy.
- Balanced accuracy.
- KNOWN recall.
- UNKNOWN recall.
- UNKNOWN precision và F1.
- Confusion matrix `[[TN, FP], [FN, TP]]`.
- Các slice `known_aol`, `unknown_100_topsearch` và `full`.

Balanced accuracy là metric chính vì số mẫu giữa hai nguồn có thể không cân
bằng.

## 10. Output

Output chính:

```text
/kaggle/working/linh-san-binary-hard-example/
├── split_manifest.json
├── initial_hard_examples.json
├── base_model/
│   ├── best_model.pt
│   └── result.json
└── hard_example_correction/
    ├── summary.json
    ├── benchmark_runs.csv
    ├── benchmark_summary.csv
    ├── hard_example_report.xlsx       # sinh bởi generate_report.py
    └── <mode>/repeat_<n>/result.json
```

Embedding cache:

```text
/kaggle/temp/linh-san-binary-hard-example-cache/
```

Cache không nằm trong downloadable output của Kaggle. Mỗi repeat ghi lại seed,
hyperparameter, history từng round, metric trước/sau và hash của baseline
checkpoint để bảo đảm các mode xuất phát từ cùng trạng thái.

## 11. Chạy trên Kaggle

1. Attach hai dataset chứa `icloud`, `icloud_100` và checkpoint
   `pretrain_AOL.pth`.
2. Bật GPU cho Kaggle notebook.
3. Chạy toàn bộ `pipeline.ipynb`.

Notebook là self-contained; `pipeline.py` được giữ đồng bộ như entrypoint chạy
trực tiếp nếu cần:

```python
%run /kaggle/working/pipeline.py
```

Sau khi pipeline chạy xong, tạo báo cáo Excel gọn bằng script riêng:

```python
!python /kaggle/working/generate_report.py \
  /kaggle/working/linh-san-binary-hard-example/hard_example_correction
```

Workbook chỉ có hai sheet, mỗi bảng tối đa chín cột:

- `Summary`: một dòng cho mỗi phương pháp, tập trung vào tỷ lệ sửa lỗi, tỷ lệ giữ
  mẫu đúng, lỗi mới, lỗi train còn lại, test balanced accuracy, chênh lệch so với
  baseline và runtime.
- `Runs`: cùng nhóm metric ở mức từng repeat để kiểm tra độ ổn định; các metric
  đầy đủ vẫn được giữ nguyên trong `summary.json` và các file CSV gốc.

