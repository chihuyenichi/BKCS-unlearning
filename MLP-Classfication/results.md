# Báo cáo kết quả baseline gần nhất

Lưu ý: báo cáo này ghi lại run đã chạy xong ngày `2026-09-13` trước khi đổi encoder sang DF-style 3 feature và trước khi tăng thêm dữ liệu. Code hiện tại đã chuyển sang:

```text
PCAP -> packet features [256, 3] -> DF-style CNN encoder -> embedding 256 -> MLP -> known/unknown
```

Do đó các số liệu dưới đây là baseline tham chiếu, không phải kết quả của cấu hình mới. Sau khi chạy lại notebook, cần cập nhật file này bằng kết quả mới.

Cấu hình cần chạy tiếp:

```text
ENCODER_MODE              = supcon
MAX_LABELS_FOR_EXPERIMENT = None
KNOWN_LABEL_COUNT         = 10
HOLDOUT_UNKNOWN_LABEL_COUNT = 0
MAX_FILES_PER_LABEL       = 200
EPOCHS                    = 16
BATCH_SIZE                = 128
LEARNING_RATE             = 2e-4
SUPCON_EPOCHS             = 8
SUPCON_LEARNING_RATE      = 1e-3
FREEZE_ENCODER_AFTER_SUPCON = True
PACKET_FEATURES           = 3
ENCODER_ARCHITECTURE      = df_style_raw_packet_encoder
```

## 1. Thông tin thí nghiệm

```text
Tên thí nghiệm     : Baseline open-world known/unknown classification
Ngày chạy          : 2026-09-13
Notebook           : MLP-Classfication/code/train_pipeline.ipynb
Log gốc            : MLP-Classfication/code/RUN_LOG.md
Mục tiêu           : Phân loại network flow thành 2 lớp known/unknown
Trạng thái         : Đã chạy hoàn chỉnh train, validation, threshold sweep, test và lưu checkpoint
```

Thí nghiệm này đánh giá pipeline cũ tại thời điểm chạy:

```text
PCAP -> packet features [256, 5] -> CNN encoder gọn -> embedding 256 -> MLP -> known/unknown
```

## 2. Môi trường chạy

```text
Runtime             : Google Colab
PyTorch version     : 2.11.0+cu128
CUDA compiled       : True
CUDA available      : True
GPU                 : Tesla T4
Device sử dụng      : cuda
```

Kết luận: thí nghiệm đã chạy trên GPU Colab, không phải CPU local.

## 3. Dataset

Dataset được dùng:

```text
/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)
```

Thống kê dataset gốc:

```text
Tổng số file PCAP/CAP/TXT/CSV       : 9005
Tổng số nhãn gốc                    : 45
Số nhãn đủ điều kiện >= 100 samples : 45
Số nhãn bị loại vì thiếu sample     : 0
```

Baseline tham chiếu chỉ dùng một phần dataset để phù hợp với Colab free GPU:

```text
MAX_LABELS_FOR_EXPERIMENT = 24
MAX_FILES_PER_LABEL       = 160
Tổng sample sau cap       = 3840
```

## 4. Chia nhãn known/unknown

File label split:

```text
/content/drive/MyDrive/unlearning-artifacts/notebook/label_split_seed42_labels24_k45_u35.json
```

Tỷ lệ chia nhãn:

```text
KNOWN_LABEL_RATIO         = 0.45
UNKNOWN_TRAIN_LABEL_RATIO = 0.35
HOLDOUT_UNKNOWN_RATIO     = 0.20
```

Kết quả chia nhãn:

| Nhóm nhãn | Số nhãn | Sample gốc | Sample sau cap | Vai trò |
| --- | ---: | ---: | ---: | --- |
| `known` | 11 | 2205 | 1760 | Có trong train, validation, test |
| `unknown-train` | 8 | 1600 | 1280 | Có trong train, validation, test |
| `unknown-holdout` | 5 | 1000 | 800 | Không vào train, dùng để đánh giá open-world |
| Tổng | 24 | 4805 | 3840 | Tập thí nghiệm hiện tại |

Các nhãn `known`:

```text
office_365
phillies_game
powerball_jackpot
quavo
realtor
soap2day
solitaire
ticketmaster
united_airlines
united_states_elections_2022
us_house_elections_2022
```

Các nhãn `unknown-train`:

```text
office_depot
qr_code_generator
take_off_dead
taylor_swift
twitter
usps
v_for_vendetta
xfinity_customer_service
```

Các nhãn `unknown-holdout`:

```text
quest_diagnostics
ross
social_security
takeoff_shooting
tmz
```

## 5. Chia train/validation/test

```text
train      = 2128 samples
validation = 656 samples
test       = 1056 samples
```

Phân bố binary label:

| Split | Known | Unknown | Tổng |
| --- | ---: | ---: | ---: |
| Train | 1232 | 896 | 2128 |
| Validation | 264 | 392 | 656 |
| Test | 264 | 792 | 1056 |

Validation có chứa một phần holdout unknown để tune threshold phát hiện unknown. Holdout unknown không được đưa vào train.

## 6. Cấu hình mô hình

Input sau parser trong baseline này:

```text
Packet tensor = [256, 5]
Mask tensor   = [256]
Binary label  = 0 known, 1 unknown
```

Encoder trong baseline này:

```text
Conv1d(5, 64, kernel_size=5, padding=2)
BatchNorm1d
GELU
Conv1d(64, 128, kernel_size=5, padding=2)
BatchNorm1d
GELU
Conv1d(128, 128, kernel_size=3, padding=1)
BatchNorm1d
GELU
Masked mean pooling
Linear(128 -> 256)
LayerNorm(256)
```

Classifier:

```text
MLP: 256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 2
Output: 2 logits [known, unknown]
```

Số tham số:

```text
Total parameters = 434,402
```

## 7. Cấu hình huấn luyện

```text
EPOCHS                       = 12
BATCH_SIZE                   = 128
LEARNING_RATE                = 5e-4
TRAIN_BATCHES                = 17
VAL_BATCHES                  = 6
USE_CLASS_WEIGHTS            = True
CLASS_WEIGHTS                = [0.8421, 1.1579]
CHECKPOINT_SCORE_METRIC      = balanced_accuracy
UNKNOWN_THRESHOLD mặc định   = 0.50
USE_THRESHOLD_SWEEP          = True
SAVE_EMBEDDINGS_AFTER_TRAIN  = False
```

Class weights được dùng vì tập train hơi lệch về `known`.

## 8. Kết quả validation

Best checkpoint theo `balanced_accuracy`:

```text
Best validation balanced_accuracy = 0.6529
```

Metric tại epoch tốt nhất:

| Metric | Giá trị |
| --- | ---: |
| `train_loss` | 0.5291 |
| `val_accuracy` | 0.6296 |
| `val_known_recall` | 0.7727 |
| `val_unknown_recall` | 0.5332 |
| `val_balanced_accuracy` | 0.6529 |

Confusion matrix validation:

```text
[[known -> known,   known -> unknown],
 [unknown -> known, unknown -> unknown]]

[[204, 60],
 [183, 209]]
```

## 9. Threshold sweep

Notebook sweep threshold cho xác suất lớp `unknown` trên validation.

Threshold tốt nhất:

```text
BEST_UNKNOWN_THRESHOLD = 0.45
```

Metric tại threshold `0.45` trên validation:

| Metric | Giá trị |
| --- | ---: |
| `balanced_accuracy` | 0.6583 |
| `known_recall` | 0.7197 |
| `unknown_recall` | 0.5969 |
| `unknown_precision` | 0.7597 |

Confusion matrix:

```text
[[190, 74],
 [158, 234]]
```

So với threshold mặc định `0.50`, threshold `0.45` tăng nhẹ balanced accuracy:

```text
threshold 0.50 -> balanced_accuracy = 0.6529
threshold 0.45 -> balanced_accuracy = 0.6583
```

## 10. Kết quả test

Test được chạy với:

```text
BEST_UNKNOWN_THRESHOLD = 0.45
```

Metric test:

| Metric | Giá trị |
| --- | ---: |
| `loss` | 0.8527 |
| `accuracy` | 0.5691 |
| `known_recall` | 0.6932 |
| `unknown_recall` | 0.5278 |
| `unknown_precision` | 0.8377 |
| `balanced_accuracy` | 0.6105 |
| `known_total` | 264 |
| `unknown_total` | 792 |
| `predicted_unknown_total` | 499 |

Confusion matrix test:

```text
[[known -> known,   known -> unknown],
 [unknown -> known, unknown -> unknown]]

[[183, 81],
 [374, 418]]
```

Diễn giải:

| Nhóm | Đúng | Sai | Recall |
| --- | ---: | ---: | ---: |
| Known | 183 | 81 | 0.6932 |
| Unknown | 418 | 374 | 0.5278 |

## 11. Smoke test

Smoke test 5 sample từ `test_records`:

```text
Smoke accuracy = 5/5 = 1.0000
```

Kết quả này chỉ dùng để kiểm tra trực quan pipeline inference. Metric chính vẫn là test set đầy đủ ở mục 10.

## 12. Thời gian chạy

```text
Epoch 1 time   = 902.0s
Validation 1   = 306.4s
Epoch 2-12     = khoảng 5.7s đến 6.0s / epoch
Validation 2-12 = khoảng 1.3s đến 1.4s / lần
Total train    = 966.7s
```

Nhận xét:

- Epoch đầu rất chậm vì PCAP được parse và ghi cache lần đầu.
- Từ epoch 2 trở đi, cache đã hoạt động nên tốc độ train ổn.
- Hiện tượng chậm ban đầu không phải dấu hiệu GPU yếu hoặc notebook bị treo.

## 13. Đánh giá kết quả

Kết quả đạt được:

- Pipeline chạy end-to-end thành công trên GPU Colab.
- Đã đọc được dataset Drive và tạo label inventory.
- Đã random split nhãn ở cấp class cho open-world.
- Đã train encoder + MLP và lưu checkpoint.
- Đã sweep threshold unknown trên validation.
- Đã test trên tập có holdout unknown.
- `unknown_recall` đã tăng rõ so với các lần chạy cũ.

Kết quả còn hạn chế:

- `balanced_accuracy = 0.6105` vẫn còn thấp.
- `unknown_recall = 0.5278`, nghĩa là gần một nửa unknown test vẫn bị đoán thành known.
- Test set lệch mạnh về unknown, nên không nên đánh giá bằng accuracy thuần.
- Baseline này dùng 5 đặc trưng packet cơ bản và encoder gọn; code hiện tại đã chuyển sang 3 feature + DF-style encoder để so sánh lại.

## 14. Kết quả trung gian với DF-style encoder

Sau baseline compact, notebook đã được đổi sang encoder DF-style 3 feature giống hướng `RawPacketEncoder` trong `supcon-model.ipynb`.

Lưu ý: kết quả dưới đây lấy từ output notebook trước khi clear output để chuẩn bị chạy cấu hình mới; chưa được export vào `RUN_LOG.md`.

```text
PACKET_FEATURES           = 3
ENCODER_ARCHITECTURE      = df_style_raw_packet_encoder
MAX_LABELS_FOR_EXPERIMENT = 24
MAX_FILES_PER_LABEL       = 160
Total parameters          = 1,417,570
```

Metric chính:

```text
best validation balanced_accuracy = 0.5875
best unknown threshold            = 0.35
test loss                         = 0.7216
test accuracy                     = 0.6364
test known_recall                 = 0.4848
test unknown_recall               = 0.6869
test unknown_precision            = 0.8000
test balanced_accuracy            = 0.5859
test confusion_matrix             = [[128, 136], [248, 544]]
epoch 1 time                      = 2435.6s
total train time                  = 2546.4s
```

So sánh nhanh với compact baseline:

| Metric | Compact baseline | DF-style trung gian | Nhận xét |
| --- | ---: | ---: | --- |
| `accuracy` | 0.5691 | 0.6364 | Tăng |
| `known_recall` | 0.6932 | 0.4848 | Giảm mạnh |
| `unknown_recall` | 0.5278 | 0.6869 | Tăng |
| `unknown_precision` | 0.8377 | 0.8000 | Giảm nhẹ |
| `balanced_accuracy` | 0.6105 | 0.5859 | Giảm |

Diễn giải:

- DF-style encoder giúp model dự đoán `unknown` nhiều hơn, nên `unknown_recall` tăng.
- `known_recall` giảm mạnh, nghĩa là nhiều known bị đẩy sang unknown; đây là lý do `balanced_accuracy` giảm dù accuracy tăng.
- Số tham số tăng từ `434,402` lên `1,417,570`, nhưng lượng dữ liệu vẫn chỉ `24 * 160`, nên mô hình có nguy cơ học lệch theo split hoặc calibration threshold.
- Hướng tiếp theo hợp lý là chuyển sang SupCon known-only: dùng toàn bộ `45` nhãn, chọn `10` known key, `35` key còn lại làm unknown cho binary classifier, pretrain encoder trên known rồi freeze trước khi train MLP.

## 15. Kết luận

Run này là baseline chạy được gần nhất của project trước khi thay encoder.

Kết luận kỹ thuật:

- Mô hình đã học được ranh giới known/unknown ở mức cơ bản.
- Khả năng phát hiện unknown đã tốt hơn các lần chạy cũ, nhưng chưa đủ mạnh.
- Chưa nên dùng model này làm baseline cuối cho instance-wise unlearning.
- Trước khi đánh giá unlearning, cần cải thiện classifier nền, đặc biệt là `unknown_recall` và `balanced_accuracy`.

## 16. Hướng cải thiện tiếp theo

Các hướng nên ưu tiên:

1. Chạy lại notebook với cấu hình hiện tại: `ENCODER_MODE='supcon'`, toàn bộ `45` nhãn, `10` known key, SupCon `8` epoch, MLP binary `16` epoch.
2. Export output notebook sang `MLP-Classfication/code/RUN_LOG.md` sau khi chạy xong.
3. Cập nhật lại báo cáo này bằng metric mới.
4. Chạy thêm nhiều seed label split để kiểm tra độ ổn định của kết quả.
5. Nếu runtime cho phép, thử toàn bộ `45` nhãn.
6. Bổ sung feature packet/flow như inter-arrival time, signed packet size theo direction, protocol và TCP flags.
7. Tách rõ `unknown-calibration` và `unknown-final-test` nếu cần đánh giá open-world nghiêm ngặt.
8. Sau khi classifier ổn định hơn, mới tạo `Df` ở cấp instance và chạy unlearning.
