# Báo cáo kết quả group-wise unlearning trên máy ảo — lần chạy trước

Nguồn số liệu chính: `MLP-Classfication/vm_code/out_data/vm_outputs.json`.

File `last_run_results_summary.csv` được giữ để tham chiếu lần chạy single-label cũ hơn. Báo cáo này dựa trên `vm_outputs.json`, tương ứng với cell cuối của notebook hiện được lưu tại [../non-vm-code/train_pipeline_vm.ipynb](../non-vm-code/train_pipeline_vm.ipynb): chạy nested group-unlearning với số nhãn cần quên lần lượt là `1`, `4`, và `8`.

Đây là kết quả của pipeline trước, không phải benchmark của `instance_wise/train_instance_wise.ipynb`. Các path và cấu hình bên dưới mô tả lần chạy lịch sử trên VM. Target unknown ở đây đo hành vi đổi dự đoán; kết quả này chưa chứng minh xóa ảnh hưởng dữ liệu hoặc bảo đảm privacy.

## 1. Mục tiêu thí nghiệm

Bài toán gốc là phân loại nhị phân:

- `AOL` được xem là `known`, nhãn binary `0`.
- `273` được xem là `unknown`, nhãn binary `1`.

Sau khi có base model, bước unlearning chọn một nhóm nhãn con trong tập `AOL` làm tập cần quên `Df`. Các mẫu thuộc `Df` được tối ưu để model dự đoán thành `unknown`. Phần còn lại `Dr` được giữ lại để model không mất khả năng phân loại ban đầu.

Nói cách khác, sau unlearning:

- Input thuộc nhãn AOL cần quên nên được dự đoán là `unknown`.
- Input không thuộc nhãn cần quên vẫn nên được phân loại đúng theo bài toán nhị phân `known/unknown`.

## 2. Cấu hình chạy trên máy ảo

Dữ liệu:

- Known dataset: `/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud`
- Unknown dataset: `/home/ubuntu/Documents/KF/data_processed/icloud_100`
- Output artifact trên máy ảo: `/home/ubuntu/Documents/huyenchi/BKCS-unlearning/MLP-Classfication/vm_code/artifacts/vm-training/experiments/aol_known_vs_273_unknown_csv_seed42`

Thông tin chạy:

- Device: `cuda:0`
- `RUN_UNLEARNING = True`
- `FORGET_LABEL_COUNTS = (1, 4, 8)`
- `FORGET_LABEL_SEED = 42`
- `UNLEARNING_REPEATS = 3`
- Optimization seeds: `42`, `43`, `44`
- `UNLEARNING_EPOCHS = 3`
- `UNLEARNING_LR = 1e-4`
- `RETAIN_LOSS_WEIGHT = 1.0`
- `BEST_UNKNOWN_THRESHOLD = 0.50`

Base model được tải lại từ checkpoint đã lưu:

```text
base_model/best_model.pt
```

Điều này có nghĩa là khi chạy unlearning, notebook không train lại base model từ đầu, mà reuse model đã huấn luyện trước đó.

## 3. Base model trước unlearning

Base model được đánh giá trên tập test gồm `14,895` mẫu.

| Metric | Giá trị |
|---|---:|
| Loss | 0.1906 |
| Accuracy | 0.9521 |
| Known recall | 0.8889 |
| Unknown recall | 0.9830 |
| Balanced accuracy | 0.9359 |
| Test samples | 14,895 |

Nhận xét:

- Model nhận diện `unknown` rất tốt, với `unknown_recall = 0.9830`.
- `known_recall = 0.8889` thấp hơn, nghĩa là một phần mẫu known bị đẩy sang unknown.
- Vì bài toán có thể lệch lớp, `balanced_accuracy = 0.9359` là metric nên ưu tiên hơn accuracy thông thường.

## 4. Định nghĩa các tập dữ liệu trong unlearning

Trong cell cuối:

- `Df`: forget set, gồm các mẫu AOL thuộc nhóm nhãn cần quên.
- `Dr`: retain set, gồm toàn bộ mẫu còn lại.
- `ftr`, `fva`, `fte`: `Df` train, validation, test.
- `rtr`, `rva`, `rte`: `Dr` train, validation, test.

Với mỗi `forget_count`, code chọn nhãn theo cùng một thứ tự shuffle bằng `FORGET_LABEL_SEED = 42`. Vì vậy các nhóm là nested:

- Nhóm 1 nhãn nằm trong nhóm 4 nhãn.
- Nhóm 4 nhãn nằm trong nhóm 8 nhãn.

Danh sách nhãn cần quên:

| Forget count | Forget labels |
|---:|---|
| 1 | `iferenc_of_telling_time_in_france_from_america` |
| 4 | `iferenc_of_telling_time_in_france_from_america`, `hst`, `ftdna`, `celtic_wedding_gown` |
| 8 | `iferenc_of_telling_time_in_france_from_america`, `hst`, `ftdna`, `celtic_wedding_gown`, `automotive_lacquer_paint_for_sale`, `twill_tapestry`, `international_fuel_tax_agreement_tennessee`, `boyz` |

Kích thước `Df/Dr`:

| Forget count | Df train | Dr train | Df val | Dr val | Df test | Dr test |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 85 | 12,574 | 15 | 2,218 | 100 | 14,795 |
| 4 | 296 | 12,363 | 52 | 2,181 | 348 | 14,547 |
| 8 | 636 | 12,023 | 112 | 2,121 | 748 | 14,147 |

Số lượng được lấy từ dòng `Df/Dr sizes` trong log gốc, không giả định mỗi nhãn đều có đúng 100 mẫu test.

## 5. Định nghĩa metric

`accuracy`:

```text
accuracy = số mẫu dự đoán đúng / tổng số mẫu
```

`known_recall`:

```text
known_recall = số mẫu known dự đoán đúng là known / tổng số mẫu known
```

`unknown_recall`:

```text
unknown_recall = số mẫu unknown dự đoán đúng là unknown / tổng số mẫu unknown
```

`balanced_accuracy`:

```text
balanced_accuracy = 0.5 * (known_recall + unknown_recall)
```

Metric này quan trọng vì nó cân bằng chất lượng trên hai lớp `known` và `unknown`.

`forget_rate`:

```text
forget_rate = số mẫu Df được dự đoán thành unknown / tổng số mẫu Df
```

Sau unlearning, `forget_rate` càng cao càng tốt, vì mục tiêu là đẩy nhóm cần quên sang `unknown`.

`retain_bal`:

```text
retain_bal = balanced_accuracy trên Dr-test
```

Metric này đo khả năng giữ lại kiến thức trên phần không cần quên.

`score` trong quá trình train:

```text
score = 0.5 * (Df-val forget_rate + Dr-val balanced_accuracy)
```

Checkpoint tốt nhất được chọn theo `score`. Do đó model sau cùng không nhất thiết là epoch cuối, mà là epoch có cân bằng tốt nhất giữa quên `Df` và giữ `Dr`.

## 6. Các baseline unlearning

Ba baseline khác nhau ở phạm vi trọng số được cập nhật:

| Scope | Trọng số được cập nhật | Ý nghĩa |
|---|---|---|
| `head_only` | Chỉ MLP classifier | Ít can thiệp nhất, encoder giữ nguyên |
| `last_encoder_block` | Block cuối encoder và MLP classifier | Cân bằng giữa giữ encoder và cho phép feature thay đổi |
| `full_encoder_and_head` | Toàn bộ encoder và MLP classifier | Can thiệp mạnh nhất, linh hoạt nhất nhưng có rủi ro làm thay đổi representation |

Loss unlearning:

```text
loss = CE(Df -> unknown) + RETAIN_LOSS_WEIGHT * CE(Dr -> nhãn gốc)
```

Với `RETAIN_LOSS_WEIGHT = 1.0`, hai thành phần loss có hệ số bằng nhau; đóng góp gradient thực tế vẫn phụ thuộc giá trị loss và batch dữ liệu.

## 7. Kết quả tổng hợp

Bảng dưới đây tổng hợp mean và standard deviation trên 3 repeat với seed `42`, `43`, `44`.

Forget và retain balanced accuracy dùng dòng aggregate đã in từ full-precision metrics trong log. Retain accuracy được tổng hợp từ log từng repeat đã làm tròn; có thể sai khác nhỏ so với artifact gốc chưa được cung cấp. Standard deviation dùng quy ước population (`ddof=0`) của notebook.

| Forget count | Scope | Df forget trước | Df forget sau mean ± std | Dr bal trước | Dr bal sau mean ± std | Dr acc trước | Dr acc sau mean ± std |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `head_only` | 0.4100 | 1.0000 ± 0.0000 | 0.9390 | 0.9474 ± 0.0003 | 0.9545 | 0.9599 ± 0.0005 |
| 1 | `last_encoder_block` | 0.4100 | 0.9733 ± 0.0125 | 0.9390 | 0.9608 ± 0.0029 | 0.9545 | 0.9702 ± 0.0004 |
| 1 | `full_encoder_and_head` | 0.4100 | 0.9600 ± 0.0283 | 0.9390 | 0.9600 ± 0.0064 | 0.9545 | 0.9688 ± 0.0033 |
| 4 | `head_only` | 0.1322 | 0.9952 ± 0.0014 | 0.9367 | 0.9363 ± 0.0038 | 0.9541 | 0.9563 ± 0.0022 |
| 4 | `last_encoder_block` | 0.1322 | 0.9895 ± 0.0149 | 0.9367 | 0.9529 ± 0.0112 | 0.9541 | 0.9661 ± 0.0057 |
| 4 | `full_encoder_and_head` | 0.1322 | 0.9818 ± 0.0098 | 0.9367 | 0.9585 ± 0.0059 | 0.9541 | 0.9699 ± 0.0044 |
| 8 | `head_only` | 0.1070 | 0.9933 ± 0.0000 | 0.9356 | 0.9397 ± 0.0016 | 0.9552 | 0.9594 ± 0.0009 |
| 8 | `last_encoder_block` | 0.1070 | 0.9898 ± 0.0023 | 0.9356 | 0.9567 ± 0.0004 | 0.9552 | 0.9699 ± 0.0005 |
| 8 | `full_encoder_and_head` | 0.1070 | 0.9875 ± 0.0035 | 0.9356 | 0.9651 ± 0.0020 | 0.9552 | 0.9751 ± 0.0004 |

## 8. Phân tích theo số nhãn cần quên

### Quên 1 nhãn

Nhóm cần quên có `100` mẫu test. Trước unlearning, `41%` mẫu trong `Df-test` đã bị dự đoán thành unknown. Sau unlearning:

- `head_only` đạt `forget_rate = 1.0000`, cao nhất và rất ổn định.
- `last_encoder_block` đạt `forget_rate = 0.9733`, nhưng giữ `Dr` tốt hơn với `retain_bal = 0.9608`.
- `full_encoder_and_head` đạt `forget_rate = 0.9600`, thấp hơn hai baseline còn lại và dao động lớn hơn.

Nếu ưu tiên quên tuyệt đối cho 1 nhãn, `head_only` là tốt nhất. Nếu ưu tiên cân bằng giữa quên và giữ lại, `last_encoder_block` hợp lý hơn.

### Quên 4 nhãn

Nhóm cần quên có `348` mẫu test. Trước unlearning, `Df-test forget_rate = 0.1322`. Sau unlearning:

- `head_only` vẫn quên rất mạnh với `forget_rate = 0.9952`, nhưng `retain_bal` hầu như không cải thiện so với trước.
- `last_encoder_block` đạt `forget_rate = 0.9895`, và `retain_bal` tăng lên `0.9529`.
- `full_encoder_and_head` đạt `forget_rate = 0.9818`, thấp hơn một chút, nhưng `retain_bal = 0.9585` và `retain_acc = 0.9699` là tốt nhất trong nhóm 4 nhãn.

Khi số nhãn cần quên tăng lên 4, việc cho phép encoder cập nhật bắt đầu có lợi. `full_encoder_and_head` cho kết quả giữ lại tốt nhất, trong khi `head_only` giữ mức quên cao nhất.

### Quên 8 nhãn

Nhóm cần quên có `748` mẫu test. Trước unlearning, `Df-test forget_rate = 0.1070`. Sau unlearning:

- `head_only` đạt `forget_rate = 0.9933`, gần như hoàn hảo, nhưng `retain_bal = 0.9397` thấp hơn rõ so với hai baseline có cập nhật encoder.
- `last_encoder_block` đạt `forget_rate = 0.9898`, `retain_bal = 0.9567`.
- `full_encoder_and_head` đạt `forget_rate = 0.9875`, `retain_bal = 0.9651`, `retain_acc = 0.9751`.

Với 8 nhãn, `full_encoder_and_head` là baseline giữ lại `Dr` tốt nhất, trong khi vẫn đạt mức quên rất cao trên `Df`.

## 9. So sánh tổng quan các baseline

`head_only`:

- Ưu điểm: forget rate cao nhất và ổn định nhất.
- Nhược điểm: retain balanced accuracy thấp hơn khi số nhãn cần quên tăng.
- Phù hợp khi mục tiêu chính là đẩy nhãn cần quên sang unknown mà muốn ít can thiệp encoder.

`last_encoder_block`:

- Ưu điểm: cân bằng tốt giữa quên và giữ, đặc biệt ở `forget_count = 1` và `8`.
- Nhược điểm: với `forget_count = 4`, độ lệch chuẩn của retain_bal cao hơn, nghĩa là kết quả phụ thuộc seed hơn.
- Phù hợp làm baseline mặc định nếu muốn tránh cập nhật toàn bộ encoder.

`full_encoder_and_head`:

- Ưu điểm: retain accuracy và retain balanced accuracy tốt nhất khi quên nhiều nhãn.
- Nhược điểm: forget rate thấp hơn `head_only` một chút, và can thiệp mạnh vào encoder.
- Phù hợp khi quy mô cần quên lớn hơn và ta chấp nhận fine-tune toàn bộ model.

## 10. Kết luận

Tất cả baseline đều đạt mức unlearning cao. So với trước unlearning, `Df-test forget_rate` tăng rõ:

- Quên 1 nhãn: từ `0.4100` lên khoảng `0.9600 - 1.0000`.
- Quên 4 nhãn: từ `0.1322` lên khoảng `0.9818 - 0.9952`.
- Quên 8 nhãn: từ `0.1070` lên khoảng `0.9875 - 0.9933`.

Nếu chọn baseline theo mục tiêu:

- Mục tiêu quên mạnh nhất: chọn `head_only`.
- Mục tiêu cân bằng và can thiệp vừa phải: chọn `last_encoder_block`.
- Mục tiêu giữ performance trên `Dr` tốt nhất khi quên nhiều nhãn: chọn `full_encoder_and_head`.

Trong lần chạy này, `last_encoder_block` là một cấu hình tham khảo để cân bằng quên/giữ; kịch bản quên 8 nhãn nên báo cáo thêm `full_encoder_and_head` vì retain performance tốt hơn. Mốc baseline trước unlearning vẫn là model gốc encoder + MLP. Ba scope cùng dùng một objective CE, không phải ba method unlearning độc lập. Chưa suy ra lựa chọn tối ưu cho instance-wise hoặc xếp hạng method mới từ báo cáo này.

## 11. Lưu ý về resume và artifact

Trong pipeline trước, base model đã có cơ chế reuse:

```text
base_model/best_model.pt
```

Tuy nhiên, cell unlearning hiện chưa có resume chi tiết theo từng `(forget_count, scope, repeat)`. Nếu quá trình đang chạy bị ngắt, chạy lại cell cuối có thể chạy lại và ghi đè các kết quả cũ của unlearning.

Mỗi kết quả unlearning được lưu theo cấu trúc:

```text
unlearning/forget_<N>_labels/repeat_<r>/<scope>/unlearned_model.pt
unlearning/forget_<N>_labels/repeat_<r>/<scope>/result.json
```

Trong đó:

- `<N>` là số nhãn cần quên: `1`, `4`, hoặc `8`.
- `<r>` là repeat index: `0`, `1`, hoặc `2`.
- `<scope>` là một trong ba baseline: `head_only`, `last_encoder_block`, `full_encoder_and_head`.

Nên bổ sung logic skip nếu `result.json` và `unlearned_model.pt` đã tồn tại, để resume thực sự theo từng baseline/repeat.
