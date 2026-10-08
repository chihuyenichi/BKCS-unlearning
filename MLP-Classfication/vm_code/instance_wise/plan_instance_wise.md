# Kế hoạch triển khai Instance-Wise Unlearning cho Known/Unknown

Tài liệu này là kế hoạch triển khai và benchmark pipeline **instance-wise unlearning** cho model phân loại nhị phân traffic:

```text
AOL        -> known   -> binary label 0
icloud_100 -> unknown -> binary label 1
```

Mục tiêu chính: khi nhận một danh sách instance AOL cần quên (`Df`), model sau unlearning phải dự đoán các instance đó thành `unknown`, trong khi giữ hành vi trên phần dữ liệu còn lại gần nhất có thể với base model và retrain oracle.

## Trạng thái implementation đã rà soát

Notebook chính là [train_instance_wise.ipynb](train_instance_wise.ipynb), chạy full experiment trên VM với CSV local. [README vận hành](../README.md) mô tả cấu hình và cách chạy hiện tại.

| Hạng mục | Trạng thái code hiện tại |
| --- | --- |
| Paths/runtime | Hai dataset VM đã cấu hình; repo root tự xác định từ working directory hoặc `BKCS_REPO_ROOT`; checkpoint ở `weight_encoder_trained/`. |
| Parser và instance identity | Byte-offset index, row SHA-256, input `[10000,3]`, mask theo packet size và split deterministic đã có. |
| Base model | `train-base` load encoder pretrained, mặc định freeze encoder và train binary MLP; lưu toàn bộ encoder + classifier. |
| Forget manifests | `make-manifests` chọn nested AOL base-train instances đang được dự đoán đúng là known. |
| Unlearning | Đã có 5 method `relabel_only`, `retain_rehearsal`, `l2ul_mas`, `l2ul_adv`, `l2ul_adv_mas`, 3 update scope và repeat loop. |
| Đánh giá | Đã có forget before/after, retain validation/test theo slice và confusion matrix. Unlearning lưu epoch cuối. |
| Resume | Reuse index/feature/adversarial cache và nạp base checkpoint đã có; resume optimizer/epoch và skip nhánh hoàn thành chưa có. |
| Còn dự kiến | Retrain oracle, command `train-oracle`/`summarize`, tổng hợp mean/std tự động, artifact README riêng và targeted Blindspot. |
| Kết quả trên VM | Cần đọc artifact của lần chạy thực tế; notebook repo chưa chứa output đã lưu, không suy ra trạng thái VM từ đó. |

Các phase, checklist và acceptance criteria bên dưới là **đặc tả đích**. Chỉ đánh dấu checklist hoàn tất khi đã xác nhận implementation và verification cần thiết.

---

## 1. Kết quả cần đạt

Sau khi hoàn thành plan, repository cần có một notebook/pipeline có thể chạy được các bước sau:

1. Kiểm tra schema CSV trên máy cá nhân bằng thư mục mẫu.
2. Train base binary model trên VM từ encoder pretrained.
3. Tạo stable instance ID cho từng CSV row.
4. Tạo và validate forget manifest theo instance ID.
5. Chạy các method unlearning từ cùng một base checkpoint.
6. Đánh giá forget, retain, collateral damage và oracle distance.
7. Lưu artifact đủ để tái lập experiment.

Output hiện có theo cell cấu hình notebook:

```text
artifacts/vm-training/instance-wise/
├── csv_index.json
├── feature_cache/
├── base_run/
│   ├── split_manifest.json
│   ├── base_model/
│   │   ├── best_model.pt
│   │   └── result.json
│   └── instance_unlearning/
│       └── <forget_manifest_stem>/
│           ├── summary.json
│           ├── adversarial_cache/        # khi chạy Adv
│           └── <method>/<scope>/repeat_<r>/
│               ├── unlearned_model.pt
│               ├── result.json           # bao gồm history
│               └── mas_anchors.pt        # khi chạy MAS
└── forget_manifests/
    ├── forget_1.json
    ├── forget_4.json
    ├── forget_10.json
    ├── forget_50.json
    └── forget_100.json
```

Paths mặc định được neo vào repo root. Dataset signatures nằm trong index/split manifest/checkpoint, history nằm trong result; các file riêng `dataset_signatures.json`, `history.json`, `config.json`, importance/manifest/README theo từng run vẫn là yêu cầu mở rộng, chưa phải toàn bộ output hiện có.

---

## 2. Quyết định đã chốt

| Nhóm | Quyết định |
| --- | --- |
| Bài toán | Chỉ xét binary classification `known/unknown`. |
| Forget target | Chỉ quên instance thuộc AOL. Target mới luôn là `unknown=1`. |
| Đơn vị unlearning | Một CSV row là một model instance. |
| Instance ID | Dùng `dataset_id:source:split_file:line_number`, kèm `row_sha256` để xác minh. |
| Split | `train_data.csv` chia deterministic thành base train/validation; `test_data.csv` là final test cố định. |
| Base checkpoint | Mọi nhánh unlearning bắt đầu độc lập từ cùng `base_model/best_model.pt`. |
| Threshold | Giữ `best_unknown_threshold` của base model, không chọn lại sau unlearning. |
| Loss | Loss là objective dùng để backprop và cập nhật trọng số, không chỉ là metric mô tả. |
| Base training | Mặc định freeze encoder pretrained, train MLP; `--fine-tune-encoder` là cấu hình tùy chọn. |
| Strict L2UL | Các mode `l2ul_*` không dùng `Dr` trong loss unlearning. |
| Baseline có `Dr` | `retain_rehearsal` được phép dùng `Dr`, nhưng phải báo cáo tách biệt. |
| Ưu tiên triển khai | `relabel_only` -> `retain_rehearsal` -> `l2ul_mas` -> `l2ul_adv` -> `l2ul_adv_mas`. |

Các điểm chưa chốt được gom ở mục 15.

---

## 3. Phạm vi và ngoài phạm vi

### Trong phạm vi

- Instance-wise unlearning cho AOL instance trong base-training split.
- So sánh nhiều method và nhiều update scope.
- Đánh giá giữ lại hành vi trên:
  - AOL retain cùng original label với `Df`;
  - AOL retain khác original label;
  - icloud_100 unknown;
  - validation/test cố định.
- Retrain oracle để biết model unlearned có gần trạng thái "chưa từng train trên Df" hay không.

### Ngoài phạm vi của phase đầu

- Xóa vật lý dữ liệu gốc.
- Certified deletion hoặc chứng minh privacy mạnh.
- Unlearning theo class/group label.
- Truy ngược CSV row về PCAP nếu chưa có provenance manifest upstream.

Nếu sau này forget request đến từ tên/path PCAP, cần bổ sung manifest upstream dạng:

```text
sample_id, source, split_file, line_number, original_pcap_path, original_label, source_sha256
```

---

## 4. Data contract

### 4.1. Đường dẫn runtime

Notebook được phát triển/review trên máy cá nhân, nhưng full experiment chạy trên VM.

Runtime dataset mặc định trên VM:

```python
AOL_DATASET_DIR = Path("/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud")
UNKNOWN_273_DATASET_DIR = Path("/home/ubuntu/Documents/KF/data_processed/icloud_100")
```

Mỗi directory phải có:

```text
train_data.csv
test_data.csv
label_mapping.csv
```

Thư mục sample trên máy cá nhân chỉ dùng để validate parser/schema:

```text
MLP-Classfication/csv_sample-20260921T063615Z-1-001/csv_sample/icloud/
```

Không được fallback âm thầm từ dataset VM sang sample khi train hoặc unlearn.

### 4.2. CSV row format

Mỗi dòng CSV là một instance:

```text
40,000 số = [label_id, relative_time, direction, packet_size] x 10000
```

Parser phải:

1. Đọc row bằng byte offset.
2. Parse đúng decimal và scientific notation.
3. Kiểm tra đúng `40,000` giá trị.
4. Reshape thành `[10000, 4]`.
5. Lấy `original_label_id` từ field đầu tiên.
6. Kiểm tra label column của packet thật lặp lại đúng `original_label_id`.
7. Nhận diện padding bằng `packet_size == 0`.
8. Loại cột label để đưa encoder input `[10000, 3]`.

Lưu ý riêng cho `label_id=0`:

- packet thật và padding đều có label column bằng 0;
- không được dùng `labels != 0` để xác định packet thật;
- tên label rỗng trong `label_mapping.csv` phải map ổn định thành `label_0`.

### 4.3. Dataset signature

Trước full run trên VM, preflight phải kiểm tra và lưu:

```text
resolved path
file size
mtime_ns
optional full-file SHA-256
label_mapping hash
feature/parser config version
```

Absolute path chỉ dùng cho provenance/signature, không đưa vào logic của `instance_id`.

---

## 5. Model và checkpoint

Model runtime:

```text
CSV row / traffic instance [10000, 3]
-> pretrained RawPacketEncoder
-> embedding 256 chiều
-> MLP classifier
-> 2 logits: known / unknown
```

Checkpoint SupCon hiện có:

```text
MLP-Classfication/vm_code/weight_encoder_trained/pretrain_AOL.pth
MLP-Classfication/vm_code/weight_encoder_trained/pretrain_273.pth
```

Các checkpoint này chứa encoder/projection head SupCon, không chứa binary MLP. Pipeline chính dùng:

```text
pretrain_AOL.pth
-> load encoder
-> train binary MLP trên AOL known + icloud_100 unknown
-> lưu base_model/best_model.pt
```

`best_model.pt` hiện chứa:

```text
model_state_dict
best_unknown_threshold
unknown_threshold
split_ids
dataset signatures
pretrain metadata / checkpoint hash
model/feature config
seed / valid_ratio
base test metrics
```

Validation history và threshold sweep nằm trong `base_model/result.json`. Hash riêng của split manifest và base validation metrics trong checkpoint là yêu cầu có thể bổ sung, không phải khóa đã lưu hiện tại.

Mọi method/scope/repeat của unlearning phải reload độc lập từ checkpoint này.

---

## 6. Instance identity và manifest

### 6.1. Instance ID

ID logic:

```text
{dataset_id}:{source}:{split_file}:{line_number}
```

Ví dụ:

```text
aol_v1:AOL:train_data.csv:1234
```

Quy ước:

- `line_number` là số dòng vật lý 1-based.
- Không dùng `byte_offset` làm ID chính vì offset dễ đổi khi CSV được tạo lại.
- Không dùng `original_label_id + thứ tự trong label` vì số instance mỗi label không đồng đều.
- `row_sha256` dùng để xác minh đúng nội dung row.

### 6.2. Split manifest

`split_manifest.json` phải được tạo một lần lúc train base model và tái sử dụng nguyên vẹn.

Protocol:

```text
train_data.csv:
    split deterministic theo (source, original_label)
    85% base_train
    15% validation

test_data.csv:
    final test cố định
    không dùng để train, chọn hyperparameter hoặc chọn epoch
```

Với unlearning:

```text
Df_train = forget IDs giao với base_train
Dr_train = base_train trừ Df_train
```

Không tự động đưa validation/test cùng nhãn vào `Df`; các instance này là control để đo collateral damage.

### 6.3. Forget manifest

Ví dụ file:

```json
{
  "dataset_id": "aol_v1",
  "source": "AOL",
  "split_file": "train_data.csv",
  "dataset_signature": {
    "bytes": 0,
    "mtime_ns": 0,
    "sha256": "optional"
  },
  "instances": [
    {
      "instance_id": "aol_v1:AOL:train_data.csv:1234",
      "line_number": 1234,
      "row_sha256": "...",
      "original_label": "label_7"
    }
  ]
}
```

Validation phải fail sớm nếu:

- instance không thuộc AOL;
- instance không thuộc base train;
- line không tồn tại;
- row hash không khớp;
- instance ID bị trùng;
- instance không xuất hiện trong split manifest của base checkpoint;
- manifest dataset signature không khớp checkpoint.

---

## 7. Pipeline triển khai

### Phase 0 - Schema validation

Mục tiêu: chứng minh parser/index hoạt động với CSV sample trên máy cá nhân.

Checklist:

- đọc được `label_mapping.csv`;
- build byte-offset index cho `train_data.csv` và `test_data.csv`;
- decode được row bất kỳ thành `[10000, 3]`;
- xử lý đúng `label_id=0`;
- lưu báo cáo schema validation.

Output:

```text
artifacts/.../schema_validation.json
```

Đây là artifact dự kiến; command `validate-schema` hiện in JSON ra output notebook, chưa tự lưu file `schema_validation.json`.

### Phase 1 - Base training

Mục tiêu: tạo base binary checkpoint dùng chung cho mọi nhánh unlearning.

Checklist:

- preflight dataset VM;
- tạo `CsvRecord` cho AOL và icloud_100;
- tạo `split_manifest.json`;
- load encoder từ `pretrain_AOL.pth`;
- train binary MLP;
- chọn `best_unknown_threshold` trên validation;
- evaluate base trên validation/test;
- lưu `base_model/best_model.pt`.

Output:

```text
base_model/best_model.pt
base_model/result.json
split_manifest.json
```

Dataset signatures được nhúng trong manifest/checkpoint, base test metrics và lịch sử nằm trong result. Không cần tìm `dataset_signatures.json`/`base_metrics.json` riêng trong implementation hiện tại.

### Phase 2 - Forget manifest generation

Mục tiêu: tạo các tập quên nested và tái lập được.

Mặc định:

```text
FORGET_INSTANCE_COUNTS = (1, 4, 10, 50, 100)
```

Quy tắc chọn:

- chỉ chọn AOL instance thuộc base train;
- ưu tiên instance base model đang dự đoán đúng là `known`;
- stratified trên nhiều `original_label` để tránh biến thành group forgetting;
- tập lớn là prefix mở rộng của tập nhỏ:

```text
Df_1 subset Df_4 subset Df_10 subset Df_50 subset Df_100
```

Output:

```text
forget_manifests/forget_1.json
forget_manifests/forget_4.json
forget_manifests/forget_10.json
forget_manifests/forget_50.json
forget_manifests/forget_100.json
```

### Phase 3 - Instance-wise unlearning

Vòng lặp chính:

```text
for forget_count:
  for method:
    for update_scope:
      for optimization_seed:
        load base_model/best_model.pt
        validate forget manifest
        freeze/unfreeze theo scope
        build optimizer mới
        run unlearning
        evaluate
        save artifacts
```

Mỗi batch unlearning chạy:

```text
forward -> compute L_total -> backward -> optimizer.step
```

Chỉ parameter có `requires_grad=True` theo update scope mới nhận gradient và được cập nhật.

### Phase 4 - Retrain oracle

Mục tiêu: có mốc so sánh cho từng forget set.

Với mỗi `Df`:

```text
oracle_train_set = Dr_train
```

Train lại cùng cấu hình base model nhưng loại đúng `Df`. Oracle tốn thời gian nên có thể chạy sau smoke test, nhưng cần cho benchmark chính thức.

### Phase 5 - Benchmark summary

Tổng hợp theo:

```text
forget_count
method
scope
optimization_seed
```

Báo cáo:

- mean/std theo seed;
- forget-retain trade-off;
- collateral damage theo original label;
- khoảng cách tới retrain oracle;
- artifact path và checkpoint hash.

---

## 8. Methods và objective

Ký hiệu:

```text
theta_0 : trọng số base model trước unlearning
theta_u : trọng số model sau unlearning
Df      : đúng các instance được yêu cầu quên
Dr      : toàn bộ base-train còn lại
y_old   : known = 0
y_star  : unknown = 1
```

Vì `Df` chỉ chọn từ AOL:

```text
model_theta_0(x_f) = known
model_theta_u(x_f) = unknown
```

### 8.1. Base model

Không train thêm. Dùng để đo metric trước unlearning.

### 8.2. Retrain oracle

Train lại model với:

```text
base train oracle = Dr_train
```

Oracle không phải method nhanh, mà là mốc để so sánh.

Loại `Df` khỏi training không bảo đảm oracle dự đoán các instance đó thành unknown. Đây là mốc cho mục tiêu loại ảnh hưởng dữ liệu, cần phân biệt với yêu cầu đổi output sang unknown. Nếu reuse encoder pretrained đã từng thấy `Df`, oracle chỉ là retrain classifier có điều kiện trên encoder đó; phải kiểm tra provenance trước khi diễn giải thành model chưa từng thấy `Df`.

### 8.3. Relabel-only

Targeted relabeling đơn giản:

```text
L_forget = CE(f_theta_u(x_f), unknown)
L_total  = L_forget
```

### 8.4. Retain-rehearsal

Baseline được phép dùng `Dr` trong loss:

```text
L_total = CE(f_theta_u(Df), unknown)
        + lambda_retain * CE(f_theta_u(Dr), original_binary_label)
```

Phải báo cáo tên riêng là `retain_rehearsal`, không gộp với strict L2UL.

### 8.5. L2UL-MAS

Tính MAS importance từ `theta_0` và chỉ từ `Df`:

```text
Omega_i = (1 / |Df|) * sum_x | d ||f_theta_0(x)||_2 / d theta_i |
```

Implementation đầu tiên dùng logits binary làm `f_theta_0(x)`. Nếu dùng embedding, phải đặt tên ablation riêng.

Code hiện dùng trị tuyệt đối của gradient norm tổng trên mini-batch rồi chia tổng số mẫu. Đây là ước lượng theo batch, có thể khác trung bình trị tuyệt đối gradient từng instance trong công thức trên do triệt tiêu gradient; chưa coi hai cách là tương đương chính xác.

Chuẩn hóa theo từng parameter tensor:

```text
Omega_norm_i = min_max_normalize(Omega_i)
anchor_i     = 1 - Omega_norm_i
```

Ý nghĩa:

- parameter liên quan mạnh đến `Df` có anchor nhỏ, được phép thay đổi;
- parameter ít liên quan đến `Df` có anchor lớn, bị giữ gần `theta_0`.

Loss:

```text
L_MAS   = sum_i anchor_i * (theta_u_i - theta_0_i)^2 / 2
L_total = L_forget + lambda_mas * L_MAS
```

Chỉ tính MAS cho parameter trainable trong scope hiện tại.

### 8.6. L2UL-Adv

Sinh targeted adversarial example từ frozen base model:

```text
x_adv = x_f + delta
target = unknown
```

Khi sinh `x_adv`:

- base model ở `eval()`;
- không cập nhật trọng số;
- chỉ cập nhật input adversarial;
- tối thiểu hóa `CE(base_model(x_adv), unknown)`;
- project perturbation vào miền hợp lệ sau mỗi bước.

Loss:

```text
L_adv   = CE(f_theta_u(x_adv), unknown)
L_total = L_forget + lambda_adv * L_adv
```

Cache adversarial theo:

```text
base checkpoint SHA-256
instance_id
PGD config
feature contract version
```

### 8.7. L2UL-Adv+MAS

Method đầy đủ:

```text
L_total = L_forget
        + lambda_adv * L_adv
        + lambda_mas * L_MAS
```

Không có `CE(Dr)` trong mode này.

### 8.8. Ràng buộc adversarial cho traffic

Không dùng một epsilon chung cho cả ba feature vì scale và semantics khác nhau.

Ràng buộc ban đầu:

```text
padding:
    giữ nguyên, luôn bằng 0

direction:
    giữ nguyên, không perturb biến rời rạc

relative_time:
    perturb nhỏ trong raw CSV feature space
    clamp không âm
    giữ quy tắc preprocessing

packet_size:
    perturb nhỏ trong raw CSV feature space
    clamp vào miền kích thước hợp lệ
    tùy config có thể round về số nguyên
```

Milestone an toàn: triển khai và benchmark `l2ul_mas` trước, sau đó mới bật constrained PGD.

PGD hiện perturb trực tiếp `legacy_raw_v1`: epsilon time mặc định `0.005`, epsilon packet size `8.0`. Không có bước scale channel ngầm; các bound cần được xác nhận theo semantics của CSV. Code giữ direction/padding, timestamp đầu, ràng buộc timestamp không giảm và packet size hợp lệ.

---

## 9. Update scopes

Giữ ba scope:

| Scope | Parameter được cập nhật | Ghi chú |
| --- | --- | --- |
| `head_only` | Chỉ MLP classifier | Encoder `eval()`. Hợp lệ cho behavioral forgetting nhưng không chứng minh encoder đã quên. |
| `last_encoder_block` | Block cuối encoder + MLP | Block đã freeze giữ `eval()`, block mở và MLP train mode. |
| `full_encoder_and_head` | Toàn bộ encoder + MLP | Blast radius lớn nhất, cần kiểm tra retain kỹ. |

Với mỗi scope:

1. Load lại base checkpoint.
2. Freeze toàn bộ parameter.
3. Mở `requires_grad=True` đúng scope.
4. Tính MAS riêng cho parameter trainable nếu method cần MAS.
5. Tạo optimizer mới.
6. Chạy unlearning độc lập.
7. Kiểm tra frozen parameter không đổi.

---

## 10. Metrics và tiêu chí thành công

Giữ threshold của base model cố định cho mọi nhánh.

### 10.1. Forget metrics

```text
forget_success_rate:
    P(predicted unknown | Df_exact)

forget_confidence_shift:
    mean P_unknown_after(Df) - mean P_unknown_before(Df)

old_label_accuracy_on_Df:
    P(predicted known | Df_exact), cần giảm
```

### 10.2. Retain metrics

```text
same_label_known_recall:
    known recall trên AOL retain cùng original_label với Df

remaining_known_recall:
    known recall trên toàn bộ AOL retain

unknown_recall:
    unknown recall trên icloud_100

retain_balanced_accuracy:
    (remaining_known_recall + unknown_recall) / 2
```

### 10.3. Collateral damage

```text
same_label_damage = same_label_known_recall_before
                  - same_label_known_recall_after

known_damage      = remaining_known_recall_before
                  - remaining_known_recall_after

balanced_damage   = retain_balanced_accuracy_before
                  - retain_balanced_accuracy_after
```

### 10.4. Oracle comparison

So sánh model unlearned với retrain oracle bằng:

- chênh lệch metric trên retain test;
- prediction agreement;
- khoảng cách probability/logit distribution;
- tùy chọn: embedding distance trên retain set.

### 10.5. Tiêu chí sàng lọc ban đầu

Giá trị tạm dùng cho development run:

```text
forget_success_rate >= 0.95
retain_balanced_accuracy drop <= 0.02
same_label_known_recall drop <= 0.05
```

Không chỉ báo cáo accuracy tổng vì mất cân bằng known/unknown có thể che giấu lỗi.

---

## 11. Implementation backlog

Notebook triển khai chính:

```text
MLP-Classfication/vm_code/instance_wise/train_instance_wise.ipynb
```

### P0 - Instance identity và base run

- [ ] Tách runtime VM config khỏi local sample config.
- [ ] Implement dataset preflight.
- [ ] Build byte-offset index và `CsvRecord`.
- [ ] Thêm `instance_id`, `row_sha256`, `dataset_signature_id`.
- [ ] Decode row thành tensor `[10000, 3]`.
- [ ] Tạo deterministic split manifest.
- [ ] Train và lưu base binary checkpoint.

### P1 - Instance-wise unlearning baseline

- [ ] Load/validate forget manifest.
- [ ] Implement `forget_sets_by_instance`.
- [ ] Implement retain slices:
  - `Df_exact`;
  - `Dr_same_label_val/test`;
  - `Dr_other_known_val/test`;
  - `Unknown_val/test`;
  - `Full_retain_val/test`.
- [ ] Implement update scopes.
- [ ] Chạy `relabel_only`.
- [ ] Chạy `retain_rehearsal`.
- [ ] Lưu history/result/artifact theo grid.

### P2 - MAS

- [ ] Implement `estimate_mas_importance`.
- [ ] Implement `invert_and_normalize_importance`.
- [ ] Implement `parameter_anchor_loss`.
- [ ] Chạy `l2ul_mas` trên cả ba scope.
- [ ] Kiểm tra MAS chỉ gồm parameter trainable.

### P3 - Adversarial regularization

- [ ] Chốt feature scale theo channel.
- [ ] Implement constrained targeted PGD.
- [ ] Kiểm tra padding/direction không đổi.
- [ ] Implement adversarial cache/manifest.
- [ ] Chạy `l2ul_adv`.
- [ ] Chạy `l2ul_adv_mas`.

### P4 - Benchmark và oracle

- [ ] Chạy nested forget counts.
- [ ] Chạy tối thiểu 3 optimization seeds.
- [ ] Train retrain oracle cho từng forget set.
- [ ] Tổng hợp mean/std và export summary.
- [ ] Viết README cho từng run.

---

## 12. Kiểm thử bắt buộc

### Unit/schema checks

- Một `instance_id` map đúng một CSV row.
- Row hash khớp sau khi seek bằng byte offset.
- Row có đúng `40,000` trường.
- Parser chấp nhận decimal và scientific notation.
- Padding được mask bằng `packet_size == 0`.
- Row `label_id=0` vẫn load/split bình thường.
- Tên rỗng của label 0 map thành `label_0`.
- Code không giả định mọi original label có cùng số instance.
- `Df` và `Dr` không giao nhau.
- `Df union Dr` bằng base train.
- Mọi `Df` đều thuộc AOL/base train.
- Instance cùng original label nhưng không nằm trong manifest vẫn thuộc retain.
- Frozen parameter không đổi sau unlearning.
- Mỗi scope chỉ thay đổi parameter được phép.
- Base threshold được giữ nguyên.

### PGD checks

- Padding giữ nguyên.
- Direction giữ nguyên.
- `relative_time` không âm.
- `packet_size` trong miền hợp lệ.
- Perturbation nằm trong bound theo từng feature.

### Smoke test

Chạy trước benchmark:

```text
forget_count = 1
optimization_seed = 1 seed
epoch = 1
scope = head_only
methods = relabel_only, l2ul_mas
```

Smoke test pass khi:

- artifact được ghi đầy đủ;
- loss giảm hoặc có gradient hợp lệ;
- metric before/after được ghi;
- frozen parameter không đổi;
- không dùng test set trong selection.

---

## 13. Notebook commands

Notebook có cell cấu hình cuối:

```python
RUN_PIPELINE = False
NOTEBOOK_COMMAND = "validate-schema"
NOTEBOOK_ARGUMENTS_BY_COMMAND = {...}
```

Khi muốn chạy, review config rồi đặt:

```python
RUN_PIPELINE = True
```

Command hiện có và command dự kiến:

| Command | Mục đích | Môi trường | Trạng thái |
| --- | --- | --- | --- |
| `validate-schema` | Kiểm tra schema CSV sample | Máy cá nhân; VM nếu đã có sample | Đã có |
| `train-base` | Train base binary model | VM | Đã có |
| `make-manifests` | Tạo nested forget manifests | VM | Đã có |
| `unlearn` | Chạy unlearning grid cho một manifest | VM | Đã có |
| `train-oracle` | Train retrain oracle | VM | Chưa triển khai |
| `summarize` | Tổng hợp benchmark | Máy cá nhân hoặc VM | Chưa triển khai |

Cấu hình notebook mặc định `RUN_PIPELINE=False` và `validate-schema`. Khi train trên VM, chọn `train-base`, bật `RUN_PIPELINE`, chạy cell cấu hình rồi cell execute. `unlearn` mặc định quên 4 instance bằng `forget_4.json`; các count khác phải đổi manifest và chạy riêng.

Mapping code dự kiến:

```text
CSV row identity:
    CsvRecord, make_instance_id, index_one_csv

Data validation:
    decode_csv_record, command_validate_schema

Base encoder + MLP:
    FlowEncoder, FlowModel, command_train_base

Instance selection:
    command_make_manifests, validate_forget_manifest

MAS:
    estimate_mas_importance
    inverted_mas_anchors
    parameter_anchor_loss

Traffic-constrained adversarial examples:
    generate_targeted_adversarial
    constrain_adversarial

Update scopes:
    set_scope
    set_unlearning_train_mode

Metrics:
    forget_metrics
    evaluate
    evaluate_slices
```

Notebook chỉ dùng Python standard library, NumPy và PyTorch. Full training/unlearning mặc định yêu cầu CUDA; CPU chỉ phù hợp cho parser/schema checks hoặc debug rất nhỏ.

---

## 14. Rủi ro

| Rủi ro | Ảnh hưởng | Cách giảm |
| --- | --- | --- |
| Không có PCAP provenance | Không chứng minh được CSV row đến từ file PCAP nào | Chỉ claim `1 CSV row = 1 model instance`; yêu cầu upstream manifest nếu cần xóa theo PCAP. |
| `head_only` chỉ đổi classifier | Encoder có thể vẫn giữ representation của `Df` | Báo cáo scope rõ ràng; không diễn giải thành xóa khỏi encoder. |
| PGD không hợp lệ về traffic semantics | Kết quả `Adv` có thể khó so sánh paper | Triển khai MAS trước; kiểm tra ràng buộc feature trước khi claim. |
| Dùng `Dr` lẫn với strict setting | Làm sai kết luận paper-style | Tách `retain_rehearsal` khỏi `l2ul_*`. |
| Threshold bị chọn lại sau unlearning | Metric bị optimistic | Giữ threshold base cố định. |
| Forget set vô tình thành group-wise | Không còn là instance-wise | Chọn stratified và giữ same-label retain control. |

---

## 15. Cần xác nhận trước benchmark chính thức

1. Forget request thực tế sẽ cung cấp line number, sample ID hay đường dẫn PCAP?
2. Có manifest nào ánh xạ CSV row về PCAP gốc không?
3. Benchmark dùng các count `(1, 4, 10, 50, 100)` hay cần thêm theo tỷ lệ phần trăm? Count mặc định khi chạy unlearning là 4.
4. Strict experiment có cấm dùng `Dr` cả trong model selection/early stopping không, hay chỉ cấm trong loss?
5. Khi triển khai oracle, giữ cùng cấu hình base đã chọn; nếu cần oracle full encoder+MLP hoặc encoder pretrain loại `Df`, báo cáo như một protocol riêng và xác minh provenance.

---

## 16. Acceptance criteria

Plan được xem là hoàn thành khi:

1. `validate-schema` chạy được trên máy cá nhân bằng CSV sample.
2. VM tạo được `best_model.pt` kèm split manifest và dataset signatures.
3. Forget manifest validate được bằng row hash và split membership.
4. Smoke test chạy được với `relabel_only` và `l2ul_mas`.
5. Mỗi unlearning branch reload từ cùng base checkpoint.
6. Result ghi đủ before/after metrics cho forget và retain slices.
7. Frozen parameter checks pass cho từng scope.
8. Benchmark summary phân biệt rõ baseline, strict L2UL methods và retrain oracle.
