# Kế hoạch và đặc tả Instance-Wise Unlearning trên Kaggle

## 0. Trạng thái tài liệu

Tài liệu này được đồng bộ với implementation hiện tại trong `pipeline.ipynb`. Notebook là nguồn chuẩn khi có khác biệt giữa tài liệu và code.

Pipeline hiện tại gồm bốn phase:

1. `validate-schema`
2. `train-base`
3. `make-manifests`
4. `unlearn`

Benchmark hiện tại gồm mười hai **mode độc lập**:

`baseline, only_head, full_mlp, last_encoder_block, full_model, l2ul_neggrad, l2ul_adv, l2ul_adv_imp, amnesiac, blindspot, bnd, mkmmd`

Quy tắc quan trọng:

- Mỗi tên mode ánh xạ tới đúng một objective và một phạm vi tham số được phép cập nhật.
- Không chạy tích Descartes giữa “method” và “scope”.
- `only_head`, `full_mlp`, `last_encoder_block` và `full_model` là bốn mode riêng.
- Ba mode `l2ul_*` triển khai riêng phương pháp instance-wise trong paper *Learning to Unlearn*; chúng không được tự động ghép với scope hay method khác.
- `amnesiac`, `blindspot`, `bnd` và `mkmmd` cũng là bốn mode riêng, mỗi mode đã có scope cố định.
- Ví dụ không tồn tại các cấu hình tự động như `blindspot + only_head` hay `mkmmd + full_mlp`.
- Command method-by-scope cũ còn tồn tại để tương thích artifact cũ nhưng không được đăng ký trong parser. Ba mode `l2ul_*` active dùng implementation mới và runner riêng.

## 1. Mục tiêu

Xây dựng benchmark instance-wise machine unlearning cho bài toán phân loại traffic nhị phân trên Kaggle.

Model gốc học cách phân biệt:

| Binary class | Giá trị | Nguồn dữ liệu |
|---|---:|---|
| Known | `0` | AOL |
| Unknown | `1` | 100 Top Search |

Sau khi base model được huấn luyện, một tập con các instance AOL trong base-train được chọn làm tập cần quên. Mục tiêu của unlearning là:

- Làm các instance cần quên được model xử lý như `UNKNOWN`.
- Giữ chất lượng dự đoán trên dữ liệu còn lại càng gần base model càng tốt.
- Đo rõ collateral damage trên các nhóm known và unknown.
- So sánh các mode từ cùng một base checkpoint, cùng threshold, cùng forget manifest và cùng protocol đánh giá.

Đây là **instance-wise unlearning**. Đơn vị bị quên là một dòng CSV cụ thể, không phải toàn bộ class gốc của AOL.

## 2. Ký hiệu và định nghĩa

| Ký hiệu | Định nghĩa |
|---|---|
| `D_train` | Tập train nhị phân đã tách từ hai file `train_data.csv` |
| `D_val` | Tập validation được tách từ source-train |
| `D_test` | Hai file `test_data.csv` ghép lại, không tham gia huấn luyện |
| `Df` | Forget set, gồm đúng các AOL instance được liệt kê trong forget manifest |
| `Dr` | Retain-train, bằng `D_train \ Df` |
| `Du_ref` | Các instance unknown thuộc `Dr`, dùng làm unknown reference cho MKMMD |
| `f_theta0` | Base model được load từ cùng một base checkpoint |
| `f_theta` | Branch model đang được unlearn |
| `z_theta(x)` | Embedding 256 chiều do encoder sinh ra |
| `l_theta(x)` | Hai logits theo thứ tự `[KNOWN, UNKNOWN]` |
| `p_theta(x)` | Softmax probability từ hai logits |
| `tau` | Unknown threshold được chọn trên validation của base model |
| `y` | Binary label thật, `0` hoặc `1` |
| `y_f` | Nhãn gốc của instance trong `Df`, luôn là `KNOWN=0` |
| `y_target` | Target unlearning cho `Df`, luôn là `UNKNOWN=1` khi mode dùng relabel |

Điều kiện tập dữ liệu:

~~~
Df subset D_train
Dr = D_train \ Df
Df intersection Dr = empty
Df union Dr = D_train
Df chỉ chứa AOL/KNOWN instance
~~~

## 3. Phạm vi bài toán

### 3.1 Trong phạm vi

- Đọc processed CSV đã được mount vào Kaggle.
- Tạo identity ổn định cho từng physical CSV row.
- Train base binary classifier từ pretrained AOL encoder.
- Chọn các instance AOL được base model dự đoán đúng để tạo forget manifest.
- Chạy các mode unlearning độc lập từ cùng base checkpoint.
- Đánh giá forget efficacy, retain utility và lỗi toàn bộ test.
- Lưu đủ provenance để truy ngược dataset, checkpoint, manifest và seed.

### 3.2 Ngoài phạm vi hiện tại

- Không đọc raw PCAP.
- Không xóa toàn bộ original class.
- Không có class-removal protocol.
- Không train một model mới từ đầu trên `Dr` làm retraining oracle.
- Không chứng minh formal privacy guarantee.
- Không tự động ghép paper method với nhiều update scope.
- Không chạy command method-by-scope cũ với `relabel_only`, `retain_rehearsal`, `l2ul_mas` hoặc `l2ul_adv_mas` trong benchmark active.

## 4. Kaggle inputs và filesystem

### 4.1 Dataset cần attach

Notebook cần ba nguồn input:

| Input | Nội dung cần có |
|---|---|
| AOL known dataset | Thư mục `icloud` |
| 100 Top Search unknown dataset | Thư mục `icloud_100` |
| Pretrained encoder dataset | Đúng một file `pretrain_AOL.pth` |

Mỗi thư mục dataset phải có đủ:

~~~
train_data.csv
test_data.csv
label_mapping.csv
~~~

Notebook tìm đệ quy bên dưới `../../../artifacts/workspace-archive/MLP-Classfication/csv_sample-20260921T063615Z-1-001/csv_sample`:

- `resolve_dataset_dir("icloud")` phải tìm đúng một thư mục hợp lệ.
- `resolve_dataset_dir("icloud_100")` phải tìm đúng một thư mục hợp lệ.
- `resolve_unique_file("pretrain_AOL.pth")` phải tìm đúng một file.

Nếu không tìm thấy hoặc tìm thấy nhiều hơn một match, pipeline dừng để tránh dùng nhầm dữ liệu.

### 4.2 Quy tắc đường dẫn

| Loại | Root |
|---|---|
| Input chỉ đọc | `../../../artifacts/workspace-archive/MLP-Classfication/csv_sample-20260921T063615Z-1-001/csv_sample` |
| Working/output | `../../../working` |
| Output riêng của pipeline | `../../../working/bkcs-unlearning` |

Input path phải nằm dưới `../../../artifacts/workspace-archive/MLP-Classfication/csv_sample-20260921T063615Z-1-001/csv_sample`. Output path phải là một thư mục con chuyên biệt dưới `../../../working`, không được trỏ trực tiếp vào root.

## 5. Data contract

### 5.1 `label_mapping.csv`

Header phải đúng thứ tự:

~~~csv
label_name,label_id
~~~

Các ràng buộc:

- `label_id` chuyển được sang integer.
- Không được trùng `label_id`.
- Mapping không được rỗng.
- Nếu `label_name` rỗng, code dùng fallback `label_<id>`.

### 5.2 Traffic CSV

File traffic không có header. Mỗi physical line là một model instance.

Hằng số:

| Tên | Giá trị |
|---|---:|
| `MAX_PACKETS` | 10,000 |
| `RAW_FIELDS_PER_PACKET` | 4 |
| `ROW_WIDTH` | 40,000 |
| `PACKET_FEATURES` | 3 |

Mỗi packet có dạng:

~~~
[label_id, relative_time, direction, packet_size]
~~~

Một dòng CSV được reshape thành:

~~~
[10000, 4]
~~~

Model nhận:

~~~
features = [relative_time, direction, packet_size]  # shape [10000, 3]
mask     = packet_size != 0                        # shape [10000]
~~~

Các ràng buộc khi decode:

- Dòng phải có đúng 40,000 giá trị số.
- Mọi real packet phải có `label_id` khớp với record.
- Padding được nhận biết qua `packet_size == 0`.
- Mỗi padding packet phải đúng `[0, 0, 0, 0]`.
- SHA256 của dòng phải khớp với hash lúc lập index.

Feature transform hiện tại là `legacy_raw_v1`. Code không thêm normalization mới trong notebook.

## 6. Identity, index, cache và split

### 6.1 Instance identity

Mỗi dòng được biểu diễn bằng `CsvRecord`:

| Field | Ý nghĩa |
|---|---|
| `instance_id` | ID ổn định của instance |
| `dataset_id` | `aol_v1` hoặc `icloud100_v1` |
| `csv_path` | Đường dẫn tuyệt đối tới source CSV |
| `byte_offset` | Vị trí byte bắt đầu dòng |
| `line_number` | Số dòng, bắt đầu từ 1 |
| `row_sha256` | SHA256 nội dung dòng |
| `source` | `AOL` hoặc `100_TOPSEARCH` |
| `original_label` | Tên class gốc |
| `original_label_id` | ID class gốc |
| `binary_label` | `KNOWN=0` hoặc `UNKNOWN=1` |
| `split_file` | `train_data.csv` hoặc `test_data.csv` |

Công thức ID:

~~~
instance_id = dataset_id:source:split_file:line_number
~~~

### 6.2 CSV byte-offset index

Index được lưu tại:

~~~
../../../working/bkcs-unlearning/csv_index.json
~~~

Version:

~~~
instance-wise-kaggle-csv-index-v1
~~~

Index chứa source-train, test, byte offset, row hash và dataset signatures. Cache index chỉ được tái sử dụng khi version và toàn bộ signatures khớp.

Dataset signature hiện gồm:

- Absolute path.
- Kích thước file theo byte.
- `mtime_ns`.

Loader seek trực tiếp tới `byte_offset`, không load toàn bộ CSV vào RAM.

### 6.3 Tensor feature cache

Feature cache tạm thời nằm ngoài thư mục output được Kaggle lưu version:

~~~
/kaggle/temp/bkcs-unlearning-feature-cache
~~~

Mỗi cache file là một `.pt` chứa:

~~~
features: FloatTensor [10000, 3]
mask: BoolTensor [10000]
~~~

Cache key phụ thuộc vào:

~~~
FEATURE_TRANSFORM
ROW_WIDTH
instance_id
row_sha256
~~~

Vì vậy thay đổi transform, row width, identity hoặc nội dung dòng sẽ sinh cache key khác.

### 6.4 Train, validation và test

Source-train là hợp của:

- AOL `train_data.csv` với binary label `KNOWN`.
- 100 Top Search `train_data.csv` với binary label `UNKNOWN`.

Source-test là hợp của hai file `test_data.csv` và được giữ cố định.

`D_train` vẫn giữ toàn bộ record của cả hai nguồn. Pipeline không downsample
UNKNOWN và không thay đổi membership của train split. Riêng loader dùng để train
base model tạo các batch có tỉ lệ `KNOWN:UNKNOWN = 1:1`; lớp ít mẫu hơn được
sample lặp theo seed, còn lớp nhiều mẫu hơn được đi qua toàn bộ trong mỗi epoch.
Validation, test, forget set và retain set luôn giữ phân phối và membership gốc.

Source-train được group theo:

~~~
(source, original_label)
~~~

Trong từng group:

~~~
n_val = max(1, round(group_size * valid_ratio)) nếu group_size >= 3
n_val = 0 nếu group_size < 3
~~~

Split dùng seed mặc định `42` và `valid_ratio=0.15`. Sau split, code kiểm tra train và validation không giao nhau và đều hợp lệ.

## 7. Model và base training

### 7.1 Encoder

`FlowEncoder` tương thích với pretrained checkpoint:

| Block | Channels |
|---|---|
| Block 1 | 3 -> 32 |
| Block 2 | 32 -> 64 |
| Block 3 | 64 -> 128 |
| Block 4 | 128 -> 256 |

Mỗi block có:

- Hai `Conv1d` kernel size 8.
- `BatchNorm1d`.
- `MaxPool1d(8, stride=4)`.
- Dropout 0.1.

Sau bốn block, encoder flatten và dùng fully connected layer để sinh embedding 256 chiều.

Checkpoint `pretrain_AOL.pth` chỉ nạp các tham số `encoder.*` và yêu cầu khớp nghiêm ngặt.

### 7.2 Binary MLP classifier

Classifier có hidden dimensions:

~~~
256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 2
~~~

Mỗi hidden stage gồm:

~~~
Linear -> LayerNorm -> GELU -> Dropout
~~~

Dropout mặc định là `0.2`. Layer cuối sinh hai logits `[KNOWN, UNKNOWN]`.

`FlowModel.forward` trả về:

~~~
(logits, embedding)
~~~

### 7.3 Objective base training

Base model dùng cross entropy không gắn class weight:

~~~
L_base = CE(logits, binary_label)
~~~

Việc cân bằng được thực hiện ở sampler thay vì loss. Mỗi batch có số KNOWN và
UNKNOWN bằng nhau, vì vậy `batch_size` phải là số chẵn và tối thiểu bằng 2. Mỗi
epoch lấy `max(n_known, n_unknown)` mẫu cho từng lớp. Sampler dùng seed cố định
và thay đổi thứ tự theo epoch; không loại record khỏi train pool. Nếu base-train
thiếu một binary class, pipeline dừng.

Optimizer:

| Tham số | Mặc định |
|---|---:|
| Optimizer | AdamW |
| Epochs | 16 |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Workers | 2 |

Mặc định encoder bị freeze và chỉ MLP được train. Flag `--fine-tune-encoder` cho phép train cả encoder.

Best base state được chọn theo validation balanced accuracy với threshold 0.5.

### 7.4 Chọn unknown threshold

Sau khi restore best base state, notebook sweep:

~~~
0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
~~~

Threshold được chọn theo thứ tự ưu tiên:

1. Balanced accuracy cao hơn.
2. Unknown recall cao hơn.
3. Accuracy cao hơn.

`D_test` chỉ được đánh giá sau khi chọn threshold. Threshold này được lưu vào checkpoint và giữ cố định cho mọi mode unlearning.

## 8. Bốn phase của pipeline

### 8.1 Phase 1: `validate-schema`

Mục tiêu:

- Resolve đúng hai dataset folder và pretrained file.
- Kiểm tra ba file bắt buộc trong mỗi dataset.
- Kiểm tra header mapping.
- Đọc một số dòng mẫu và xác minh width, label, padding, byte offset và row hash.
- Ghi provenance của pretrained encoder.

Input chính:

| Argument | Mặc định |
|---|---|
| `--aol-dir` | Auto-resolve thư mục `icloud` |
| `--unknown-dir` | Auto-resolve thư mục `icloud_100` |
| `--output-dir` | `../../../working/bkcs-unlearning` |
| `--rows` | 2 |

Output:

~~~
../../../working/bkcs-unlearning/schema_validation.json
~~~

### 8.2 Phase 2: `train-base`

Mục tiêu:

- Build hoặc reuse CSV index.
- Tạo train/validation/test split.
- Load pretrained AOL encoder.
- Train binary MLP base model.
- Chọn best model và unknown threshold.
- Đánh giá base test.

Input bổ sung:

| Argument | Mặc định |
|---|---:|
| `--pretrained-encoder` | Auto-resolve `pretrain_AOL.pth` |
| `--epochs` | 16 |
| `--learning-rate` | 2e-4 |
| `--weight-decay` | 1e-4 |
| `--dropout` | 0.2 |
| `--threshold-grid` | 0.30 đến 0.70, bước 0.05 |

Output:

~~~
../../../working/bkcs-unlearning/base_run/split_manifest.json
../../../working/bkcs-unlearning/base_run/base_model/best_model.pt
../../../working/bkcs-unlearning/base_run/base_model/result.json
~~~

Base checkpoint lưu ít nhất:

- Model state và model config.
- Best unknown threshold.
- Train, validation và test instance IDs.
- Cấu hình balanced sampling, pool count và số mẫu mỗi lớp mỗi epoch.
- Dataset signatures.
- Split seed và validation ratio.
- Pretrained encoder provenance.
- Base test metrics.

### 8.3 Phase 3: `make-manifests`

Mục tiêu:

- Load base checkpoint và đúng split đã dùng khi train.
- Score các AOL instance trong base-train bằng fixed threshold.
- Chỉ giữ các AOL instance đang được dự đoán đúng là `KNOWN`.
- Tạo thứ tự chọn deterministic và stratified theo original label.
- Sinh các forget set lồng nhau.

Input chính:

| Argument | Mặc định |
|---|---|
| `--base-checkpoint` | Bắt buộc |
| `--counts` | `1,4,10,50,100` |
| `--selection-seed` | 42 |
| `--manifest-output-dir` | `.../forget_manifests` |

Quan hệ lồng nhau phải giữ:

~~~
forget_1 subset forget_4 subset forget_10 subset forget_50 subset forget_100
~~~

Output:

~~~
../../../working/bkcs-unlearning/forget_manifests/forget_1.json
../../../working/bkcs-unlearning/forget_manifests/forget_4.json
../../../working/bkcs-unlearning/forget_manifests/forget_10.json
../../../working/bkcs-unlearning/forget_manifests/forget_50.json
../../../working/bkcs-unlearning/forget_manifests/forget_100.json
~~~

Mỗi manifest chứa:

- Selection strategy và selection seed.
- Requested count.
- Target binary label `UNKNOWN`.
- Base checkpoint path và SHA256.
- Dataset signatures.
- Danh sách instance ID, row hash, source, original label và original label ID.
- Xác suất unknown trước unlearning.

### 8.4 Phase 4: `unlearn`

Mục tiêu:

- Xác minh manifest thuộc đúng base checkpoint và dataset.
- Tạo `Df`, `Dr` và `Du_ref`.
- Tạo các validation/test slice cố định.
- Chạy tuần tự từng mode trong `--modes`.
- Chạy nhiều repeat cho mỗi non-baseline mode.
- Ghi result từng repeat và bảng benchmark tổng hợp.

Input chính:

| Argument | Mặc định |
|---|---:|
| `--base-checkpoint` | Bắt buộc |
| `--forget-manifest` | Bắt buộc |
| `--modes` | Cả mười hai mode |
| `--repeats` | 3 |
| `--optimization-seed` | 42 |
| `--mode-config-json` | JSON chứa preset tối ưu riêng cho từng mode |
| `--save-models` | False nếu không thêm flag |

`--mode-config-json` thay thế các hyperparameter unlearning dùng chung trước đây. Mỗi mode khai báo riêng số epoch, learning rate, weight decay, số step tối đa, trọng số forget/retain, early-stop target và các tham số đặc thù của objective. Cấu hình thực tế được chép vào `summary.json` và từng `result.json`.

## 9. Kiến trúc benchmark mode

### 9.1 Ánh xạ cố định

| Mode | Objective | Trainable scope | Vai trò |
|---|---|---|---|
| `baseline` | Không update | None | Mốc so sánh từ base checkpoint |
| `only_head` | Targeted rehearsal | Final classifier head | Baseline phạm vi nhỏ nhất |
| `full_mlp` | Targeted rehearsal | Toàn bộ MLP | Baseline chỉ cập nhật classifier |
| `last_encoder_block` | Targeted rehearsal | Encoder block cuối + MLP | Baseline cập nhật representation cục bộ |
| `full_model` | Targeted rehearsal | Toàn encoder + MLP | Baseline cập nhật toàn model |
| `l2ul_neggrad` | Paper misclassification | Full model | Negative CE trên nhãn gốc của `Df` |
| `l2ul_adv` | Paper ADV | Full model | Misclassification loss + targeted L2-PGD support set |
| `l2ul_adv_imp` | Paper ADV + IMP | Full model | ADV + inverted MAS parameter regularization |
| `amnesiac` | Adapted Amnesiac | Full MLP | Gradient ascent trên nhãn gốc của `Df` |
| `blindspot` | Adapted Blindspot | Full model | Soft unknown target cho `Df`, distill `Dr` |
| `bnd` | Adapted Boundary Unlearning | Full MLP | Đẩy logit unknown vượt known theo margin |
| `mkmmd` | Adapted MKMMD | Full model | Relabel và căn chỉnh embedding distribution |

Các mode `only_head`, `full_mlp`, `last_encoder_block` và `full_model` dùng cùng objective nhưng có preset optimizer riêng phù hợp với độ rộng scope. Vì vậy benchmark active so sánh các mode đã được cấu hình thực dụng, không phải strict ablation chỉ thay đổi scope. Strict scope ablation phải chạy thêm một protocol dùng chung hyperparameter. Chúng không phải option scope gắn thêm vào các paper method.

Các paper method có scope cố định trong `MODE_SPECS`. Muốn nghiên cứu một paper method với scope khác phải tạo mode mới có tên riêng, không thay đổi ngầm ý nghĩa mode hiện tại.

### 9.2 Phạm vi tham số

#### `only_head`

Chỉ train `model.classifier[-1]`, tức Linear cuối từ 32 chiều sang 2 logits. Encoder và các hidden layer của MLP bị freeze và ở eval mode.

#### `full_mlp`

Train toàn bộ `model.classifier`. Encoder bị freeze và ở eval mode.

#### `last_encoder_block`

Train:

~~~
encoder.conv4
encoder.conv4_4
encoder.batch_norm4
encoder.fc
classifier
~~~

Các encoder block trước bị freeze. Conv, dropout, FC của block cuối và classifier được đặt train mode; `BatchNorm1d` vẫn ở eval mode để khóa running statistics.

#### `full_model`

Train toàn bộ encoder và classifier.

Ở mọi scope unlearning có encoder trainable, affine weights của BatchNorm vẫn có thể nhận gradient nhưng running mean/variance luôn bị khóa ở base checkpoint. Với `Df` chỉ có một hoặc vài instance, cập nhật running statistics từ `Df` sẽ làm lệch toàn bộ retained distribution dù chưa xét gradient.

## 10. Objective của từng mode

### 10.1 `baseline`

Không tạo optimizer, không backward và không cập nhật trọng số.

~~~
theta = theta0
~~~

Baseline chỉ chạy một lần dù `--repeats` lớn hơn 1.

### 10.2 Targeted rehearsal

Áp dụng cho:

~~~
only_head
full_mlp
last_encoder_block
full_model
~~~

Forget loss:

~~~
L_forget = CE(l_theta(x_f), UNKNOWN)
~~~

Retain loss:

~~~
L_retain = CE(l_theta(x_r), y_r)
~~~

Total loss:

~~~
L_total = lambda_forget * L_forget + lambda_retain * L_retain
~~~

Ý nghĩa:

- `L_forget` đẩy từng instance trong `Df` sang nhãn unknown.
- `L_retain` giữ khả năng phân loại trên `Dr`.
- Bốn mode khác nhóm tham số nhận gradient và preset optimizer; objective vẫn giống nhau.

### 10.3 Instance-wise L2UL theo paper

Tham khảo: [Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers](https://doi.org/10.1609/aaai.v38i10.28996).

Ba mode này dùng full model, optimizer SGD, momentum `0.9` và bắt đầu độc lập từ cùng base checkpoint. Preset được đặt riêng theo hành vi thực tế của từng mode:

- `l2ul_neggrad`: learning rate `1e-2`, weight decay `1e-4`, tối đa 1000 update một-mẫu. Budget dài và learning rate này bám theo official implementation, vì negative CE có gradient rất nhỏ khi base model đã gần như chắc chắn mẫu là KNOWN.
- `l2ul_adv` và `l2ul_adv_imp`: learning rate `1e-3`, weight decay `1e-4`, tối đa 100 epoch và dừng sớm. Learning rate thận trọng hơn được dùng cho traffic để hạn chế phá retain utility.

#### `l2ul_neggrad`

Misclassification loss của paper:

~~~
L_MS = -CE(l_theta(x_f), y_f)
~~~

Do optimizer minimize loss âm, xác suất của nhãn gốc bị giảm. Để kết quả phù hợp với bài toán hiện tại, mode chỉ dừng khi accuracy theo `argmax` trên `Df` bằng 0 và mọi forget instance đồng thời được dự đoán UNKNOWN theo calibrated threshold, hoặc khi hết budget.

#### `l2ul_adv`

Trước khi cập nhật model, code đóng băng `f_theta0` và sinh targeted L2-PGD support set. Paper gốc làm việc trên ảnh; bản traffic chỉ cho phép thay đổi timestamp và packet size, giữ nguyên direction, packet mask và padding. L2 được tính trong không gian chuẩn hóa theo flow time span và maximum packet size.

Preset mặc định sinh 20 biến thể cho mỗi forget instance, chạy 100 PGD step với `epsilon=2.0`, step size `0.1`, random start. Trong bài toán nhị phân chỉ có một non-original target là `UNKNOWN`; không thể random giữa nhiều wrong class như bài toán đa lớp của paper. Chỉ các biến thể thật sự đạt target UNKNOWN theo `argmax` mới được đưa vào support set. Pipeline dừng với lỗi rõ ràng nếu một forget instance không tạo được support sample hợp lệ.

~~~
L_ADV = n_f/(n_f+n_adv) * L_MS
      + n_adv/(n_f+n_adv) * CE(l_theta(x_adv), UNKNOWN)
~~~

Adversarial set luôn được sinh từ base model đóng băng trước unlearning và được cache theo checkpoint hash, forget rows, feature transform và PGD config.

#### `l2ul_adv_imp`

Code ước lượng MAS importance trên `Df` bằng absolute gradient của squared output L2 norm theo Eq. (2). Importance được min-max normalize theo parameter tensor rồi đảo:

~~~
Omega_inv = 1 - normalize(Omega_MAS)
L_IMP = 1/2 * sum_i Omega_inv_i * (theta_i - theta0_i)^2
L_total = L_ADV + 10 * L_IMP
~~~

ADV dùng hệ số đơn vị trước khi cân theo số mẫu trong hai batch. IMP dùng `lambda_importance=10`, khớp preset mặc định trong official implementation.

Do đó các tham số ít quan trọng với `Df` bị neo gần base checkpoint mạnh hơn, còn tham số liên quan tới `Df` được phép thay đổi nhiều hơn. ADV và IMP chỉ thuộc hai mode L2UL tương ứng; chúng không được thêm vào Blindspot, BND, MKMMD hay các scope baseline.

### 10.4 `amnesiac`

Tham khảo: [Amnesiac Machine Learning](https://doi.org/10.1609/aaai.v35i13.17371).

Forget loss:

~~~
L_forget = -CE(l_theta(x_f), y_f)
~~~

Total loss:

~~~
L_total = lambda_forget * L_forget + lambda_retain * CE(l_theta(x_r), y_r)
~~~

Vì optimizer minimize `L_total`, dấu âm làm gradient ascent trên CE của nhãn gốc. Model bị đẩy ra xa quyết định `KNOWN` cho `Df` trong khi retain rehearsal hạn chế suy giảm trên `Dr`.

Đây là bản thích nghi hậu huấn luyện. Nó không phải exact update subtraction của paper vì base training hiện không lưu từng batch update.

### 10.5 `blindspot`

Tham khảo: [Fast Yet Effective Machine Unlearning](https://doi.org/10.1609/aaai.v37i6.25879).

Tạo một teacher là bản copy bị freeze của base model.

Với `Df`, dùng soft target cố định:

~~~
q_forget = [1 - c, c]
c = blindspot_unknown_confidence = 0.90
~~~

Forget loss:

~~~
L_forget = T^2 * KL(q_forget || p_theta,T(x_f))
~~~

Với `Dr`, teacher giữ hành vi base:

~~~
L_retain = T^2 * KL(p_theta0,T(x_r) || p_theta,T(x_r))
~~~

Total loss:

~~~
L_total = lambda_forget * L_forget + lambda_retain * L_retain
T = 2.0 mặc định
~~~

Teacher chỉ cung cấp target và không nhận gradient.

### 10.6 `bnd`

Tham khảo: [Boundary Unlearning](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Boundary_Unlearning_Rapid_Forgetting_of_Deep_Networks_via_Shifting_the_CVPR_2023_paper.html).

Đặt:

~~~
delta = logit_UNKNOWN - logit_KNOWN
~~~

Forget loss:

~~~
L_forget = mean(ReLU(margin - delta))
margin = 0.5 trong preset Kaggle Free
~~~

Total loss:

~~~
L_total = lambda_forget * L_forget + lambda_retain * CE(l_theta(x_r), y_r)
~~~

Loss bằng 0 khi unknown logit đã cao hơn known logit ít nhất một khoảng `margin`.

### 10.7 `mkmmd`

Nền tảng metric: [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html).

Mode này dùng:

- `Df` cho forget embedding.
- `Du_ref` cho unknown reference embedding.
- `Dr` cho retain embedding.
- Frozen base teacher cho cả unknown-reference và retain-reference embedding.

Classification terms:

~~~
L_forget_cls = CE(l_theta(x_f), UNKNOWN)
L_retain_cls = CE(l_theta(x_r), y_r)
~~~

Distribution terms:

~~~
L_mkmmd_forget = MKMMD(z_theta(Df), z_theta0(Du_ref))
L_mkmmd_retain = MKMMD(z_theta(Dr), z_theta0(Dr))
~~~

Total loss:

~~~
L_total =
    lambda_forget * L_forget_cls
    + lambda_retain * L_retain_cls
    + lambda_mkmmd_forget * L_mkmmd_forget
    + lambda_mkmmd_retain * L_mkmmd_retain
~~~

MKMMD implementation dùng biased estimate:

~~~
MMD(X,Y) = mean(Kxx) + mean(Kyy) - 2 * mean(Kxy)
~~~

Base bandwidth là median của các positive pairwise squared distances trên batch ghép. Kernel là trung bình của các RBF kernel với multipliers:

~~~
0.5, 1.0, 2.0, 4.0
~~~

## 11. Loss cập nhật trọng số như thế nào

Đây là phần phân biệt giữa **tối ưu** và **đánh giá**.

Với targeted rehearsal, Amnesiac, Blindspot, BND và MKMMD, mỗi train step:

1. Lấy một batch từ `Df` và một batch từ `Dr`.
2. Với `mkmmd`, lấy thêm một batch từ `Du_ref`.
3. Forward branch model để tính các thành phần loss.
4. Cộng các thành phần thành `L_total` theo mode.
5. Gọi `optimizer.zero_grad(set_to_none=True)`.
6. Gọi `L_total.backward()` để tính gradient.
7. Clip gradient norm ở mức 5.0.
8. Gọi `AdamW.step()`.
9. Chỉ các parameter có `requires_grad=True` trong scope của mode được cập nhật.

Ba mode `l2ul_*` dùng runner khác:

1. `l2ul_neggrad` lấy batch `Df`; hai mode ADV lấy thêm batch từ adversarial cache.
2. Tính `L_MS`, tùy mode cộng `L_ADV` và `L_IMP`.
3. Backward và gọi `SGD.step()` trên toàn bộ model; runner paper không dùng retain rehearsal và không clip gradient.
4. Sau mỗi update, tính cả accuracy `argmax` và `forget_success_rate` theo calibrated threshold trên `Df`.
5. Dừng khi cả hai mục tiêu đạt 100%, hoặc khi hết budget của mode.

Điều kiện `argmax` giữ metric gốc của paper; điều kiện threshold bảo đảm kết quả cũng thỏa chính sách `forget instance -> UNKNOWN` của bài toán nhị phân hiện tại.

Do đó loss không chỉ được tính để báo cáo. `L_total` trực tiếp quyết định gradient và thay đổi trọng số.

Sau mỗi epoch và sau khi train xong, code tính metric bằng `model.eval()` và `torch.no_grad()`. Các phép đánh giá này:

- Không gọi `backward()`.
- Không gọi `optimizer.step()`.
- Không cập nhật trọng số.
- Chỉ đo kết quả của model hiện tại.

## 12. Protocol chạy unlearning

### 12.1 Khởi tạo công bằng

Mỗi repeat của mỗi mode:

- Bắt đầu từ một deep copy mới của cùng `f_theta0`.
- Không kế thừa trọng số từ mode chạy trước.
- Dùng cùng `Df`, `Dr`, `Du_ref`, validation/test slices và threshold.
- Dùng optimization seed bằng `optimization_seed + repeat`.
- Tạo mới retain loader với chính seed của repeat; sampler state không được kế thừa từ mode chạy trước.
- Retain batches giữ tỷ lệ `KNOWN:UNKNOWN = 1:1`, giống protocol base training. Điều này tránh việc phân bố `Dr` gốc nghiêng về UNKNOWN cộng hưởng với forget target UNKNOWN.

Thứ tự mode không làm thay đổi checkpoint khởi đầu.

### 12.2 Số step mỗi epoch

Với từng mode:

~~~
steps = max(len(forget_loader), min(len(retain_loader), 10), 1)
~~~

Nếu `mode_config.max_steps_per_epoch > 0`:

~~~
steps = min(default_steps, mode_config.max_steps_per_epoch)
~~~

Forget, retain và unknown-reference loaders được cycle để cung cấp đủ batch.

### 12.3 Preset tối ưu theo mode

Các preset dưới đây là ngân sách thực dụng cho GPU Kaggle Free. Scope càng rộng thì learning rate càng thấp và retain weight càng cao. `amnesiac` có ngân sách dài hơn vì negative CE tạo gradient rất nhỏ khi base model đang cực kỳ tự tin vào nhãn gốc.

| Mode | Epoch tối đa | LR | `lambda_forget` | `lambda_retain` | Tham số riêng |
|---|---:|---:|---:|---:|---|
| `baseline` | 0 | 0 | 0 | 0 | Không update |
| `only_head` | 8 | 3e-4 | 2.0 | 1.0 | `min_epochs=3` |
| `full_mlp` | 4 | 7.5e-5 | 1.0 | 2.0 | `min_epochs=2` |
| `last_encoder_block` | 3 | 5e-5 | 1.0 | 2.0 | `min_epochs=2` |
| `full_model` | 3 | 2.5e-5 | 1.0 | 2.5 | `min_epochs=2` |
| `amnesiac` | 12 | 3e-4 | 5.0 | 0.5 | `min_epochs=3` |
| `blindspot` | 4 | 7.5e-5 | 1.0 | 2.0 | `T=2`, `confidence=0.90` |
| `bnd` | 5 | 7.5e-5 | 1.0 | 2.0 | `margin=0.5` |
| `mkmmd` | 4 | 2.5e-5 | 1.0 | 2.5 | `lambda_mmd_f=0.5`, `lambda_mmd_r=2.0` |

Tất cả non-baseline mode dùng `weight_decay=1e-4`, tối đa 10 step mỗi epoch, AdamW và gradient clipping 5.0. Sau `min_epochs`, branch dừng sớm ngay khi `forget_success >= 1.0`. Early stopping dùng `Df` và chỉ giới hạn số update; test set không tham gia quyết định dừng.

Model tại epoch dừng hoặc epoch cuối là model được đánh giá cuối cùng. Đây là preset benchmark, không phải kết quả hyperparameter search tối ưu; so sánh cuối cùng vẫn phải đọc đồng thời forget success và retain/test damage.

### 12.4 Giải phóng tài nguyên

Sau mỗi repeat:

- Ghi result ra disk.
- Xóa branch model.
- Gọi `torch.cuda.empty_cache()` nếu đang dùng CUDA.

## 13. Evaluation protocol

### 13.1 Quy tắc dự đoán

~~~
prediction = UNKNOWN nếu probability_unknown >= tau
prediction = KNOWN nếu probability_unknown < tau
~~~

`tau` là threshold của base model và không được tune lại theo từng mode.

### 13.2 Metrics chung

`evaluate` trả về:

| Metric | Định nghĩa |
|---|---|
| `loss` | Mean cross entropy trên slice |
| `accuracy` | Tỷ lệ dự đoán đúng |
| `known_recall` | Recall của class `KNOWN` |
| `unknown_recall` | Recall của class `UNKNOWN` |
| `balanced_accuracy` | Trung bình recall của các class có mặt |
| `unknown_threshold` | Threshold đang dùng |
| `total` | Số instance |
| `confusion_matrix` | Ma trận 2x2 |

Quy ước confusion matrix:

~~~
rows    = true [KNOWN, UNKNOWN]
columns = predicted [KNOWN, UNKNOWN]
~~~

Vì vậy:

~~~
FP = confusion[KNOWN][UNKNOWN]
FN = confusion[UNKNOWN][KNOWN]
~~~

### 13.3 Forget metrics

Trên đúng `Df`:

~~~
forget_success_rate = count(prediction == UNKNOWN) / |Df|
mean_probability_unknown = mean(p_UNKNOWN(x_f))
count = |Df|
~~~

Forget success cao hơn nghĩa là nhiều instance cần quên đã chuyển sang unknown. Metric này phải được đọc cùng retain/test metrics để tránh chọn một model quên tốt nhưng phá hỏng toàn bộ classifier.

Riêng metric theo paper:

~~~
paper_misclassification_rate = count(argmax(l_theta(x_f)) != y_f) / |Df|
~~~

Trong bài toán nhị phân, `argmax != KNOWN` đồng nghĩa `argmax == UNKNOWN`. Metric này dùng ngưỡng quyết định implicit `0.5`, còn `forget_success_rate` dùng fixed threshold `tau` của base model (ví dụ `0.65`), nên hai giá trị có thể khác nhau.

### 13.4 Retain slices

Validation và test đều được chia thành:

| Slice | Nội dung |
|---|---|
| `same_original_label` | AOL records có original label xuất hiện trong `Df` |
| `other_known` | AOL records có original label không xuất hiện trong `Df` |
| `unknown_100` | Tất cả 100 Top Search records |
| `full_retain` | Toàn bộ records của split |

`same_original_label` đo collateral damage gần nhất với các instance bị quên. `other_known` đo tác động lên known classes khác. `unknown_100` đo khả năng nhận diện unknown. `full_retain` dùng cho so sánh tổng thể.

### 13.5 Benchmark error

Từ `after_test["full_retain"]`:

~~~
total_error = FP + FN
~~~

So với baseline:

~~~
total_error_reduction_percent =
    100 * (baseline_total_error - total_error) / baseline_total_error
~~~

Giá trị dương là giảm lỗi so với baseline. Nếu baseline total error bằng 0, reduction được ghi `NaN` để tránh chia cho 0.

`benchmark_runs.csv` lưu từng repeat. `benchmark_summary.csv` lưu mean và population standard deviation với `ddof=0` cho:

~~~
fn
fp
total_error
total_error_reduction_percent
~~~

## 14. Output artifacts

~~~
../../../working/bkcs-unlearning/
├── schema_validation.json
├── csv_index.json
├── forget_manifests/
│   ├── forget_1.json
│   ├── forget_4.json
│   ├── forget_10.json
│   ├── forget_50.json
│   └── forget_100.json
└── base_run/
    ├── split_manifest.json
    ├── base_model/
    │   ├── best_model.pt
    │   └── result.json
    └── instance_unlearning/
        └── forget_<count>/
            ├── summary.json
            ├── benchmark_runs.csv
            ├── benchmark_summary.csv
            ├── baseline/repeat_0/result.json
            ├── only_head/repeat_<r>/result.json
            ├── full_mlp/repeat_<r>/result.json
            ├── last_encoder_block/repeat_<r>/result.json
            ├── full_model/repeat_<r>/result.json
            ├── paper_adversarial_cache/paper_l2pgd_<hash>.pt
            ├── l2ul_neggrad/repeat_<r>/result.json
            ├── l2ul_adv/repeat_<r>/result.json
            ├── l2ul_adv_imp/repeat_<r>/result.json
            ├── amnesiac/repeat_<r>/result.json
            ├── blindspot/repeat_<r>/result.json
            ├── bnd/repeat_<r>/result.json
            └── mkmmd/repeat_<r>/result.json
~~~

Nếu thêm `--save-models`, mỗi non-baseline repeat còn có:

~~~
unlearned_model.pt
~~~

Mặc định notebook không thêm flag này để tránh nhân kích thước output theo số mode và repeat.

### 14.1 `result.json` của từng repeat

Các nhóm field chính:

- Mode, mode specification, repeat và optimization seed.
- Runtime.
- Base checkpoint hash.
- Forget manifest path và hash.
- Dataset signatures.
- Forget instance IDs và original labels.
- Optional saved model path.
- Before/after forget metrics theo cả fixed threshold của task và `argmax` criterion của paper.
- Before/after validation và test metrics.
- Với ADV: adversarial cache path và generation success metrics.
- Per-epoch history gồm loss components, trainable parameter names và validation metrics.

### 14.2 `summary.json`

Summary cấp manifest chứa:

- Provenance chung.
- Danh sách mode và `MODE_SPECS`.
- Unlearning hyperparameters.
- Before metrics.
- Toàn bộ experiment results.
- Raw benchmark rows.
- Aggregate benchmark summary.
- Tổng runtime.

### 14.3 Tạo báo cáo XLSX từ output

Script `generate_report.py` đọc trực tiếp schema `summary.json` do phase
`unlearn` tạo ra. Script không cần chạy lại model và không đọc `feature_cache` hay
checkpoint. Mỗi workbook có:

- Sheet `Summary` so sánh mean theo repeat giữa tất cả mode.
- Một sheet riêng cho từng mode.
- Run context, hyperparameter thực tế, epoch thực chạy và runtime.
- Forget success, xác suất UNKNOWN trên `Df`, Accuracy, Precision, Recall, F1,
  Balanced Accuracy và các retain recall.
- TN, FP, FN, TP, tổng lỗi, delta và phần trăm thay đổi so với baseline.
- Công thức và conditional formatting tương thích khi upload lên Google Sheets.

Input có thể là `summary.json`, một folder `forget_<count>`, hoặc một folder cha.
Ví dụ tạo một workbook cho `forget_1`:

~~~powershell
python generate_report.py `
  outputs/bkcs-unlearning/base_run/instance_unlearning/forget_1 `
  --output outputs/unlearning_benchmark_forget_1.xlsx `
  --csv-output outputs/unlearning_report.csv `
  --verify
~~~

Ví dụ quét tất cả `forget_<count>` bên dưới `instance_unlearning`:

~~~powershell
python generate_report.py `
  outputs/bkcs-unlearning/base_run/instance_unlearning `
  --output outputs/xlsx_reports `
  --verify
~~~

Nếu tìm thấy nhiều `summary.json`, script tạo một workbook riêng cho mỗi forget
count để các sheet method không bị trộn giữa các protocol. Tên mặc định là
`unlearning_benchmark_forget_<count>.xlsx`. Dependency duy nhất ngoài Python
standard library là `openpyxl`; nếu môi trường chưa có thì cài bằng
`pip install openpyxl`.

`--csv-output` là tùy chọn và chỉ dùng khi input chọn đúng một `summary.json`.
CSV là bảng phẳng để xử lý bằng code; XLSX vẫn là báo cáo chính vì giữ được
`Summary` và các sheet riêng cho từng method.

## 15. Cấu hình notebook hiện tại

Cell cấu hình đặt:

~~~python
RUN_PIPELINE = True
~~~

Cell cuối chạy tuần tự:

~~~python
NOTEBOOK_COMMANDS = [
    "validate-schema",
    "train-base",
    "make-manifests",
    "unlearn",
]
~~~

Cell `NOTEBOOK_ARGUMENTS_BY_COMMAND` hiện cấu hình:

- Base training 16 epochs trên `cuda:0`, dùng balanced binary batches 1:1.
- Sinh các manifest size `1,4,10,50,100`.
- `FORGET_COUNT = 1`, nên phase unlearn dùng `forget_1.json`.
- Chạy cả mười hai mode.
- Chạy 3 repeats cho mỗi non-baseline mode.
- Mỗi mode dùng preset epoch, learning rate và loss weight riêng trong `DEFAULT_MODE_HYPERPARAMETERS`.
- Không lưu branch checkpoint do chưa có `--save-models`.

Lưu ý: danh sách mode được loop bên trong phase `unlearn`. Danh sách phase được loop ở cell cuối. Đây là hai vòng lặp khác nhau.

## 16. Kế hoạch chạy

### 16.1 Preflight

1. Attach đủ ba Kaggle datasets.
2. Bật GPU accelerator.
3. Run các cell định nghĩa.
4. Kiểm tra log resolved paths.
5. Kiểm tra CUDA available và free disk.
6. Chạy `validate-schema` trước khi train dài.

### 16.2 Smoke run

Trước full benchmark, cấu hình tạm:

~~~
modes = baseline,only_head
repeats = 1
mode_config.baseline.epochs = 0
mode_config.only_head.epochs = 1
mode_config.only_head.min_epochs = 1
mode_config.only_head.max_steps_per_epoch = 1
forget_manifest = forget_1.json
~~~

Smoke run đạt khi:

- Manifest validation qua.
- Hai mode bắt đầu từ cùng checkpoint.
- Loss hữu hạn.
- `result.json` và hai benchmark CSV được tạo.
- Không có CUDA out-of-memory.

### 16.3 Full run mặc định

Sau smoke run:

~~~
modes = baseline,only_head,full_mlp,last_encoder_block,full_model,l2ul_neggrad,l2ul_adv,l2ul_adv_imp,amnesiac,blindspot,bnd,mkmmd
repeats = 3
mode_config = DEFAULT_MODE_HYPERPARAMETERS
forget_manifest = forget_1.json
~~~

### 16.4 Benchmark nhiều forget count

`make-manifests` tạo năm manifest nhưng cấu hình notebook hiện chỉ chạy manifest được chọn bởi `FORGET_COUNT`, mặc định là `forget_1.json`.

Để đánh giá theo kích thước forget set, chạy phase `unlearn` riêng cho từng:

~~~
forget_1.json
forget_4.json
forget_10.json
forget_50.json
forget_100.json
~~~

Mỗi manifest có output root riêng theo stem nên không ghi đè kết quả của count khác.

## 17. Resume và tránh chạy lại không cần thiết

Vì `RUN_PIPELINE=True` và `NOTEBOOK_COMMANDS` hiện chứa cả bốn phase, chạy lại cell cuối sẽ train base lại.

Khi base checkpoint và manifests đã hợp lệ:

- Bỏ `train-base` khỏi `NOTEBOOK_COMMANDS` nếu chỉ muốn chạy benchmark lại.
- Bỏ `make-manifests` nếu không đổi checkpoint, signatures hoặc selection seed.
- Giữ `unlearn` và đổi `--forget-manifest` hoặc `--modes` theo thí nghiệm.
- Không dùng một manifest cũ với checkpoint mới, vì hash validation sẽ dừng pipeline.

## 18. Quản lý kích thước Kaggle output

Output lớn có thể làm Save Version chậm hoặc thất bại khi vượt quota của Kaggle.

Thiết kế hiện tại giảm output bằng cách:

- Chỉ lưu một base checkpoint.
- Không lưu checkpoint của từng branch theo mặc định.
- Luôn lưu JSON/CSV metrics nhỏ hơn model checkpoint.
- Giải phóng branch khỏi GPU sau mỗi repeat.

Nguồn có thể chiếm disk lớn nhất là `feature_cache` vì mỗi decoded instance có một file tensor riêng. Vì vậy cache mặc định nằm trong `/kaggle/temp`, không nằm dưới `/kaggle/working` và không được đóng gói vào output của Save Version.

Quy trình vận hành:

1. Giữ feature cache trong `/kaggle/temp` lúc train và benchmark để tránh decode CSV lặp lại.
2. Kiểm tra free disk trước full run.
3. Chỉ thêm `--save-models` khi thực sự cần tải từng branch.
4. Không xóa cache giữa các mode vì sẽ làm tăng đáng kể thời gian chạy.

Kaggle loại bỏ vùng temp sau session; không cần cleanup cache trước Save Version.

## 19. Reproducibility và kiểm tra bắt buộc

### 19.1 Reproducibility

- Seed Python, NumPy và PyTorch.
- Seed toàn bộ CUDA devices nếu có.
- Bật deterministic CuDNN và tắt benchmark mode.
- Split seed mặc định 42.
- Manifest selection seed mặc định 42.
- Optimization seed của repeat là `42 + repeat`.
- Cùng threshold cho toàn bộ mode.

### 19.2 Data integrity

Pipeline phải dừng nếu:

- Dataset folder hoặc file bắt buộc bị thiếu/rỗng.
- Mapping header sai hoặc label ID trùng.
- CSV row sai width.
- Packet label không nhất quán.
- Padding không phải all-zero.
- Row hash thay đổi.
- Dataset signatures khác index/checkpoint/manifest.

### 19.3 Experiment integrity

Pipeline phải dừng nếu:

- `Df` chứa record không thuộc base-train.
- `Df` chứa record không phải AOL/KNOWN.
- Forget manifest được tạo từ checkpoint khác.
- `Df union Dr` không bằng base-train.
- Không có unknown reference trong `Dr`.
- Mode không nằm trong `MODE_SPECS`.
- Repeat hoặc epochs nhỏ hơn 1.
- Blindspot temperature không dương.
- Blindspot unknown confidence không nằm trong `(0.5, 1.0)`.
- Boundary margin không dương.
- MKMMD bandwidth multiplier không dương.

## 20. Cách đọc kết quả

Không chọn mode chỉ bằng `forget_success_rate`.

Một mode tốt cần đồng thời:

1. Forget success cao trên `Df`.
2. Mean unknown probability trên `Df` tăng rõ so với before.
3. `same_original_label` không suy giảm quá lớn.
4. `other_known` giữ known recall.
5. `unknown_100` giữ hoặc cải thiện unknown recall.
6. `full_retain` có total error thấp.
7. Kết quả ổn định qua repeats, thể hiện bằng standard deviation nhỏ.
8. Runtime và số tham số cập nhật phù hợp với chi phí chấp nhận được.

So sánh chính:

| Câu hỏi | Nhóm mode cần so |
|---|---|
| Update scope ảnh hưởng ra sao? | `only_head`, `full_mlp`, `last_encoder_block`, `full_model` |
| Objective paper-adapted có tốt hơn relabel không? | `amnesiac`, `blindspot`, `bnd`, `mkmmd` so với bốn targeted modes |
| ADV/IMP của instance-wise paper có giảm collateral damage không? | `l2ul_neggrad`, `l2ul_adv`, `l2ul_adv_imp` |
| Unlearning có phá base behavior không? | Mọi mode so với `baseline` |
| Kết quả có ổn định không? | Mean/std qua ba repeats |

## 21. Hạn chế hiện tại

- Amnesiac là bản post-hoc gradient-ascent adaptation, không phải exact reversal của update history.
- Blindspot được biến đổi thành binary known-to-unknown setting với soft unknown target.
- BND được biến đổi thành logit margin cho đúng hai class.
- MKMMD căn chỉnh embedding theo unknown data trong `Dr`, không dùng external unknown dataset thứ ba.
- Paper gốc là image multiclass; bản traffic nhị phân chỉ có một wrong target và dùng miền perturbation timestamp/size có ràng buộc thay cho pixel space.
- `epsilon=2.0` là bán kính trong không gian traffic đã chuẩn hóa, không đồng nhất về đơn vị với epsilon pixel của paper; cần ablation riêng.
- Chưa có retrain-on-`Dr` oracle để đo khoảng cách tới exact retraining.
- Chưa có membership inference hoặc privacy audit.
- Các mode L2UL dừng khi vừa đạt paper argmax criterion vừa đạt task threshold criterion; các mode còn lại dừng theo threshold forget success sau `min_epochs`. Chưa có best-epoch selection theo retain utility.
- Cấu hình cell cuối chỉ benchmark một manifest tại một thời điểm.
- Feature cache có thể lớn nhưng nằm trong `/kaggle/temp`, nên vẫn cần theo dõi disk trong session dù không làm phình output Save Version.

## 22. Acceptance criteria

Pipeline được xem là hoàn thành cho một benchmark run khi:

- `validate-schema` ghi được schema report hợp lệ.
- Base model, split manifest và base result được ghi đầy đủ.
- Forget manifest có đúng count, đúng base checkpoint hash và đúng dataset signatures.
- Mọi mode chạy độc lập từ cùng base checkpoint.
- Không có phép ghép ngầm giữa scope modes và paper methods.
- Loss được dùng để backward/update đúng trainable scope.
- Evaluation không cập nhật trọng số.
- Before/after metrics có đủ forget, validation và test slices.
- `benchmark_runs.csv` có một dòng cho mỗi mode/repeat.
- `benchmark_summary.csv` có mean/std cho mọi mode.
- Baseline chỉ có một repeat.
- Output model branch chỉ xuất hiện khi bật `--save-models`.
- Toàn bộ artifact cần thiết nằm dưới `/kaggle/working/bkcs-unlearning`.

## 23. Trạng thái implementation hiện tại

Đã có trong notebook:

- Recursive Kaggle input discovery.
- Schema validation.
- Byte-offset CSV index và row hashing.
- Tensor feature cache.
- Deterministic split.
- Checkpoint-compatible encoder và binary MLP.
- Base training, model selection và threshold selection.
- Deterministic balanced binary batch sampler 1:1 cho base training.
- Nested forget manifests.
- Mười hai independent benchmark modes.
- Paper L2UL NegGrad, ADV và ADV+IMP với targeted constrained L2-PGD, SGD và stopping rule riêng.
- Adapted Amnesiac, Blindspot, BND và MKMMD objectives.
- Per-epoch forget/retain evaluation.
- Per-repeat result JSON.
- Raw và aggregate benchmark CSV.
- Optional branch checkpoint saving.
- Sequential phase runner trong cell cuối.

Đã kiểm tra cục bộ:

- Notebook JSON hợp lệ.
- Tất cả code cell compile được.
- Parser nhận cấu hình của cả bốn phase.
- Mapping mode, objective và scope đúng với thiết kế độc lập.
- Công thức benchmark total error và reduction hoạt động trên dữ liệu kiểm thử tổng hợp.

Chưa xác nhận cục bộ end-to-end vì môi trường local không có đầy đủ PyTorch/NumPy và không có Kaggle-mounted datasets. Full tensor run và benchmark cuối phải được xác nhận trên Kaggle GPU.
