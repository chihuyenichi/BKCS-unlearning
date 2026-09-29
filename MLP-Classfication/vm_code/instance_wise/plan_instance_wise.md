# Kế hoạch Instance-Wise Unlearning cho bài toán Known/Unknown

## 1. Mục tiêu

Xây dựng pipeline **instance-wise unlearning** cho bài toán phân loại nhị phân network traffic:

```text
AOL         -> known   (binary label 0)
icloud_100  -> unknown (binary label 1)
```

Model cơ sở:

```text
CSV row / network-flow instance [10000, 3]
-> pretrained RawPacketEncoder
-> embedding 256 chiều
-> MLP classifier
-> 2 logits: known / unknown
```

Khi có yêu cầu quên một tập instance `Df` thuộc AOL, model sau unlearning phải:

```text
instance trong Df               -> unknown
instance AOL còn lại            -> vẫn known
instance cùng original label    -> vẫn known nếu không nằm trong Df
instance từ icloud_100           -> vẫn unknown
```

Kế hoạch này chỉ xét bài toán nhị phân `known/unknown`. `original_label` của AOL chỉ được giữ làm metadata để chia dữ liệu và đo mức quên nhầm; nó không phải output class của MLP.

## 2. Phạm vi và định nghĩa

Ký hiệu:

```text
theta_0 : trọng số model binary trước unlearning
theta_u : trọng số model sau unlearning
Df      : đúng các instance được yêu cầu quên
Dr      : toàn bộ instance base-train còn lại
y_old   : known = 0 đối với Df của AOL
y_star  : unknown = 1, target mới của Df
```

Mục tiêu hành vi:

```text
model_theta_0(x_f) = known
model_theta_u(x_f) = unknown, với mọi x_f thuộc Df
```

Đây là **targeted relabeling instance-wise** theo hướng của paper `Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers`.

Một yêu cầu chỉ được gọi là instance-wise khi `Df` được chọn bằng danh sách instance ID. Không chọn `Df` bằng toàn bộ `original_label`.

Ví dụ đúng:

```text
Df = {AOL/train dòng 120, AOL/train dòng 981, AOL/train dòng 4102}
```

Ví dụ không dùng cho pipeline chính:

```text
Df = tất cả instance có original_label == B
```

Trường hợp thứ hai là group/class-wise unlearning.

## 3. Data contract đã xác nhận từ notebook VM

### 3.1. Đường dẫn runtime dự kiến trên VM

Repository và notebook được viết, review và kiểm thử cấu trúc trên máy cá nhân. Hai đường dẫn dưới đây **không phải đường dẫn dữ liệu trên máy phát triển hiện tại**; chúng chỉ là cấu hình runtime sau khi code được đẩy lên máy ảo:

```python
AOL_DATASET_DIR = Path('/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud')
UNKNOWN_273_DATASET_DIR = Path('/home/ubuntu/Documents/KF/data_processed/icloud_100')
```

Khi chạy experiment đầy đủ trên VM, vai trò nhị phân của từng nguồn là:

```text
/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud
    source key   = AOL
    binary label = known (0)

/home/ubuntu/Documents/KF/data_processed/icloud_100
    source key   = 273
    binary label = unknown (1)
```

Mỗi directory phải có đúng các input bắt buộc:

```text
<DATASET_DIR>/
├── train_data.csv
├── test_data.csv
└── label_mapping.csv
```

Quy trình môi trường:

```text
Máy cá nhân:
    viết và review code
    kiểm thử parser/schema bằng thư mục csv_sample trong repository
    không yêu cầu /home/ubuntu/... tồn tại

Máy ảo:
    pull/copy code từ repository
    trỏ config tới dataset đầy đủ trong /home/ubuntu/...
    chạy preflight, index, base training và unlearning
```

Khi thực thi trên VM, notebook đọc CSV trực tiếp từ filesystem của VM, không mount Google Drive và không parse lại PCAP trong pipeline này.

Hai path `/home/ubuntu/...` là giá trị mặc định cho **VM execution**, không phải điều kiện để import hoặc kiểm tra notebook trên máy cá nhân. Code khai báo chúng trong config cell hoặc nhận qua environment variable để có thể đổi máy ảo mà không sửa loader.

Chỉ khi bắt đầu một full run trên VM, trước khi index dữ liệu, preflight mới phải kiểm tra:

- hai directory tồn tại và resolve thành hai vị trí khác nhau;
- đủ ba file bắt buộc trong mỗi directory;
- các file là regular file, đọc được và không rỗng;
- `label_mapping.csv` có hai cột `label_name,label_id`;
- output/cache directory không nằm bên trong raw dataset directory;
- dataset signatures được lưu cùng base checkpoint và split manifest.

Không đưa absolute path vào phần logic của `instance_id`. ID dùng source key ổn định `AOL` hoặc `273`; resolved absolute path chỉ nằm trong dataset signature để kiểm tra provenance. Nhờ vậy, việc chuyển nguyên dataset sang mount point khác không làm thay đổi danh tính logic của instance, miễn nội dung và manifest vẫn khớp.

Không tự động dùng `csv_sample` thay cho dataset đầy đủ khi train. Thư mục mẫu chỉ dùng cho parser/unit test và hiện không đại diện đủ cả hai nguồn known/unknown.

Mỗi nguồn dữ liệu có:

```text
train_data.csv
test_data.csv
label_mapping.csv
```

Notebook `train_pipeline_vm.ipynb` đang xử lý dữ liệu như sau:

- Mỗi **dòng CSV được xem là một sample/instance**.
- Mỗi dòng có `40,000` số, tương ứng `10,000` packet x 4 trường:

```text
[label_id, relative_time, direction, packet_size] x 10000
```

- Dòng được reshape thành `[10000, 4]`.
- Cột `label_id` lặp lại theo packet được kiểm tra rồi loại bỏ.
- Input thực tế của encoder là:

```text
[10000, 3] = [relative_time, direction, packet_size]
```

- `packet_size == 0` được dùng để nhận biết padding và tạo mask.
- Notebook lập byte-offset index để seek đúng một dòng, không nạp toàn bộ file CSV vào RAM.

### 3.2. Contract xác nhận từ thư mục CSV mẫu

Contract chính thức cho phần implementation được lấy từ:

```text
MLP-Classfication/csv_sample-20260921T063615Z-1-001/csv_sample/icloud/
├── train_data.csv
├── test_data.csv
└── label_mapping.csv
```

Thống kê của bộ mẫu:

```text
train_data.csv:
    4,893 dòng dữ liệu
    khoảng 986 MB

test_data.csv:
    4,895 dòng dữ liệu
    khoảng 987 MB

label_mapping.csv:
    header = label_name,label_id
    50 label ID, từ 0 đến 49
```

`train_data.csv` và `test_data.csv` không có header. Mỗi newline kết thúc đúng một model instance. Một dòng điển hình bắt đầu như sau:

```text
0.0,0.0,0.0,103.0,
0.0,0.0000009999999974752427,0.0,103.0,
0.0,0.02249700017273426,0.0,103.0,
...
```

Các giá trị được nhóm theo thứ tự:

```text
packet_0 = [label_id, relative_time, direction, packet_size]
packet_1 = [label_id, relative_time, direction, packet_size]
...
packet_9999 = [label_id, relative_time, direction, packet_size]
```

Parser phải tuân thủ các invariant sau:

1. Đọc một dòng bằng byte offset và parse số thực, bao gồm scientific notation.
2. Dòng phải có chính xác `40,000` giá trị; nếu không thì fail.
3. Reshape thành `[10000, 4]` bằng `float32`.
4. `original_label_id` của row được lấy bằng `int(float(first_field))`.
5. Với packet thật, cột label lặp lại `original_label_id`.
6. Với packet padding, cả nhóm là `[0, 0, 0, 0]`.
7. Mask được tính bằng `packet_size != 0`, không tính từ cột label.
8. Sau khi kiểm tra label, loại cột label để tạo input `[10000, 3]`.

Một row mẫu có `label_id=1` đã xác nhận:

```text
773 packet thật:
    label column = 1
    direction thuộc {0, 1}
    relative_time trong mẫu từ 0 đến khoảng 30.49
    packet_size trong mẫu từ 92 đến 1412

9,227 packet padding:
    [0, 0, 0, 0]
```

Không giả định mỗi label luôn có đúng 100 instance. Bộ mẫu có một số label chỉ có khoảng 48-97 dòng trong mỗi split.

#### Trường hợp `label_id=0`

Trong `label_mapping.csv` mẫu, dòng `label_id=0` có `label_name` rỗng. Loader hiện tại chuyển tên rỗng thành fallback `label_0`.

Với class 0, cột label của packet thật và padding đều mang giá trị 0, nên không thể dùng cột label để phân biệt packet thật. Điều này không làm hỏng mask vì mask được xác định bằng `packet_size != 0`, nhưng code phải:

- lấy class của row từ first field;
- map tên rỗng thành `label_0` hoặc một tên fallback ổn định;
- không loại bỏ row chỉ vì `label_id == 0`;
- không dùng `labels != 0` như kiểm tra duy nhất cho tính hợp lệ của class 0.

Định dạng mẫu này là data contract áp dụng cho cả nguồn AOL và nguồn icloud_100. Code không được phụ thuộc vào số row cụ thể của thư mục mẫu; các con số trên chỉ dùng để xác nhận schema.

### 3.3. Quan hệ với file PCAP

Pipeline raw-PCAP khác trong repository quy ước một PCAP/CAP/PCAPNG là một sample. Tuy nhiên, hai file CSV mà notebook VM đang đọc **không chứa đường dẫn PCAP gốc hoặc sample ID gốc**.

Vì vậy, với dữ liệu VM hiện tại, điều có thể xác nhận chắc chắn là:

```text
1 CSV row = 1 model instance
```

Chưa được phép khẳng định có thể truy ngược:

```text
CSV row -> chính xác file .pcap nào
```

trừ khi bổ sung manifest từ quá trình tạo CSV.

Nếu yêu cầu xóa đến từ tên/path PCAP, cần có file provenance dạng:

```text
sample_id, source, split_file, line_number, original_pcap_path, original_label, source_sha256
```

## 4. Định danh instance

### 4.1. ID đề xuất

Theo đề xuất dùng vị trí dòng, ID logic là:

```text
{dataset_id}:{source}:{split_file}:{line_number}
```

`line_number` là số dòng vật lý **1-based** trong `train_data.csv` hoặc `test_data.csv`. Hai file này không có header nên dòng 1 cũng là instance đầu tiên.

Ví dụ:

```text
aol_v1:AOL:train_data.csv:1234
```

Không được dùng `original_label_id + thứ tự trong label` làm ID vì CSV được lưu theo dòng và số lượng mỗi label không đồng đều.

Không dùng `byte_offset` làm ID chính vì byte offset thay đổi nếu CSV được tạo lại hoặc một dòng phía trước thay đổi độ dài.

Để tránh chọn nhầm instance khi file CSV bị thay thế, manifest phải kèm:

```text
dataset fingerprint:
    resolved path
    file size
    mtime_ns
    optional SHA-256 của toàn file

instance verification:
    line_number
    byte_offset dùng trong run hiện tại
    SHA-256 của raw CSV row
```

`line_number` là khóa dễ đọc; `row_sha256` là khóa xác minh nội dung.

### 4.2. Forget manifest

Dùng một file JSON, ví dụ `forget_instances.json`:

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
      "original_label": "..."
    }
  ]
}
```

Khi load manifest, code phải fail sớm nếu:

- instance không thuộc AOL;
- instance không thuộc base-training split;
- line không tồn tại;
- row hash không khớp;
- instance ID bị trùng;
- instance không xuất hiện trong `split_manifest.json` của base model.

## 5. Split protocol

Giữ protocol của `plan_demo.md`:

- `test_data.csv` là final test cố định và không dùng để train hoặc chọn hyperparameter.
- `train_data.csv` của từng nguồn được chia deterministic theo `(source, original_label)`:
  - 85% base train;
  - 15% validation.
- AOL luôn có binary target `known=0`.
- icloud_100 luôn có binary target `unknown=1`.

`split_manifest.json` phải được lưu và tái sử dụng nguyên vẹn. Không chạy lại phép chia sau khi nhận forget request.

### 5.1. Tập dùng cho unlearning

Chỉ instance đã thực sự tham gia base training mới có thể là thành viên `Df`:

```text
Df_train = forget IDs giao với base train
Dr_train = base train trừ Df_train
```

Quan trọng:

```text
Các instance khác có cùng original_label với Df vẫn thuộc Dr.
```

Không tự động đưa validation/test cùng nhãn vào `Df`. Chúng là control để đo collateral forgetting.

### 5.2. Các tập đánh giá

```text
Df_exact:
    chính các instance được yêu cầu quên

Dr_same_label_val/test:
    instance không bị quên nhưng có original_label xuất hiện trong Df

Dr_other_known_val/test:
    AOL thuộc các original label khác

Unknown_val/test:
    instance từ icloud_100

Full_retain_val/test:
    toàn bộ dữ liệu đánh giá không thuộc Df
```

## 6. Checkpoint và trọng số

Thư mục `MLP-Classfication/vm_code/weight_trained` hiện có:

```text
pretrain_AOL.pth
pretrain_273.pth
```

Hai file này là checkpoint SupCon gồm:

```text
encoder.*
head.*  # projection head của SupCon
```

Chúng **không chứa MLP classifier known/unknown**.

Pipeline chính dùng:

```text
pretrain_AOL.pth
-> load encoder tương thích input [10000, 3]
-> train MLP binary trên AOL known + icloud_100 unknown
-> lưu base_model/best_model.pt
```

Checkpoint bắt buộc để bắt đầu unlearning là:

```text
artifacts/vm-training/experiments/<BASE_RUN_ID>/base_model/best_model.pt
```

Checkpoint này phải chứa tối thiểu:

```text
model_state_dict       # encoder + MLP
best_unknown_threshold
split_manifest hash
dataset signatures
pretrain checkpoint hash
model/feature config
base validation/test metrics
```

Mọi nhánh unlearning phải khởi tạo độc lập từ cùng một `best_model.pt`. Không chạy scope sau dựa trên trọng số của scope trước.

## 7. Các baseline và phương pháp cần so sánh

### 7.1. Base model

Model trước unlearning, dùng làm mốc.

### 7.2. Retrain oracle

Train lại binary classifier/model theo cùng cấu hình nhưng loại đúng `Df` khỏi base train:

```text
base train oracle = Dr_train
```

Oracle tốn thời gian nhưng cần thiết để biết model unlearned có gần với trạng thái “chưa từng train trên Df” hay không.

### 7.3. Relabel-only

Chỉ dùng targeted relabeling:

```text
L_forget = CE(model(x_f), unknown)
```

Đây là baseline hành vi đơn giản nhất.

### 7.4. Retain-rehearsal baseline

Giữ cách train hiện tại nhưng chọn `Df` theo instance ID:

```text
L = CE(model(Df), unknown)
  + lambda_retain * CE(model(Dr), original_binary_label)
```

Baseline này được phép dùng `Dr` khi unlearning, nên không phải thiết lập data-free của paper. Cần ghi tên rõ là `retain_rehearsal`.

### 7.5. L2UL-MAS

Dùng targeted relabeling cùng weight-importance regularization của paper, không dùng `Dr` trong loss unlearning.

### 7.6. L2UL-Adv

Dùng targeted relabeling cùng adversarial-example regularization, không dùng `Dr` trong loss unlearning.

### 7.7. L2UL-Adv+MAS

Phương pháp đầy đủ được ưu tiên:

```text
targeted relabeling + adversarial regularization + MAS anchor
```

## 8. Kỹ thuật unlearning theo paper

### 8.1. Targeted relabeling

Vì `Df` chỉ chọn từ AOL:

```text
y_old  = known   = 0
y_star = unknown = 1
```

Loss quên:

```text
L_forget = CE(f_theta_u(x_f), y_star)
```

Không sửa dữ liệu gốc trên đĩa và không ghi đè binary label trong base manifest. `y_star` chỉ là unlearning target tạo lúc runtime.

### 8.2. MAS importance trên Df

Tính importance từ model gốc `theta_0` và chỉ từ `Df`:

```text
Omega_i = (1 / |Df|) * sum_x | d ||f_theta_0(x)||_2 / d theta_i |
```

Trong implementation đầu tiên, `f_theta_0(x)` là vector logits binary. Có thể chạy ablation dùng embedding, nhưng phải ghi tên riêng.

Chuẩn hóa importance theo từng parameter tensor:

```text
Omega_norm_i = min_max_normalize(Omega_i)
anchor_i     = 1 - Omega_norm_i
```

Việc đảo importance là có chủ đích:

- parameter liên quan mạnh đến `Df` có anchor nhỏ, được phép thay đổi;
- parameter ít liên quan đến `Df` có anchor lớn, bị giữ gần `theta_0`.

MAS regularization:

```text
L_MAS = sum_i anchor_i * (theta_u_i - theta_0_i)^2 / 2
```

Chỉ tính và lưu importance cho parameter trainable trong scope đang chạy.

### 8.3. Targeted adversarial examples

Với mỗi `(x_f, known)`, dùng một bản frozen của base model để tạo:

```text
x_adv = x_f + delta
target = unknown
```

Trong lúc sinh `x_adv`:

- model ở `eval()`;
- không cập nhật trọng số;
- chỉ cập nhật input adversarial;
- tối thiểu hóa `CE(base_model(x_adv), unknown)`;
- sau mỗi bước phải project perturbation vào miền cho phép.

Adversarial loss khi unlearning:

```text
L_adv = CE(f_theta_u(x_adv), unknown)
```

Adversarial samples được sinh từ base checkpoint trước khi unlearning và cache theo:

```text
base checkpoint SHA-256
instance_id
PGD config
feature contract version
```

### 8.4. Ràng buộc adversarial cho traffic

Không áp dụng một epsilon trực tiếp cho cả ba raw feature vì chúng có scale và semantics khác nhau.

Ràng buộc ban đầu:

```text
mask/padding:
    giữ nguyên; vị trí padding luôn bằng 0

direction:
    giữ nguyên; không perturb biến rời rạc

relative_time:
    cho phép perturb nhỏ trong scaled space
    clamp không âm
    bảo toàn quy tắc thời gian của preprocessing

packet_size:
    cho phép perturb nhỏ trong scaled space
    clamp vào miền kích thước packet hợp lệ
    tùy cấu hình có thể round về giá trị nguyên
```

Nên dùng scale riêng theo channel:

```text
delta_raw[channel] = delta_scaled[channel] * feature_scale[channel]
```

Không thay đổi preprocessing/normalization của encoder pretrained. Việc scale ở đây chỉ phục vụ giới hạn perturbation rồi chuyển ngược về raw tensor tương thích checkpoint.

Milestone an toàn là triển khai `L2UL-MAS` trước, sau đó mới bật constrained PGD.

### 8.5. Loss tổng

Đối với phương pháp đầy đủ:

```text
L_total = L_forget
        + lambda_adv * L_adv
        + lambda_mas * L_MAS
```

Không có `CE(Dr)` trong mode `l2ul_adv_mas`.

Đối với baseline hiện tại:

```text
L_total = L_forget
        + lambda_retain * L_retain
```

Hai mode phải được báo cáo tách biệt.

## 9. Phạm vi cập nhật trọng số

Giữ ba scope của notebook hiện tại:

```text
head_only:
    chỉ MLP classifier

last_encoder_block:
    conv4, conv4_4, batch_norm4, fc của encoder + MLP

full_encoder_and_head:
    toàn bộ encoder + MLP
```

Mỗi scope:

1. Load lại cùng base checkpoint.
2. Freeze tất cả parameter.
3. Mở `requires_grad=True` đúng module của scope.
4. Tính MAS riêng cho parameter trainable.
5. Tạo optimizer mới.
6. Chạy unlearning độc lập.

Khi `head_only`, encoder phải ở `eval()` để BatchNorm/Dropout không thay đổi trạng thái.

Khi chỉ mở block cuối, các block encoder đã freeze phải giữ `eval()`; chỉ module được cập nhật và MLP chuyển sang train mode.

`head_only` là baseline hợp lệ cho behavioral forgetting, nhưng không đủ để kết luận thông tin của `Df` đã bị xóa khỏi encoder.

## 10. Thiết kế experiment instance-wise

### 10.1. Quy mô forget request

Mặc định đề xuất các tập nested:

```text
FORGET_INSTANCE_COUNTS = (1, 10, 50, 100)
```

Dùng một thứ tự instance cố định theo `FORGET_INSTANCE_SEED`; tập lớn là prefix mở rộng của tập nhỏ:

```text
Df_1 subset Df_10 subset Df_50 subset Df_100
```

Chọn stratified trên nhiều `original_label` AOL để tránh vô tình biến experiment thành quên một nhãn.

Lưu toàn bộ selection vào manifest trước khi train. Không random lại giữa các method/scope/repeat.

### 10.2. Repeats

```text
SELECTION_SEED:
    cố định Df

OPTIMIZATION_SEEDS:
    tối thiểu 3 seed cho từng method x scope x forget_count
```

Selection seed và optimization seed phải tách riêng trong artifact.

### 10.3. Chọn checkpoint unlearning

Đối với mode theo paper không dùng `Dr`:

- dùng số epoch cố định đã chọn từ development experiments; hoặc
- early stop khi forget success trên `Df_exact` đạt target;
- không sweep threshold sau unlearning.

Đối với nghiên cứu có cho phép dùng validation retain để chọn epoch, phải ghi rõ đây là evaluation-assisted selection, không phải strict `Df`-only setting.

Final test chỉ chạy một lần sau khi method/hyperparameter đã chốt.

## 11. Metrics

Giữ `best_unknown_threshold` của base model cố định cho toàn bộ nhánh unlearning.

Không chọn threshold mới sau unlearning.

### 11.1. Forget metrics

```text
forget_success_rate:
    P(predicted unknown | Df_exact)

forget_confidence_shift:
    mean P_unknown_after(Df) - mean P_unknown_before(Df)

old_label_accuracy_on_Df:
    P(predicted known | Df_exact), cần giảm
```

### 11.2. Retain metrics

```text
same_label_known_recall:
    known recall trên instance không bị quên nhưng cùng original_label với Df

remaining_known_recall:
    known recall trên toàn bộ AOL retain

unknown_recall:
    unknown recall trên icloud_100

retain_balanced_accuracy:
    (remaining_known_recall + unknown_recall) / 2
```

### 11.3. Collateral damage

```text
same_label_damage = same_label_known_recall_before
                  - same_label_known_recall_after

known_damage      = remaining_known_recall_before
                  - remaining_known_recall_after

balanced_damage   = retain_balanced_accuracy_before
                  - retain_balanced_accuracy_after
```

### 11.4. So sánh oracle

So sánh unlearned model với retrain oracle bằng:

- chênh lệch metric trên retain test;
- agreement của prediction;
- khoảng cách phân phối probability/logit;
- tùy chọn: khoảng cách embedding trên retain set.

Membership-inference hoặc privacy attack là đánh giá bổ sung nếu muốn tuyên bố mạnh hơn behavioral forgetting.

### 11.5. Tiêu chí thành công ban đầu

Giá trị mặc định để sàng lọc, cần chốt lại sau development run:

```text
forget_success_rate >= 0.95
retain_balanced_accuracy drop <= 0.02
same_label_known_recall drop <= 0.05
```

Không chỉ báo cáo accuracy tổng vì mất cân bằng known/unknown có thể che giấu lỗi.

## 12. Thay đổi cần thực hiện trong notebook

### 12.1. Config cell

Tách rõ runtime config của VM khỏi fixture trên máy phát triển. Giá trị `/home/ubuntu/...` chỉ được resolve khi chạy full experiment trên VM:

```python
VM_AOL_DATASET_DIR = Path(
    os.environ.get(
        'AOL_DATASET_DIR',
        '/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud',
    )
)
VM_UNKNOWN_273_DATASET_DIR = Path(
    os.environ.get(
        'UNKNOWN_273_DATASET_DIR',
        '/home/ubuntu/Documents/KF/data_processed/icloud_100',
    )
)

LOCAL_CSV_SAMPLE_DIR = (
    Path.cwd()
    / 'MLP-Classfication'
    / 'csv_sample-20260921T063615Z-1-001'
    / 'csv_sample'
    / 'icloud'
)
```

`LOCAL_CSV_SAMPLE_DIR` chỉ phục vụ parser/schema test. Nó không được dùng làm fallback âm thầm cho base training hoặc unlearning. Full run phải nhận explicit mode/config và fail sớm nếu hai dataset VM không tồn tại.

Thay:

```text
FORGET_LABEL_COUNTS
FORGET_LABEL_SEED
```

bằng:

```text
FORGET_MANIFEST_PATH
FORGET_INSTANCE_COUNTS
FORGET_INSTANCE_SEED
UNLEARNING_METHODS
PGD_EPS_BY_FEATURE
PGD_ALPHA_BY_FEATURE
PGD_STEPS
LAMBDA_ADV
LAMBDA_MAS
```

### 12.2. CsvRecord và index

Bổ sung hoặc tính được:

```text
instance_id
row_sha256
dataset_signature_id
base_split_membership
```

`byte_offset` vẫn giữ để đọc nhanh.

### 12.3. Split manifest

Manifest phải lưu danh sách instance ID của:

```text
train
validation
test
```

và hash của dataset/index config.

### 12.4. Selection functions

Thay `forget_sets(records, labels)` bằng:

```python
def forget_sets_by_instance(records, forget_ids):
    forget_ids = set(forget_ids)
    forget = [record for record in records if record.instance_id in forget_ids]
    retain = [record for record in records if record.instance_id not in forget_ids]
    return forget, retain
```

Thêm validation bắt buộc cho manifest.

### 12.5. Unlearning functions mới

```text
load_forget_manifest(...)
validate_forget_manifest(...)
estimate_mas_importance(...)
invert_and_normalize_importance(...)
parameter_anchor_loss(...)
generate_targeted_traffic_adversarial(...)
run_instance_unlearning(...)
evaluate_forget_instances(...)
evaluate_retain_slices(...)
```

### 12.6. Experiment loop

Thay vòng lặp theo `forget_labels` bằng:

```text
forget_count
-> exact forget_instance_ids
-> method
-> update scope
-> optimization repeat
```

Mọi nhánh đều load lại base checkpoint.

### 12.7. Artifact output

```text
artifacts/vm-training/experiments/<RUN_ID>/
├── base_model/
│   └── best_model.pt
├── split_manifest.json
├── forget_manifests/
│   ├── forget_1.json
│   ├── forget_10.json
│   ├── forget_50.json
│   └── forget_100.json
└── instance_unlearning/
    ├── summary.json
    └── forget_<N>/
        └── <method>/
            └── <scope>/
                └── repeat_<R>/
                    ├── unlearned_model.pt
                    ├── result.json
                    ├── history.json
                    ├── mas_importance.pt
                    ├── adversarial_manifest.json
                    └── README.md
```

Không bắt buộc lưu toàn bộ adversarial tensor nếu có thể tái tạo deterministic; nếu lưu, phải kèm checkpoint/config hash.

## 13. Kiểm thử

### 13.1. Unit checks

- Một `instance_id` map đúng một CSV row.
- Row hash khớp sau khi seek bằng byte offset.
- `Df` và `Dr` không giao nhau.
- Mỗi row có đúng `40,000` trường và reshape được thành `[10000, 4]`.
- Parser chấp nhận số thực ở dạng decimal và scientific notation.
- Packet padding có `packet_size == 0` và bị mask khỏi encoder input.
- Row thuộc `label_id=0` vẫn được index, split và load bình thường.
- Tên rỗng của label 0 được map deterministic thành `label_0`.
- Code không giả định mọi original label có cùng số instance.
- `Df union Dr` bằng base train.
- Mọi `Df` đều thuộc AOL/base train.
- Instance cùng original label nhưng không có trong manifest vẫn thuộc `Dr`.
- Frozen parameter không thay đổi bitwise sau unlearning.
- Mỗi scope chỉ thay đổi parameter được cho phép.
- Padding và direction không đổi sau constrained PGD.
- Adversarial tensor nằm trong bound theo từng feature.
- Base threshold được giữ nguyên.

### 13.2. Smoke test

Chạy với:

```text
1 forget instance
1 optimization seed
1 epoch
head_only
relabel_only và l2ul_mas
```

Xác minh artifact và metric trước khi chạy grid đầy đủ.

### 13.3. Sanity checks

- `forget_success_before` phải được ghi lại; nếu instance đã là unknown trước unlearning thì không dùng nó để chứng minh hiệu quả quên.
- Ưu tiên chọn `Df` trong các instance base model đang dự đoán đúng là known.
- `Dr_same_label` phải không rỗng.
- Không dùng test để chọn instance, epoch, lambda hoặc epsilon.

## 14. Thứ tự triển khai

### Giai đoạn 1 — Instance identity

1. Bổ sung stable instance ID và row hash.
2. Lưu/reuse split manifest.
3. Tạo và validate forget manifest.
4. Thay selection theo label bằng selection theo instance ID.

### Giai đoạn 2 — Instance-wise baseline

1. Chạy `relabel_only`.
2. Chạy `retain_rehearsal`.
3. Giữ ba update scope.
4. Thêm same-label collateral metrics.

### Giai đoạn 3 — MAS

1. Implement MAS từ `Df`.
2. Đảo normalized importance đúng theo paper/official implementation.
3. Chạy `l2ul_mas` trên ba scope.

### Giai đoạn 4 — Adversarial regularization

1. Chốt feature scaling và miền hợp lệ.
2. Implement constrained targeted PGD.
3. Kiểm thử tính hợp lệ của traffic tensor.
4. Chạy `l2ul_adv` và `l2ul_adv_mas`.

### Giai đoạn 5 — Benchmark

1. Chạy nested forget counts.
2. Chạy tối thiểu ba optimization seeds.
3. Train retrain oracle cho từng forget set.
4. Tổng hợp forget/retain/collateral/oracle metrics.

## 15. Rủi ro và giới hạn

### 15.1. Row ID không phải PCAP provenance

Line number đủ để chọn model instance trong CSV cố định, nhưng không đủ để chứng minh file PCAP gốc nào đã bị xóa. Cần upstream manifest nếu yêu cầu xóa gắn với file nguồn.

### 15.2. Head-only không xóa representation

MLP có thể đổi output sang unknown trong khi encoder vẫn giữ embedding cũ của `Df`. Vì vậy phải báo cáo scope và không diễn giải `head_only` thành xóa hoàn toàn ảnh hưởng khỏi model.

### 15.3. Traffic adversarial validity

Perturbation hợp lệ về norm chưa chắc hợp lệ về network semantics. Không công bố kết quả `Adv` là paper-equivalent cho tới khi các ràng buộc feature được kiểm tra.

### 15.4. Behavioral forgetting không phải certified deletion

`Df -> unknown` đáp ứng mục tiêu hành vi của dự án, nhưng không tự động đảm bảo privacy hoặc phân phối tham số giống retrain oracle.

### 15.5. Checkpoint provenance

`pretrain_AOL.pth` cung cấp encoder pretrained, không cung cấp binary MLP. Mọi experiment phải ghi rõ base binary checkpoint và hash của nó.

## 16. Các quyết định mặc định cần xác nhận

Plan tạm dùng các mặc định sau:

1. Chỉ quên instance thuộc AOL và target luôn là `unknown=1`.
2. Dùng `source + split_file + line_number` làm ID dễ đọc, kèm row hash để xác minh.
3. Chỉ instance trong base-training split được phép vào `Df`.
4. Quy mô benchmark ban đầu là `(1, 10, 50, 100)` instance, stratified trên nhiều original label.
5. `base_model/best_model.pt` là điểm bắt đầu của mọi nhánh unlearning.
6. Triển khai `relabel_only` và `l2ul_mas` trước constrained adversarial PGD.
7. Giữ threshold của base model cố định sau unlearning.

Các điểm cần người thực hiện xác nhận trước benchmark chính thức:

- Có manifest nào ánh xạ từng CSV row về file PCAP gốc hay không?
- Forget request thực tế sẽ cung cấp line number, sample ID hay đường dẫn PCAP?
- Quy mô `Df` mong muốn có đúng là `(1, 10, 50, 100)` hay cần tỷ lệ phần trăm?
- Strict experiment có cấm dùng `Dr` cả trong train lẫn model selection hay chỉ cấm dùng `Dr` trong loss?

## 17. Implementation notebook và cách chạy

Code triển khai của plan nằm trực tiếp trong notebook:

```text
MLP-Classfication/vm_code/instance_wise/train_instance_wise.ipynb
```

Notebook chứa toàn bộ parser, model, base training, manifest selection, MAS, constrained PGD và unlearning. Nó không import implementation từ file `.py` khác.

### 17.1. Cấu trúc notebook

Notebook được chia thành các nhóm cell:

```text
1. Imports, constants, records, common helpers
2. CSV paths, byte-offset index, instance IDs, splits
3. Row decoding, tensor cache, data loaders
4. Checkpoint-compatible encoder và binary MLP
5. Evaluation, base training, threshold, checkpoint loading
6. Update scopes, MAS, constrained PGD, unlearning loop
7. Forget manifests và retain slices
8. Pipeline phase implementations
9. Argument definitions dùng nội bộ
10. Notebook phase configuration
11. Execute selected phase
```

Cell cấu hình cuối notebook có:

```python
RUN_PIPELINE = False
NOTEBOOK_COMMAND = "validate-schema"
NOTEBOOK_ARGUMENTS_BY_COMMAND = {...}
```

Phải review cấu hình trước, sau đó đặt:

```python
RUN_PIPELINE = True
```

Notebook chỉ chạy đúng một phase được chọn; không tự động train/unlearn khi vừa mở file.

### 17.2. Kiểm tra schema trên máy cá nhân

Chọn:

```python
NOTEBOOK_COMMAND = "validate-schema"
RUN_PIPELINE = True
```

Phase này dùng:

```text
MLP-Classfication/csv_sample-20260921T063615Z-1-001/csv_sample/icloud
```

Nó không yêu cầu các path `/home/ubuntu/...` tồn tại và không chạy model training.

### 17.3. Train base model trên VM

Sau khi đưa repository lên VM, chọn:

```python
NOTEBOOK_COMMAND = "train-base"
RUN_PIPELINE = True
```

Config mặc định trong notebook trỏ tới:

```text
AOL known:
    /home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud

273 unknown:
    /home/ubuntu/Documents/KF/data_processed/icloud_100

pretrained encoder:
    MLP-Classfication/vm_code/weight_trained/pretrain_AOL.pth
```

Mặc định encoder được freeze và chỉ train binary MLP. Checkpoint output:

```text
artifacts/vm-training/instance-wise/base_run/base_model/best_model.pt
```

### 17.4. Tạo nested forget manifests

Sau khi có base checkpoint, đổi:

```python
NOTEBOOK_COMMAND = "make-manifests"
RUN_PIPELINE = True
```

Notebook mặc định tạo các tập nested:

```text
forget_1.json
forget_10.json
forget_50.json
forget_100.json
```

Chỉ AOL instance thuộc base train và được base model dự đoán đúng là known mới đủ điều kiện được chọn.

### 17.5. Chạy instance-wise unlearning

Chọn manifest cần chạy trong `NOTEBOOK_ARGUMENTS_BY_COMMAND["unlearn"]`, sau đó đặt:

```python
NOTEBOOK_COMMAND = "unlearn"
RUN_PIPELINE = True
```

Mặc định notebook so sánh:

```text
methods:
    relabel_only
    retain_rehearsal
    l2ul_mas
    l2ul_adv
    l2ul_adv_mas

scopes:
    head_only
    last_encoder_block
    full_encoder_and_head
```

Mỗi method/scope/repeat luôn load độc lập từ cùng base checkpoint. Threshold unknown của base model được giữ nguyên.

### 17.6. Mapping từ plan sang notebook code

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

Notebook chỉ dùng Python standard library, NumPy và PyTorch; không thêm dependency adversarial bên ngoài. Full training/unlearning mặc định yêu cầu CUDA. CPU chỉ phù hợp cho schema/parser checks hoặc debug nhỏ, không dành cho benchmark chính thức.

