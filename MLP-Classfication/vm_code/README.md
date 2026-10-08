# Pipeline phân loại và instance-wise unlearning trên máy ảo

`vm_code` chứa code để chạy bằng CPU/GPU của **máy ảo (VM)**. Code được phát triển/review trên máy cá nhân, cập nhật qua Git, rồi chạy trên VM với dataset local đã được cung cấp. Dữ liệu đầu vào, cache và kết quả thí nghiệm đều nằm trên filesystem của VM.

Notebook chính là [instance_wise/train_instance_wise.ipynb](instance_wise/train_instance_wise.ipynb), với kế hoạch tại [instance_wise/plan_instance_wise.md](instance_wise/plan_instance_wise.md). README này mô tả implementation hiện có; các đầu việc trong plan chưa chắc đã được triển khai.

## 1. Cấu trúc thư mục

| Đường dẫn | Vai trò |
| --- | --- |
| `instance_wise/train_instance_wise.ipynb` | Notebook chính: tạo model gốc và quên từng instance. Toàn bộ logic nằm trong notebook. |
| `instance_wise/plan_instance_wise.md` | Đặc tả instance-wise, protocol đánh giá và backlog. |
| `instance_wise/requirements.txt` | Dependency Python của notebook. |
| `weight_encoder_trained/` | Checkpoint encoder SupCon pretrained; chưa có binary MLP đã train. |
| `out_data/` | Output từ các thí nghiệm trên VM, báo cáo tiếng Việt và CSV giải thích kết quả. |

Notebook instance-wise dùng `RUN_PIPELINE` và `NOTEBOOK_COMMAND` để điều khiển phase thực thi.

## 2. Dataset local trên VM

Hai đường dẫn tuyệt đối đã được người dùng cung cấp:

```python
VM_AOL_DATASET_DIR = Path("/home/ubuntu/Documents/KF/data_processed/AOL_csv/icloud")
VM_UNKNOWN_273_DATASET_DIR = Path("/home/ubuntu/Documents/KF/data_processed/icloud_100")
```

| Dataset | Nhãn binary | Ý nghĩa |
| --- | --- | --- |
| AOL | `known = 0` | Nguồn known; các instance cần quên được chọn từ đây. |
| icloud_100, còn gọi là 273 | `unknown = 1` | Nguồn unknown dùng khi train MLP và đánh giá. |

Mỗi dataset phải chứa `train_data.csv`, `test_data.csv`, `label_mapping.csv`.

Mỗi dòng CSV không có header, gồm 40.000 giá trị:

```text
[label_id, relative_time, direction, packet_size] × 10.000
    → reshape [10000, 4]
    → bỏ cột label, giữ features [10000, 3] + mask [10000]
```

Feature dùng `legacy_raw_v1`, giữ giá trị time/direction/packet size theo contract của encoder pretrained. Input có shape `[10000,3]`; **256 là chiều embedding đầu ra encoder**. Padding được nhận diện bằng `packet_size == 0`, kể cả khi `label_id == 0`.

Một dòng CSV là một model instance. Nhãn gốc được giữ trong metadata để chia split và đánh giá ảnh hưởng trên các mẫu cùng nhãn. Notebook không trực tiếp đọc `.pcap`, và chưa có manifest ánh xạ CSV row về PCAP gốc.

`train_data.csv` được chia deterministic theo `(source, original_label)`, tỷ lệ mặc định 85% train / 15% validation. `test_data.csv` là final test cố định, không dùng chọn epoch hoặc threshold. Sample trên máy cá nhân chỉ dùng kiểm tra parser/schema; training không tự fallback sang sample.

## 3. Model gốc dùng chung cho các method

```text
features [10000, 3] + mask
    → FlowEncoder tương thích encoder SupCon pretrained
    → embedding 256 chiều
    → MLP: 256 → 512 → 256 → 128 → 64 → 32 → 2
    → logits known / unknown
```

Phase `train-base` nạp **phần encoder** từ `pretrain_AOL.pth`, khởi tạo binary MLP mới và train trên cả AOL lẫn icloud_100. Mặc định encoder được freeze và ở `eval()`; thêm `--fine-tune-encoder` mới cập nhật encoder cùng MLP.

Optimizer là AdamW; base loss là CrossEntropy có trọng số cân bằng lớp. Cấu hình mặc định: 16 epoch, batch size 16, learning rate `2e-4`, weight decay `1e-4`, seed 42. Checkpoint được chọn theo validation balanced accuracy tại threshold 0,5; sau đó chọn threshold trên validation và đánh giá test.

**Model gốc `M0` là encoder + MLP sau phase `train-base`, trước unlearning**, lưu trong `base_run/base_model/best_model.pt`. `pretrain_AOL.pth` riêng lẻ chưa phải model phân loại hoàn chỉnh.

Checkpoint chứa trạng thái toàn bộ model, model/feature config, threshold, split IDs, dataset signatures, provenance/hash encoder pretrained và test metrics. Lịch sử validation và threshold sweep nằm trong `base_run/base_model/result.json`.

Mọi nhánh unlearning bắt đầu từ cùng checkpoint; code tạo `copy.deepcopy(base_model)` cho từng method/scope/repeat. Threshold được giữ cố định sau unlearning.

### Đường dẫn encoder và repo root

Folder checkpoint hiện tại là:

```text
MLP-Classfication/vm_code/weight_encoder_trained/pretrain_AOL.pth
MLP-Classfication/vm_code/weight_encoder_trained/pretrain_273.pth
```

Cấu hình `train-base` lấy đường dẫn encoder từ `DEFAULT_PRETRAIN`, trỏ tới `weight_encoder_trained/pretrain_AOL.pth`.

Notebook xác định `REPO_ROOT` từ working directory hiện tại hoặc các folder cha chứa notebook trong repo. Có thể đặt biến môi trường `BKCS_REPO_ROOT` thành đường dẫn clone trước khi khởi động Jupyter nếu kernel chạy ngoài repo. Encoder, sample và artifact mặc định đều được neo vào repo root, không phụ thuộc folder chứa notebook.

Checkpoint `pretrain_AOL.pth` được theo dõi trong Git và đồng bộ qua clone/pull. Các file `.pth` chưa được theo dõi bị ignore; `pretrain_273.pth` là checkpoint bổ sung local, không cần cho pipeline mặc định và không tự được đồng bộ. Xác nhận checkpoint tồn tại trên VM; notebook không tự tải checkpoint thay thế. [README encoder](weight_encoder_trained/README.md) mô tả cách nạp riêng encoder bằng `strict=True`.

## 4. Chạy notebook trên VM

Sử dụng môi trường Python đã cài NumPy, PyTorch và Jupyter; dependency notebook được liệt kê trong `instance_wise/requirements.txt`. Trên VM đã có PyTorch CUDA hoạt động, giữ môi trường GPU đã xác minh. Full training/unlearning mặc định dùng `cuda:0`; có thể kiểm tra bằng:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT FOUND")'
```

Từ thư mục repo trên VM, mở Jupyter bằng môi trường đã tạo:

```bash
cd /home/ubuntu/Documents/huyenchi/BKCS-unlearning
conda activate bkcs-unlearning
python -m notebook
```

Nếu repo được clone sang vị trí khác, thay đường dẫn `cd`. Trong Jupyter, mở `MLP-Classfication/vm_code/instance_wise/train_instance_wise.ipynb` và chọn kernel của môi trường `bkcs-unlearning`.

Notebook tự xác định repo root khi Jupyter khởi tạo kernel ở repo root hoặc một folder con. Không cần thêm cell `os.chdir()` hay hardcode vị trí clone. Các đường dẫn tùy chỉnh dạng tương đối mà người dùng truyền thêm vẫn được hiểu theo working directory của kernel; dùng path tuyệt đối cho output riêng ngoài repo. Sau đó:

1. Chạy tất cả cell định nghĩa ở các mục 1–9.
2. Ở mục **Notebook phase configuration**, kiểm tra log `Repo root`, `Encoder checkpoint` và các biến `VM_*`.
3. Đặt `RUN_PIPELINE = True`, chọn `NOTEBOOK_COMMAND` theo bảng dưới.
4. Chạy lại cell cấu hình, rồi cell **Execute selected phase**. Khi đổi phase, cần chạy lại cả hai cell này.

| Thứ tự | `NOTEBOOK_COMMAND` | Kết quả |
| --- | --- | --- |
| Tùy chọn | `validate-schema` | Kiểm tra CSV sample; mặc định dành cho máy phát triển, chỉ chạy trên VM nếu đã cấu hình sample có thật. |
| 1 | `train-base` | Train và lưu model gốc encoder + MLP. |
| 2 | `make-manifests` | Tạo tập quên nested gồm 1, 4, 10, 50, 100 instance AOL từ base train. |
| 3 | `unlearn` | Chạy các method/scope/repeat cho manifest được chọn. |

Notebook mặc định `RUN_PIPELINE = False` và `NOTEBOOK_COMMAND = "validate-schema"`, nên **Run All với cấu hình mặc định không train model**. Để tạo model gốc trên VM, chọn `"train-base"` và bật `RUN_PIPELINE`.

`make-manifests` chọn các instance AOL được model gốc dự đoán đúng là known, phân bố qua nhiều nhãn gốc. Manifest giữ instance ID, SHA-256 của row và hash checkpoint để kiểm tra đúng dữ liệu/model.

Cấu hình `unlearn` mặc định quên **4 instance**, chọn **`forget_4.json`**, 5 method × 3 scope × 3 repeat = 45 nhánh. Các count 1/10/50/100 cần đổi `--forget-manifest` rồi chạy riêng. Nên smoke test trước bằng `forget_1.json`, method `relabel_only,l2ul_mas`, scope `head_only`, `--repeats 1`, `--unlearning-epochs 1`.

Cell đang chạy thường hiển thị `[*]`; log train-base có `Base epoch ...`, log unlearning có `[method/scope] epoch ...`. Run All cũng thực thi phase đã chọn nên không chạy lại khi một lượt dài vẫn đang hoạt động.

## 5. Instance-wise unlearning hiện có

`Df` chỉ gồm **đúng các dòng AOL thuộc base train được liệt kê trong manifest**; target mới là `unknown=1`. `Dr` là base train trừ các instance đó. Các mẫu cùng nhãn gốc nhưng không có trong manifest vẫn thuộc phần cần giữ, kể cả validation/test.

| Method | Objective hiện có |
| --- | --- |
| `relabel_only` | `CE(Df → unknown)` |
| `retain_rehearsal` | `CE(Df → unknown) + lambda_retain × CE(Dr → nhãn binary gốc)` |
| `l2ul_mas` | Forget CE + penalty giữ trọng số gần model gốc theo inverted MAS anchor. |
| `l2ul_adv` | Forget CE + CE trên targeted adversarial examples sinh từ `Df`. |
| `l2ul_adv_mas` | Forget CE + adversarial CE + MAS penalty. |

Các mode `l2ul_*` không dùng `Dr` trong loss cập nhật, nhưng code vẫn đánh giá retain validation trong lịch sử chạy. `retain_rehearsal` có dùng `Dr` trong loss và cần báo cáo riêng. Unlearning hiện lưu model ở epoch cuối, chưa chọn best epoch hoặc early stopping.

| Update scope | Thành phần được cập nhật |
| --- | --- |
| `head_only` | Chỉ MLP; encoder ở `eval()`. |
| `last_encoder_block` | Block CNN cuối, BatchNorm cuối, `encoder.fc` và MLP. |
| `full_encoder_and_head` | Toàn bộ encoder + MLP. |

Ba scope là phạm vi cập nhật, không phải ba thuật toán khác nhau. Việc chỉ cập nhật MLP có thể đổi output thành unknown nhưng không chứng minh representation trong encoder đã quên instance đó.

## 6. Artifact và khả năng chạy lại

Với cấu hình `VM_OUTPUT_ROOT = REPO_ROOT / "artifacts/vm-training/instance-wise"`, output nằm trên VM:

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
│       └── forget_4/
│           ├── adversarial_cache/         # khi chạy method Adv
│           ├── summary.json
│           └── <method>/<scope>/repeat_<r>/
│               ├── unlearned_model.pt
│               ├── result.json            # gồm history và after metrics
│               └── mas_anchors.pt         # khi chạy method MAS
└── forget_manifests/
    ├── forget_1.json
    ├── forget_4.json
    ├── forget_10.json
    ├── forget_50.json
    └── forget_100.json
```

Đây là layout theo **cell cấu hình notebook**; parser defaults đã đồng bộ với layout này. `summary.json` chứa baseline before và danh sách kết quả từng nhánh. Các artifact riêng dự kiến trong plan như README/history/config theo run chưa được triển khai đầy đủ.

CSV được đọc theo byte offset, decode từng row khi cần; feature cache lưu features/mask, không giữ toàn bộ CSV trong RAM. Dung lượng RAM/VRAM vẫn phụ thuộc batch size, workers, model, cache adversarial và scope. Nếu thiếu VRAM, giảm `--batch-size`; nếu áp lực RAM do loader, giảm `--num-workers`.

Index, feature cache và adversarial cache có cơ chế tái sử dụng. Base checkpoint có thể nạp để chạy manifest/unlearning mà không train lại base. Tuy nhiên, **chưa có resume optimizer/epoch khi training bị gián đoạn**, và chưa tự bỏ qua nhánh unlearning đã hoàn thành. Chạy lại cùng đường dẫn có thể ghi đè artifact; muốn giữ các thí nghiệm độc lập, đổi output root và các path checkpoint/manifest/cache tương ứng.

## 7. Đánh giá và đọc kết quả

| Chỉ số | Ý nghĩa |
| --- | --- |
| `accuracy` | Tỷ lệ dự đoán đúng theo nhãn binary trên tập đang đánh giá. |
| `known_recall` | Tỷ lệ known được dự đoán đúng là known. |
| `unknown_recall` | Tỷ lệ unknown được dự đoán đúng là unknown. |
| `balanced_accuracy` | Trung bình recall của hai lớp khi cả hai lớp có mặt. |
| `confusion_matrix` | Hàng là nhãn thật, cột là nhãn dự đoán; thứ tự `[known, unknown]`. |
| `forget_success_rate` | Tỷ lệ instance thuộc `Df` được dự đoán thành unknown. |
| `mean_probability_unknown` | Trung bình xác suất unknown trên `Df`. |

Validation/test được báo cáo theo các slice `same_original_label`, `other_known`, `unknown_273`, `full_retain`. Recall của lớp không có trong một slice là `NaN`; với slice chỉ có một lớp, code tính balanced accuracy từ recall của lớp hiện diện.

So sánh unlearning cần cùng model gốc, manifest, split và threshold. Báo cáo forget trước/sau cùng retain và ảnh hưởng lên các mẫu cùng nhãn; accuracy tổng thể riêng lẻ chưa đủ. Nếu coi unknown là lớp dương: FP là known bị nhận nhầm unknown, FN là unknown bị nhận nhầm known. Trên `Df`, dự đoán unknown là mục tiêu mới nên phải đo riêng, không coi đó là lỗi retain.

Đánh giá model gốc bằng `base_run/base_model/result.json` trên VM; so sánh method bằng `instance_unlearning/<manifest>/summary.json`. Khi đọc báo cáo, dùng metadata của từng run để xác định checkpoint, forget manifest, split và protocol.

## 8. Hướng mở rộng

- Retrain oracle, command `train-oracle`/`summarize`, tổng hợp mean/std tự động và README riêng cho từng run nằm trong plan, chưa được notebook hiện tại triển khai đầy đủ.
- Targeted Blindspot đã được thảo luận: student bắt đầu từ model gốc, retain học theo output của frozen teacher, forget học target unknown. Notebook chưa có teacher–student/distillation loss hoặc method này.
- Amnesiac, Blindspot nguyên bản, Boundary và MK-MMD trong bảng paper chưa được triển khai ở notebook hiện tại. Tài liệu giải thích tại [paper/vi_Unlearning_Methods_Overview.md](../../paper/vi_Unlearning_Methods_Overview.md).

## 9. Git và kết quả riêng trên VM

Sau khi thay đổi code được commit/push từ máy phát triển, cập nhật trong thư mục repo trên VM bằng:

```bash
git pull origin main
```

Nếu VM có chỉnh sửa chưa commit, kiểm tra `git status` và giữ lại các thay đổi đó trước khi pull. Git chỉ đồng bộ những file đã được commit; dataset local, cache và checkpoint phát sinh trên VM không tự được đồng bộ.

Để chia sẻ output notebook, lưu notebook bằng Jupyter sau khi chạy hoặc cung cấp các `result.json`/`summary.json`. Script xuất output mặc định đọc notebook instance-wise và ghi kết quả vào `artifacts/`. Chạy từ repo root:

```bash
python MLP-Classfication/vm_code/out_data/extract_vm_outputs.py
```

Script nhận `--notebook`/`--output` để chỉ định nguồn và nơi lưu. Chi tiết các báo cáo tại [out_data/README.md](out_data/README.md).
