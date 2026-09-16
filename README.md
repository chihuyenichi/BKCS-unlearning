# Network Flow Unlearning

Workspace này triển khai bài toán phân loại network flow theo kịch bản open-world và chuẩn bị nền cho kỹ thuật instance-wise unlearning.

Pipeline hiện tại:

```text
Raw network flow / PCAP
        |
        v
Packet parser + preprocessing
        |
        v
Tensor packet features [MAX_PACKETS, 3]
        |
        v
CNN encoder
        |
        v
Embedding 256 chiều
        |
        v
MLP classifier
        |
        v
Binary output: known / unknown
        |
        v
Instance-wise unlearning: quên Df, giữ Dr
```

## Mục Lục

- [1. Mục tiêu](#1-mục-tiêu)
- [2. Cấu trúc workspace](#2-cấu-trúc-workspace)
- [3. Dữ liệu Google Drive](#3-dữ-liệu-google-drive)
- [4. Dataset baseline](#4-dataset-baseline)
- [5. Input và encoder](#5-input-và-encoder)
- [6. Nhãn known/unknown](#6-nhãn-knownunknown)
- [7. Cấu hình chạy hiện tại](#7-cấu-hình-chạy-hiện-tại)
- [8. Notebook chính](#8-notebook-chính)
- [9. Script CLI](#9-script-cli)
- [10. Metric và đánh giá](#10-metric-và-đánh-giá)
- [11. Output artifact](#11-output-artifact)
- [12. Unlearning](#12-unlearning)
- [13. Log và tài liệu phụ trợ](#13-log-và-tài-liệu-phụ-trợ)
- [14. Lưu ý hiệu năng](#14-lưu-ý-hiệu-năng)
- [15. Việc cần làm tiếp](#15-việc-cần-làm-tiếp)

## 1. Mục tiêu

Bài toán hiện tại:

- Input là raw network flow dạng `.pcap`.
- Mỗi file PCAP đang được xem là một sample.
- Parser chuyển mỗi PCAP thành tensor packet features.
- Encoder học vector đặc trưng 256 chiều từ chuỗi packet.
- MLP phân loại binary thành `known` hoặc `unknown`.
- Sau khi classifier ổn định, áp dụng instance-wise unlearning để quên một tập mẫu `Df` nhưng vẫn giữ hiệu năng trên `Dr`.

Đây là bài toán open-world/binary detection theo hướng hiện tại: encoder SupCon được pretrain chỉ trên một số nhãn `known`, sau đó encoder bị freeze; MLP binary mới được train với `known` và các nhãn còn lại được gán `unknown`.

## 2. Cấu trúc workspace

```text
unlearning/
├── README.md
├── paper/
│   ├── Instance-Wise Unlearning.pdf
│   ├── Instance-Wise Unlearning.md
│   ├── Chundawat et al. - 2023 - Zero-Shot Machine Unlearning.pdf
│   └── zero retrain.pdf
├── scripts/
│   ├── colab_sync_project.sh
│   └── colab_run_notebook.sh
└── MLP-Classfication/
    ├── plan.md
    ├── results.md
    ├── requirements.txt
    ├── train_pipeline.py
    └── code/
        ├── CHI.md
        ├── RUN_LOG.md
        ├── export_notebook_outputs.py
        ├── supcon-model.ipynb
        └── train_pipeline.ipynb
```

Ý nghĩa chính:

- `paper/`: tài liệu nền về unlearning.
- `paper/Instance-Wise Unlearning.md`: bản viết lại ngắn gọn bằng tiếng Việt từ paper chính.
- `MLP-Classfication/plan.md`: mô tả bài toán, giả định, pipeline, known/unknown, Df/Dr và hướng đánh giá.
- `MLP-Classfication/results.md`: báo cáo kết quả chạy hiện tại, gồm môi trường, dataset, cấu hình, metric validation/test, nhận xét và kết luận.
- `MLP-Classfication/code/train_pipeline.ipynb`: notebook chính, chạy được trong VS Code/local hoặc Google Colab.
- `MLP-Classfication/code/CHI.md`: danh sách overview của từng code cell; cần cập nhật khi thêm/xóa/sửa cell.
- `MLP-Classfication/train_pipeline.py`: bản CLI tương ứng với notebook.
- `MLP-Classfication/code/export_notebook_outputs.py`: xuất output của notebook sang Markdown để AI/người đọc nhanh.
- `MLP-Classfication/code/RUN_LOG.md`: log notebook đã export gần nhất.
- `MLP-Classfication/code/supcon-model.ipynb`: notebook cũ để tham khảo hướng SupCon/embedding.
- `scripts/colab_sync_project.sh`: đóng gói project local, upload lên Colab VM và giải nén thành folder remote.
- `scripts/colab_run_notebook.sh`: chạy notebook chính trong folder remote trên Colab bằng Colab CLI.

## 3. Dữ liệu Google Drive

Folder dữ liệu gốc:

```text
https://drive.google.com/drive/folders/10uCO2H7CPPS76wfxFnJWKtKM6HY9fGhJ
```

Trong Colab, folder này đã được mount qua Google Drive shortcut. Tên folder chính xác từng gặp trong log là:

```text
/content/drive/MyDrive/Traffic FingerPrinting 
```

Lưu ý có dấu cách ở cuối tên folder. Notebook hiện thử nhiều biến thể tên để giảm lỗi:

```text
Traffic FingerPrinting 
Traffic FingerPrinting
Tracffic FingerPrinting
```

Cấu trúc Drive đã ghi nhận:

```text
Traffic FingerPrinting /
├── Các bài báo/
├── Data/
├── keyword/
├── open-world results/
└── tor.zip
```

Các folder dữ liệu trong `Data/`:

```text
273 (200samples key)/
273 (50 Iphone)/
273 (key HPL 300 samples)/
273 (lan 1)/
273 (lan 2)/
AOL (lan 1)/
AOL (lan 2)/
Data iPad/
Data iPhone/
Homepage - subpage/
SupconNet/
preprocessed/
```

## 4. Dataset baseline

Dataset baseline hiện tại:

```text
/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)
```

Lý do chọn:

- Truy cập được từ Colab.
- Cấu trúc rõ: mỗi folder con là một nhãn.
- Có `9005` file PCAP.
- Có `45` nhãn gốc.
- Mỗi nhãn có ít nhất khoảng `100` sample, phù hợp để split theo class.

Đánh giá 5 dataset đầu đã kiểm tra:

| Dataset | Trạng thái | Cấu trúc | Khuyến nghị |
| --- | --- | --- | --- |
| `273 (200samples key)` | Dùng được, `9005` PCAP | `<label>/*.pcap` | Dataset baseline chính. |
| `273 (lan 1)` | Dùng được | `<label>/*.pcap` | Mở rộng sau baseline để kiểm tra ổn định cùng họ dữ liệu. |
| `AOL (lan 1)` | Dùng được | `<label>/*.pcap` | Dùng sau để test open-world/generalization. |
| `Data iPad` | Dùng được | Có tầng `Data iPad new/<label>/*.pcap` | Dùng sau để test cross-device/domain shift. |
| `Data iPhone` | Dùng được | Có tầng `Data iPhone new/<label>/*.pcap` | Dùng sau để test cross-device/domain shift. |

Hiện tại chưa nên trộn nhiều dataset ngay từ đầu. Nên ổn định pipeline trên `273 (200samples key)` trước, sau đó mới thêm `273 (lan 1)`, `AOL`, `iPad`, `iPhone` để kiểm tra generalization/domain shift.

## 5. Input và encoder

Input gốc là PCAP. Parser đọc packet IPv4/IPv6 bằng Scapy và lấy các thông tin:

- timestamp tương đối;
- hướng packet;
- packet size.

Mỗi packet được chuyển thành 3 đặc trưng theo hướng `RawPacketEncoder` trong `supcon-model.ipynb`:

```text
log(1 + relative_time)
direction
log(1 + packet_size) / log(1 + 65535)
```

Mỗi PCAP được cắt hoặc padding về `MAX_PACKETS = 256`, tạo:

```text
features: [256, 3]
mask    : [256]
label   : 0 hoặc 1
```

Encoder hiện tại là CNN 1D kiểu DF-style, bám theo `RawPacketEncoder` trong `supcon-model.ipynb` nhưng giữ output embedding 256 chiều để đúng pipeline MLP/unlearning:

```text
Conv1d(3 -> 32)
Conv1d(32 -> 32)
MaxPool
Dropout

Conv1d(32 -> 64)
Conv1d(64 -> 64)
MaxPool
Dropout

Conv1d(64 -> 128)
Conv1d(128 -> 128)
MaxPool
Dropout

Conv1d(128 -> 256)
Conv1d(256 -> 256)
MaxPool
Dropout

Flatten
Linear(flat_dim -> 256)
LayerNorm(256)
```

Ý nghĩa: encoder học pattern theo chuỗi packet, sau đó nén flow thành embedding 256 chiều.

Classifier hiện tại là MLP 5 hidden layers:

```text
256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 2
```

Output gồm 2 logits:

```text
0 = known
1 = unknown
```

## 6. Nhãn known/unknown

Dataset có nhãn gốc từ tên folder. Notebook không gán `known/unknown` ngẫu nhiên theo từng sample. Việc random được thực hiện ở cấp nhãn/class.

Quy trình hiện tại:

1. Quét toàn bộ nhãn gốc trong dataset.
2. Lọc các nhãn có ít nhất `MIN_SAMPLES_PER_LABEL`.
3. Random theo seed cố định.
4. Chia nhãn vào 3 nhóm:

```text
KNOWN_LABELS
UNKNOWN_LABELS
HOLDOUT_UNKNOWN_LABELS
```

Ý nghĩa:

- `KNOWN_LABELS`: nhãn được xem là known.
- `UNKNOWN_LABELS`: nhãn unknown dùng để train/validation/test binary classifier sau khi encoder đã freeze.
- `HOLDOUT_UNKNOWN_LABELS`: nhãn unknown không xuất hiện trong train classifier, chỉ dùng khi muốn test open-world nghiêm ngặt hơn. Mặc định hiện tại đặt `0` để dùng nhiều unknown cho binary classifier.

File split được lưu để tái lập:

```text
label_split_seed<seed>_labels<size>_known<known_count>_holdout<holdout_count>_mode<encoder_mode>.json
```

Với cấu hình hiện tại, notebook sẽ tạo split mới dạng:

```text
label_split_seed42_labelsall_known10_holdout0_modesupcon.json
```

Nếu file này đã tồn tại, notebook đọc lại thay vì random lại.

## 7. Cấu hình chạy hiện tại

Cấu hình hiện tại nhắm tới Colab free GPU:

```text
MAX_PACKETS = 256
PACKET_FEATURES = 3
ENCODER_ARCHITECTURE = 'df_style_raw_packet_encoder'
ENCODER_MODE = 'supcon'
EMBEDDING_DIM = 256
BATCH_SIZE = 128
EPOCHS = 16
LEARNING_RATE = 2e-4
DROPOUT = 0.20
HIDDEN_DIMS = [512, 256, 128, 64, 32]
```

Cấu hình dữ liệu:

```text
AUTO_LABEL_SPLIT = True
MIN_SAMPLES_PER_LABEL = 100
MAX_LABELS_FOR_EXPERIMENT = None
KNOWN_LABEL_COUNT = 10
HOLDOUT_UNKNOWN_LABEL_COUNT = 0
MAX_FILES_PER_LABEL = 200
HOLDOUT_VALIDATION_RATIO = 0.25
```

Cấu hình SupCon:

```text
SUPCON_EPOCHS = 8
SUPCON_LEARNING_RATE = 1e-3
SUPCON_TEMPERATURE = 0.10
SUPCON_PROJECTION_DIM = 128
FREEZE_ENCODER_AFTER_SUPCON = True
SAVE_SUPCON_ENCODER_CHECKPOINTS = True
SUPCON_CHECKPOINT_EVERY_N_BATCHES = LOG_EVERY_N_BATCHES
SAVE_BINARY_TRAINING_CHECKPOINTS = True
BINARY_TRAINING_CHECKPOINT_EVERY_N_BATCHES = LOG_EVERY_N_BATCHES
```

Hai mode encoder là hai lựa chọn thay thế nhau, không chồng hai encoder:

```text
ENCODER_MODE = 'cross_entropy'
PCAP -> encoder -> embedding 256 -> MLP -> CrossEntropyLoss
Train: cập nhật encoder + MLP bằng loss phân loại.

ENCODER_MODE = 'supcon'
PCAP known-only -> encoder -> embedding 256 -> projection 128 -> SupConLoss
Train encoder: chỉ dùng known key.
Sau SupCon: bỏ projection head, freeze encoder, train MLP binary known/unknown trên embedding 256.
```

`SupConPacketNet` trong notebook chỉ là wrapper tạm thời để gắn projection head khi train SupCon. Nó không phải encoder thứ hai nối tiếp encoder chính.

Trong mode `supcon`, encoder được lưu riêng trong lúc pretrain để tránh mất trọng số nếu interrupt cell train:

```text
supcon_encoder_latest.pt
supcon_encoder_final.pt
```

`supcon_encoder_latest.pt` được ghi đè định kỳ trong quá trình train và cũng được lưu lại nếu bắt được `KeyboardInterrupt`. `supcon_encoder_final.pt` chỉ có khi SupCon pretrain chạy xong. Checkpoint này chứa `encoder_state_dict`; `projector_state_dict` chỉ dùng để kiểm tra/resume SupCon, còn MLP binary dùng embedding từ encoder.

Phần train MLP binary trong cùng cell cũng lưu checkpoint riêng:

```text
binary_training_latest.pt
binary_training_best.pt
```

`binary_training_latest.pt` là trạng thái mới nhất của model encoder + classifier trong lúc train. `binary_training_best.pt` là best state theo validation metric hiện chọn, mặc định là `balanced_accuracy`. File `best_model.pt` ở cell test vẫn là checkpoint cuối cùng sau khi đã sweep threshold và evaluate test.

Ý nghĩa training hiện tại:

```text
Stage 1: SupCon pretrain encoder chỉ trên 10 known key.
Stage 2: Freeze encoder, giữ BatchNorm ở eval mode để encoder không cập nhật bằng unknown.
Stage 3: Train MLP binary known/unknown trên embedding từ encoder frozen.
```

Phân biệt train encoder và dùng encoder:

```text
Train encoder:
input -> encoder -> loss -> backprop -> cập nhật trọng số encoder

Dùng encoder:
input -> encoder đã train -> embedding 256
không backprop, không cập nhật trọng số
```

Cấu hình metric/evaluate:

```text
USE_CLASS_WEIGHTS = True
CHECKPOINT_SCORE_METRIC = 'balanced_accuracy'
UNKNOWN_THRESHOLD = 0.50
THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
USE_THRESHOLD_SWEEP = True
SAVE_EMBEDDINGS_AFTER_TRAIN = False
```

Cache packet feature được tách theo `ENCODER_ARCHITECTURE`, `PACKET_FEATURES` và `MAX_PACKETS`, nên khi đổi từ feature 5 chiều sang 3 chiều notebook sẽ tự tạo cache mới thay vì đọc nhầm cache cũ.

Cấu hình device:

```text
DEVICE_NAME = ''
REQUIRE_CUDA = True
```

`DEVICE_NAME = ''` nghĩa là tự chọn `cuda` nếu runtime thật sự có CUDA. Với cấu hình hiện tại dùng toàn bộ `45` nhãn, tối đa `200` PCAP/nhãn, SupCon pretrain và encoder DF-style, `REQUIRE_CUDA=True` để notebook dừng sớm nếu runtime chỉ có CPU. Log GPU đúng cần có dạng:

```text
PyTorch version: 2.11.0+cu128
torch_cuda_compiled=True
torch_cuda_available=True
CUDA device: Tesla T4
Device đang dùng: cuda
```

## 8. Notebook chính

Notebook chính:

```text
MLP-Classfication/code/train_pipeline.ipynb
```

Notebook hiện hỗ trợ hai kiểu runtime:

- VS Code/local filesystem: ưu tiên cho workflow hiện tại. Cấu hình path dữ liệu bằng biến môi trường `DATA_DIR`/`DATA_DIRS`, hoặc sửa `LOCAL_DATA_DIR_LIST` trong cell cấu hình.
- Google Colab + Google Drive: chỉ dùng khi mở notebook trong Colab thật sự; cell cấu hình mới mount Drive trong trường hợp này.

Thứ tự chạy:

1. Cài dependency runtime.
2. Xác định runtime và dataset path.
3. Import thư viện, chọn device.
4. Định nghĩa parser PCAP, Dataset, encoder, MLP, train/evaluate/unlearning.
5. Quét label inventory.
6. Random/read label split.
7. Tạo `FlowRecord`.
8. Split train/validation/test và tạo DataLoader.
9. Khởi tạo model.
10. Train encoder + MLP.
11. Sweep threshold trên validation.
12. Evaluate test với threshold tốt nhất.
13. Smoke test 5 PCAP.
14. Chạy unlearning nếu `RUN_UNLEARNING = True`.
15. Xuất `run_summary.json` và `run_summary.md`.

Mỗi code cell trong notebook phải bắt đầu bằng:

```python
# OVERVIEW: ...
```

Khi thêm/xóa/sửa cell, cần cập nhật:

```text
MLP-Classfication/code/CHI.md
```

## 9. Script CLI

Script CLI:

```text
MLP-Classfication/train_pipeline.py
```

Cài dependency:

```bash
pip install -r MLP-Classfication/requirements.txt
```

Ví dụ train ngoài notebook:

```bash
python MLP-Classfication/train_pipeline.py train \
  --input-dir "/path/to/Data/273 (200samples key)" \
  --known-labels facebook,google \
  --unknown-labels tiktok,youtube \
  --holdout-unknown-labels twitter \
  --output-dir ./artifacts
```

CLI vẫn dùng pipeline CrossEntropy/DF-style gọn để chạy ngoài notebook. SupCon known-only hiện được triển khai trong notebook chính.

```text
batch_size = 128
epochs = 16
learning_rate = 2e-4
log_every_n_batches = 2
holdout_validation_ratio = 0.25
unknown_threshold = 0.50
checkpoint_score_metric = balanced_accuracy
threshold_sweep = enabled by default
```

Notebook vẫn là source chính cho thực nghiệm hiện tại. CLI dùng khi cần chạy không qua Jupyter.

### Chạy bằng Colab CLI

Colab CLI không mount trực tiếp `cwd` local như một remote filesystem. Workflow đúng là:

```text
local project
  -> zip project
  -> colab upload
  -> extract to /content/unlearning
  -> run notebook/script inside Colab VM
```

Session mặc định:

```text
SESSION=unlearning
REMOTE_ROOT=/content/unlearning
GPU=T4
```

Tạo/kiểm tra session:

```bash
colab sessions
colab new -s unlearning --gpu T4
colab status -s unlearning
```

Sync project local hiện tại lên Colab:

```bash
scripts/colab_sync_project.sh
```

Script này sẽ:

- tạo session `unlearning` nếu chưa có;
- zip project local, bỏ qua `.git`, cache, checkpoint, PCAP local;
- upload archive lên `/content/unlearning_project.zip`;
- giải nén thành `/content/unlearning`.

Dataset vẫn nằm trong Google Drive, nên Drive phải được mount trong session Colab trước khi chạy notebook. Nếu `colab drivemount` bị `Access blocked`, mở session bằng browser:

```bash
colab url -s unlearning
```

Sau đó chạy trong Colab browser:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Kiểm tra dataset:

```python
from pathlib import Path

data_dir = Path('/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)')
print(data_dir.exists())
print(len(list(data_dir.rglob('*.pcap'))))
```

Khi Drive đã mount, chạy notebook chính bằng CLI:

```bash
scripts/colab_run_notebook.sh
```

Nếu muốn vừa sync vừa chạy:

```bash
SYNC_FIRST=1 scripts/colab_run_notebook.sh
```

Các biến môi trường có thể chỉnh:

```bash
SESSION=unlearning
REMOTE_ROOT=/content/unlearning
DATA_DIR="/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)"
TIMEOUT=7200
```

Ví dụ:

```bash
DATA_DIR="/content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)" \
TIMEOUT=10800 \
SYNC_FIRST=1 \
scripts/colab_run_notebook.sh
```

Output remote sau khi chạy:

```text
/content/unlearning/MLP-Classfication/code/train_pipeline.executed.ipynb
/content/unlearning/MLP-Classfication/code/RUN_LOG.md
```

Muốn lấy file từ Colab về local, dùng:

```bash
colab download -s unlearning \
  /content/unlearning/MLP-Classfication/code/RUN_LOG.md \
  MLP-Classfication/code/RUN_LOG.md
```

## 10. Metric và đánh giá

Accuracy thuần không đủ cho bài toán này vì dữ liệu có thể lệch giữa `known` và `unknown`.

Metric chính hiện tại:

```text
known_recall
unknown_recall
balanced_accuracy = (known_recall + unknown_recall) / 2
unknown_precision
confusion_matrix
```

Confusion matrix có dạng:

```text
[[known -> known,   known -> unknown],
 [unknown -> known, unknown -> unknown]]
```

Checkpoint được chọn theo:

```text
balanced_accuracy
```

Sau train, notebook sweep threshold trên validation:

```text
UNKNOWN_THRESHOLD candidates = 0.20 -> 0.80
```

Threshold tốt nhất được lưu thành:

```text
BEST_UNKNOWN_THRESHOLD
```

Cell test và smoke test dùng threshold này thay vì argmax mặc định. Mục tiêu là giảm tình trạng model đoán quá nhiều `known` và bỏ sót `unknown`.

Log cũ trước khi sửa threshold/class balance cho thấy vấn đề:

```text
test accuracy     = 0.3320
known_recall      = 0.9259
unknown_recall    = 0.1632
balanced_accuracy = 0.5445
confusion_matrix  = [[150, 12], [477, 93]]
```

Nhận xét: model nhận diện `known` tốt hơn nhiều so với `unknown`; cần ưu tiên cải thiện `unknown_recall` và `balanced_accuracy`, không chỉ nhìn accuracy.

Kết quả compact baseline ngày `2026-09-13` với Colab GPU, `24` nhãn và tối đa `160` PCAP/nhãn:

```text
device                  = cuda, Tesla T4
samples sau cap          = 3840
train                   = 2128  known=1232, unknown=896
validation              = 656   known=264,  unknown=392
test                    = 1056  known=264,  unknown=792
best val balanced_acc   = 0.6529
best unknown threshold  = 0.45
test accuracy           = 0.5691
test known_recall       = 0.6932
test unknown_recall     = 0.5278
test unknown_precision  = 0.8377
test balanced_accuracy  = 0.6105
test confusion_matrix   = [[183, 81], [374, 418]]
```

So với log cũ, `unknown_recall` tăng từ `0.1632` lên `0.5278`, accuracy tăng từ `0.3320` lên `0.5691`, và balanced accuracy tăng từ `0.5445` lên `0.6105`. Đây là baseline tham chiếu đã export vào `RUN_LOG.md`, nhưng vẫn chưa đủ mạnh để dùng làm mốc unlearning cuối vì gần một nửa unknown test vẫn bị đoán thành known.

Kết quả trung gian sau khi đổi sang DF-style encoder 3 feature nhưng vẫn dùng `24` nhãn và tối đa `160` PCAP/nhãn:

```text
total parameters        = 1,417,570
best val balanced_acc   = 0.5875
best unknown threshold  = 0.35
test accuracy           = 0.6364
test known_recall       = 0.4848
test unknown_recall     = 0.6869
test unknown_precision  = 0.8000
test balanced_accuracy  = 0.5859
test confusion_matrix   = [[128, 136], [248, 544]]
epoch 1 time            = 2435.6s
```

Diễn giải: accuracy và unknown recall tăng, nhưng balanced accuracy giảm vì known recall tụt mạnh. Vì yêu cầu hiện tại là encoder không học từ unknown, notebook đã đổi sang SupCon known-only: dùng toàn bộ `45` nhãn, chọn `10` known key, `35` key còn lại làm unknown cho binary classifier, pretrain encoder bằng SupCon rồi freeze trước khi train MLP.

Chi tiết kết quả được lưu ở:

```text
MLP-Classfication/results.md
```

Lưu ý: kết quả DF-style ở trên là kết quả trung gian từ output notebook, chưa được export thành `RUN_LOG.md`. Sau lần chạy cấu hình SupCon known-only hiện tại, cần export lại log và cập nhật `results.md`.

## 11. Output artifact

Output Colab mặc định:

```text
/content/drive/MyDrive/unlearning-artifacts/notebook
```

Các file quan trọng:

```text
best_model.pt
label_inventory.json
label_split_seed<seed>_labels<size>_k<known>_u<unknown>.json
metrics.json
training_history.json
run_summary.json
run_summary.md
```

Nếu bật `SAVE_EMBEDDINGS_AFTER_TRAIN = True`, có thêm:

```text
embeddings.pt
```

Lưu ý: lưu embedding toàn bộ có thể lâu vì phải forward lại nhiều PCAP. Hiện mặc định tắt để cell test nhanh hơn.

## 12. Unlearning

Unlearning hiện tại là instance-wise theo hướng:

- `Df`: tập instance cần quên.
- `Dr`: phần còn lại cần giữ.
- Với `Df`, target có thể bị flip sang class đối nghịch để làm model mất khả năng nhận diện đúng mẫu cần quên.
- Với `Dr`, dùng retain loss và regularization để giữ model gần tham số gốc.

Notebook có cell unlearning nhưng mặc định:

```text
RUN_UNLEARNING = False
```

Chỉ bật sau khi classifier open-world đủ ổn định. Nếu classifier nền còn kém, metric unlearning sẽ khó diễn giải.

## 13. Log và tài liệu phụ trợ

File tổng hợp overview cell:

```text
MLP-Classfication/code/CHI.md
```

File export log notebook:

```text
MLP-Classfication/code/RUN_LOG.md
```

File kết quả đã diễn giải:

```text
MLP-Classfication/results.md
```

Script export log:

```bash
python3 MLP-Classfication/code/export_notebook_outputs.py \
  MLP-Classfication/code/train_pipeline.ipynb \
  MLP-Classfication/code/RUN_LOG.md
```

`RUN_LOG.md` phù hợp để gửi lại cho AI đọc nhanh vì chỉ chứa output đã chạy. `run_summary.json` và `run_summary.md` phù hợp để so sánh nhiều lần chạy trên Drive.

## 14. Lưu ý hiệu năng

Các điểm dễ gây chậm:

- Lần đầu parse PCAP sẽ chậm vì cache packet tensor chưa có.
- Nếu chạy CPU, train/test có thể rất lâu.
- Test full dataset và lưu embedding toàn bộ có thể lâu dù train đã xong.
- Nếu interrupt từ VS Code không dừng, có thể runtime Colab vẫn đang chạy ngầm; nên stop/restart runtime từ Colab.

Với Colab free GPU, cấu hình hiện tại dùng toàn bộ `45` nhãn, chọn `10` known key, `35` key còn lại làm unknown cho binary classifier, tối đa `200` PCAP/nhãn, giữ `batch_size = 128`, pretrain SupCon `8` epoch rồi freeze encoder.

Nếu cần quick debug:

```text
MAX_LABELS_FOR_EXPERIMENT = 15
KNOWN_LABEL_COUNT = 5
HOLDOUT_UNKNOWN_LABEL_COUNT = 0
MAX_FILES_PER_LABEL = 50
EPOCHS = 3
BATCH_SIZE = 64
MAX_TRAIN_BATCHES_PER_EPOCH = 8
MAX_EVAL_BATCHES = 5
MAX_TEST_BATCHES = 5
SAVE_EMBEDDINGS_AFTER_TRAIN = False
```

Nếu cần baseline mạnh hơn:

```text
MAX_LABELS_FOR_EXPERIMENT = None
MAX_FILES_PER_LABEL = None
```

Nhưng cấu hình full dùng toàn bộ `45` nhãn và `9005` PCAP có thể tốn thời gian đáng kể.

## 15. Việc cần làm tiếp

1. Chạy lại notebook với cấu hình hiện tại: toàn bộ `45` nhãn, `10` known key, `35` unknown key, SupCon `8` epoch và binary MLP `16` epoch.
2. Export output notebook sang `MLP-Classfication/code/RUN_LOG.md` sau khi chạy xong.
3. Cập nhật `MLP-Classfication/results.md` bằng metric mới; giữ run cũ làm baseline tham chiếu.
4. Chạy thêm nhiều seed label split để kiểm tra metric có ổn định không.
5. Nếu runtime cho phép, thử toàn bộ `45` nhãn.
6. Thêm feature packet/flow như inter-arrival time, signed packet size, protocol và TCP flags.
7. Nếu cần đánh giá open-world nghiêm ngặt, tách riêng `unknown-calibration` và `unknown-final-test`.
8. Khi `balanced_accuracy` và `unknown_recall` ổn định hơn, tạo `Df` ở cấp instance và chạy unlearning.
