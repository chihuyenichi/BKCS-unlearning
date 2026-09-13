# Kế hoạch xây dựng MLP cho Open-World Classification và Unlearning

Vai trò của file này là mô tả bài toán, phương pháp, giả định dữ liệu, pipeline và kế hoạch đánh giá. Kết quả chạy thực tế được lưu riêng trong `MLP-Classfication/results.md`.

## 1. Bối cảnh bài toán

Bài toán thuộc lĩnh vực phát hiện và phân loại network flow trong mạng máy tính.
Raw network flow được lưu trong các file `.pcap`. Mỗi file có thể chứa một flow hoặc một phiên capture. File PCAP được parser thành chuỗi packet, sau đó encoder biến đổi chuỗi này thành một vector đặc trưng có kích thước 256.

Pipeline tổng quát:

```text
Raw network flow / PCAP
        |
        v
Packet parser + Encoder
        |
        v
Vector đặc trưng x thuộc R^256
        |
        v
MLP classifier
        |
        v
Nhãn dự đoán
```

Các file PCAP đã được tổ chức theo các folder nhãn. MLP sẽ học trên embedding 256 chiều, không học trực tiếp trên raw packet hoặc raw flow.

## 2. Mục tiêu chính

Mục tiêu ban đầu là xây dựng một classifier nhị phân cho kịch bản open-world:

- Input: một vector đặc trưng 256 chiều.
- Output: một trong hai lớp `known` hoặc `unknown`.
- Mục tiêu của classifier: xác định flow thuộc nhóm đã biết hay nằm ngoài nhóm đã biết.

Ký hiệu:

```text
x thuộc R^256       : vector đặc trưng của một flow
y thuộc {0, 1}      : nhãn nhị phân
y = 0               : known
y = 1               : unknown
f_theta(x)          : MLP classifier
```

MLP có thể trả về hai logits, sau đó dùng `softmax` để tính xác suất:

```text
f_theta(x) = [z_known, z_unknown]
p(y | x) = softmax(f_theta(x))
```

## 3. Phân biệt open-world classification và unlearning

Hai khái niệm này có liên quan nhưng không đồng nhất.

### 3.1. Open-world classification

Open-world classification trả lời câu hỏi:

> Flow này thuộc nhóm đã biết hay thuộc nhóm chưa biết đối với hệ thống?

Trong đó:

- `known`: các class đã được hệ thống biết và sử dụng để xây dựng mô hình.
- `unknown`: các class hoặc mẫu nằm ngoài phạm vi đã biết.

Nếu muốn đánh giá đúng khả năng open-world, các class `unknown` trong tập test nên được giữ tách biệt và không để lộ trực tiếp trong quá trình huấn luyện. Nếu dùng toàn bộ mẫu unknown để huấn luyện như một lớp âm thông thường, bài toán sẽ gần với binary classification `known/unknown` hơn là đánh giá khả năng nhận diện unknown chưa từng thấy.

### 3.2. Machine unlearning

Unlearning trả lời câu hỏi:

> Sau khi có yêu cầu xóa một số dữ liệu, làm thế nào để mô hình không còn phụ thuộc vào dữ liệu đó nhưng vẫn giữ hiệu năng trên phần dữ liệu còn lại?

Trong unlearning, cần định nghĩa hai tập:

- `D_f` hoặc `Df`: forget set, các mẫu cụ thể cần quên.
- `D_r` hoặc `Dr`: retain set, các mẫu cần tiếp tục được ghi nhớ.

`Df` không bắt buộc phải trùng với toàn bộ lớp `unknown`. `Df` có thể là một tập nhỏ các vector nằm rải rác trong nhiều class hoặc một nhóm class cụ thể.

## 4. Định nghĩa dữ liệu đề xuất

### 4.1. Dữ liệu đầu vào

Nguồn dữ liệu thực tế đang được dùng là folder Drive đã mount trong Colab:

```text
/content/drive/MyDrive/Traffic FingerPrinting /Data/
```

Tên folder gốc có dấu cách ở cuối: `Traffic FingerPrinting `. Log Colab đã xác nhận folder này nằm trong `MyDrive`, có thư mục con `Data`, và `Data/273 (200samples key)` có `9005` file `.pcap`.

Các folder raw data đã thấy trong `Data`:

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

Với pipeline hiện tại, mỗi mẫu dữ liệu ban đầu là một file PCAP nằm dưới một folder nhãn:

```text
Traffic FingerPrinting /Data/<dataset>/<label>/<sample>.pcap
```

Notebook mặc định đọc đệ quy từ nguồn baseline:

```text
Data/273 (200samples key)
```

Các nguồn còn lại được giữ như ứng viên mở rộng sau baseline:

```text
Data/273 (lan 1)
Data/AOL (lan 1)
Data/Data iPad/Data iPad new
Data/Data iPhone/Data iPhone new
```

Đánh giá nhanh các nguồn đã kiểm tra:

| Dataset | Phù hợp hiện tại | Vai trò đề xuất |
| --- | --- | --- |
| `273 (200samples key)` | Cao | Baseline đầu tiên vì đã xác nhận `9005` PCAP và cấu trúc `<label>/*.pcap`. |
| `273 (lan 1)` | Cao | Mở rộng sau baseline để kiểm tra ổn định cùng họ dữ liệu `273`. |
| `AOL (lan 1)` | Trung bình | Dùng cho open-world/generalization vì label là query phrase, có thể nhiễu hơn. |
| `Data iPad/Data iPad new` | Trung bình | Dùng cho cross-device/domain shift sau khi pipeline ổn. |
| `Data iPhone/Data iPhone new` | Trung bình | Dùng cho cross-device/domain shift sau khi pipeline ổn. |

Mỗi packet hợp lệ được chuyển thành 5 đặc trưng:

```text
[log(1 + relative_time), direction, source_port, destination_port, packet_size]
```

Chuỗi packet được cắt hoặc padding về `max_packets`. Tham số `max_packets` là số packet tối đa của một sample, không phải kích thước embedding.

Sau encoder, mỗi mẫu có dạng:

```text
(x_i, c_i)
```

Trong đó:

- `x_i` là vector embedding 256 chiều được tạo từ file PCAP.
- `c_i` là nhãn gốc của flow, ví dụ loại ứng dụng, giao thức, dịch vụ hoặc loại lưu lượng.

Nếu một file PCAP chứa nhiều flow độc lập, cần tách flow theo 5-tuple trước khi tạo sample:

```text
(src_ip, src_port, dst_ip, dst_port, protocol)
```

### 4.2. Ánh xạ sang bài toán nhị phân

Chọn một tập class đã biết `C_known` và một tập class dùng để đại diện cho unknown `C_unknown`.

```text
c_i thuộc C_known   -> y_i = known
c_i thuộc C_unknown -> y_i = unknown
```

Việc ánh xạ này chỉ phục vụ classifier nhị phân. Nhãn gốc `c_i` vẫn cần được giữ lại để phân tích chi tiết theo từng class và xây dựng các tập unlearning.

Notebook hiện dùng chiến lược autosplit ở cấp class:

1. Quét toàn bộ nhãn gốc trong `DATA_DIR`.
2. Lưu inventory nhãn và số sample mỗi nhãn vào `label_inventory.json`.
3. Lọc các nhãn có đủ sample theo `MIN_SAMPLES_PER_LABEL`.
4. Random các nhãn đủ điều kiện vào ba nhóm `KNOWN_LABELS`, `UNKNOWN_LABELS` và `HOLDOUT_UNKNOWN_LABELS`.
5. Lưu cấu hình split vào `label_split_seed<seed>_labels<size>_k<known>_u<unknown>.json` để tái lập thí nghiệm và tránh đọc nhầm split cũ có số nhãn hoặc ratio khác.

Không random từng PCAP vào known/unknown, vì như vậy cùng một nhãn có thể xuất hiện ở cả train và holdout unknown, gây rò rỉ thông tin trong bài toán open-world.

Cấu hình mặc định:

```text
AUTO_LABEL_SPLIT = True
KNOWN_LABEL_RATIO = 0.45
UNKNOWN_TRAIN_LABEL_RATIO = 0.35
HOLDOUT_UNKNOWN_LABEL_RATIO = 0.20
HOLDOUT_VALIDATION_RATIO = 0.25
MIN_SAMPLES_PER_LABEL = 100
MAX_LABELS_FOR_EXPERIMENT = 24
MAX_FILES_PER_LABEL = 160
```

Nếu cần chạy full baseline, đặt `MAX_LABELS_FOR_EXPERIMENT = None` và `MAX_FILES_PER_LABEL = None`. Với dataset `273 (200samples key)`, autosplit full đã ghi nhận `45` nhãn và `9005` sample, nên chạy end-to-end encoder + MLP trên CPU Colab có thể rất lâu.

Cấu hình chạy thử nhanh hiện tại:

```text
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
MAX_TRAIN_BATCHES_PER_EPOCH = None
MAX_EVAL_BATCHES = None
MAX_TEST_BATCHES = None
LOG_EVERY_N_BATCHES = 2
USE_CLASS_WEIGHTS = True
USE_THRESHOLD_SWEEP = True
UNKNOWN_THRESHOLD = 0.50
CHECKPOINT_SCORE_METRIC = 'balanced_accuracy'
SAVE_EMBEDDINGS_AFTER_TRAIN = False
```

Cấu hình này nhắm tới Colab free GPU: dùng nhiều nhãn và sample hơn bản debug, tăng tỷ lệ `unknown-train`, tăng batch size để tận dụng GPU, nhưng vẫn cap số PCAP mỗi nhãn để không chạy full `9005` PCAP ngay từ đầu. `USE_CLASS_WEIGHTS=True` giúp loss phạt class ít hơn mạnh hơn, giảm xu hướng đoán toàn `known`. `HOLDOUT_VALIDATION_RATIO=0.25` đưa một phần holdout unknown vào validation để calibrate threshold, nhưng các holdout unknown này vẫn không được đưa vào train. `USE_THRESHOLD_SWEEP=True` chọn `BEST_UNKNOWN_THRESHOLD` trên validation trước khi evaluate test. `CHECKPOINT_SCORE_METRIC='balanced_accuracy'` chọn checkpoint theo trung bình recall của `known` và `unknown`, phù hợp hơn accuracy thuần trong open-world. Khi báo cáo kết quả chính thức, cần bỏ giới hạn hoặc ghi rõ cấu hình thí nghiệm. Việc lưu `embeddings.pt` toàn bộ nên xem là bước phân tích riêng, vì nó phải forward lại nhiều PCAP và có thể mất lâu trên Colab CPU/free runtime.

Device mặc định nên để `DEVICE_NAME = ''` để notebook tự chọn `cuda` nếu PyTorch/runtime có CUDA. Nếu đặt cứng `DEVICE_NAME = 'cuda'` nhưng gặp `Torch not compiled with CUDA enabled`, nghĩa là kernel hiện tại là PyTorch CPU-only hoặc Colab chưa bật GPU. Khi đó cần bật `Runtime > Change runtime type > Hardware accelerator > GPU`, restart session và chạy lại từ cell dependency/import; không chỉ chạy lại cell model.

### 4.3. Phân chia theo vai trò

Nên phân chia dữ liệu thành các nhóm sau:

- `D_train_known`: các mẫu known dùng để huấn luyện.
- `D_val_known`: các mẫu known dùng để chọn siêu tham số và ngưỡng quyết định.
- `D_test_known`: các mẫu known dùng để đánh giá khả năng giữ lại tri thức.
- `D_train_unknown`: dữ liệu unknown phụ trợ, nếu cần huấn luyện classifier nhị phân.
- `D_test_unknown`: các class hoặc mẫu unknown dùng để đánh giá open-world.
- `Df`: các mẫu được yêu cầu unlearning.
- `Dr`: phần dữ liệu còn lại cần được bảo toàn sau unlearning.

Lưu ý: một mẫu `unknown` trong bài toán open-world không tự động là một mẫu thuộc `Df`. Chỉ đưa mẫu vào `Df` khi có yêu cầu quên dữ liệu đó.

## 5. Kiến trúc MLP đề xuất

Vì encoder đã chuyển mỗi PCAP thành vector đặc trưng 256 chiều, MLP là lựa chọn phù hợp cho classifier head.

File chính để phát triển implementation và chạy thực nghiệm là `code/train_pipeline.ipynb`. Code được chia thành các cell có comment `# OVERVIEW:` ở đầu mỗi cell. File `code/CHI.md` là index overview của notebook và phải được cập nhật khi thêm, xóa hoặc đổi mục đích cell. File `train_pipeline.py` chỉ được dùng như bản CLI/tái sử dụng khi cần chạy ngoài Jupyter.

Kiến trúc ban đầu có thể gồm 5 đến 7 hidden layers:

```text
Input: 256
 -> Linear
 -> Normalization
 -> ReLU
 -> Dropout
 -> Linear
 -> ReLU
 -> Dropout
 -> Linear
 -> ReLU
 -> Linear
 -> ReLU
 -> Linear
 -> ReLU
 -> Linear
 -> Output: 2 logits
```

Một cấu hình cụ thể để bắt đầu có thể là:

```text
256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 2
```

Trong đó:

- `256`: kích thước vector từ encoder.
- Các lớp ở giữa: hidden layers của MLP.
- `2`: hai logits tương ứng với `known` và `unknown`.

Không nên cố định 5 đến 7 layers là điều kiện bắt buộc ngay từ đầu. Cần so sánh một số cấu hình nhỏ hơn và lớn hơn để kiểm tra overfitting, thời gian huấn luyện và hiệu năng trên unknown.

Các thành phần có thể sử dụng:

- `ReLU` hoặc `GELU` làm hàm kích hoạt.
- `BatchNorm1d` hoặc `LayerNorm` để ổn định huấn luyện.
- `Dropout` để giảm overfitting.
- `CrossEntropyLoss` cho hai logits, hoặc `BCEWithLogitsLoss` nếu chỉ dùng một logit.

Trong giai đoạn đầu, nên cố định encoder và chỉ huấn luyện MLP. Sau khi có baseline ổn định, mới thử fine-tune encoder để biết hiệu năng đến từ embedding hay từ classifier.

## 6. Quy trình thực hiện gồm hai giai đoạn

### Giai đoạn A: Huấn luyện classifier open-world

Mục tiêu là xây dựng classifier nhị phân cơ sở.

Các bước:

1. Dùng encoder để chuyển raw network flow thành vector 256 chiều.
2. Giữ lại nhãn gốc của từng vector.
3. Ánh xạ nhãn gốc sang `known` hoặc `unknown` theo cách chia class đã định nghĩa.
4. Huấn luyện MLP trên tập train.
5. Chọn threshold trên tập validation thay vì mặc định luôn dùng threshold 0.5.
6. Đánh giá trên các class hoặc mẫu unknown chưa được dùng trong huấn luyện.

Loss cơ bản:

```text
L_cls = CrossEntropy(f_theta(x), y)
```

Kết quả của giai đoạn này là mô hình `M_pre`, dùng làm mô hình trước unlearning.

### Giai đoạn B: Instance-wise unlearning

Khi phát sinh yêu cầu xóa, xác định cụ thể tập `Df` và `Dr`:

```text
Df: các vector cần quên
Dr: các vector cần giữ lại
```

Mục tiêu của bước unlearning:

- giảm khả năng mô hình dự đoán đúng hoặc ghi nhớ các mẫu trong `Df`;
- giữ hiệu năng trên `Dr` gần với mô hình trước unlearning;
- hạn chế việc thay đổi toàn bộ mô hình không cần thiết.

Có thể dùng hai cách diễn giải cho output `known/unknown`:

1. Nếu yêu cầu nghiệp vụ là mẫu bị xóa phải bị loại khỏi nhóm đã biết, đặt mục tiêu `Df: known -> unknown`.
2. Nếu bám sát mục tiêu của paper, không bắt buộc mọi mẫu `Df` phải chuyển thành `unknown`; chỉ cần mô hình không còn dự đoán đúng nhãn gốc hoặc không còn giữ thông tin của chúng, trong khi vẫn bảo toàn `Dr`.

Do đó, `known/unknown` là output của classifier, còn unlearning là quy trình cập nhật mô hình sau khi đã huấn luyện. Không nên xem việc gán nhãn `unknown` trong bước huấn luyện ban đầu là đã thực hiện unlearning.

## 7. Có thể dùng Instance-Wise Unlearning không?

Có thể, và đây là hướng phù hợp nếu yêu cầu xóa nhắm tới từng flow hoặc từng vector cụ thể.

Instance-wise unlearning phù hợp khi:

- chỉ một số vector cần bị xóa;
- các vector cần xóa có thể nằm ở nhiều class khác nhau;
- không muốn xóa toàn bộ một class;
- cần giữ lại hiệu năng của mô hình trên các vector không bị yêu cầu xóa.

Phương pháp này không nên được dùng để thay thế open-world classifier. Cách kết hợp hợp lý là:

```text
MLP classifier             -> nhận diện known / unknown
        |
        v
Mô hình đã huấn luyện M_pre
        |
        v
Instance-wise unlearning trên Df
        |
        v
Mô hình sau unlearning M_unlearn
```

Tư tưởng từ paper *Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers* có thể áp dụng cho MLP:

- dùng các mẫu trong `Df` để tạo mục tiêu quên;
- dùng `Dr` để bảo toàn kiến thức còn lại;
- có thể ép mô hình dự đoán sai hoặc chuyển sang nhãn khác trên `Df`;
- dùng regularization để giới hạn ảnh hưởng của bước cập nhật.

Hai hướng regularization đáng thử:

- **Adversarial examples**: tạo các biến thể gần với mẫu retain để bảo vệ biên quyết định trên `Dr`.
- **Weight importance**: ước lượng tầm quan trọng của tham số và hạn chế thay đổi các tham số quan trọng đối với dữ liệu retain.

## 8. Loss và chiến lược unlearning sơ bộ

Có thể bắt đầu bằng mục tiêu tổng hợp:

```text
L_total = L_forget + lambda_r * L_retain + lambda_reg * L_reg
```

Trong đó:

- `L_forget`: làm giảm khả năng mô hình ghi nhớ hoặc dự đoán đúng trên `Df`.
- `L_retain`: duy trì khả năng dự đoán đúng trên `Dr`.
- `L_reg`: regularization để giới hạn thay đổi tham số hoặc bảo toàn biên quyết định.
- `lambda_r`, `lambda_reg`: hệ số cân bằng cần chọn bằng validation.

Các baseline nên triển khai theo thứ tự:

1. Huấn luyện lại từ đầu trên `Dr`, dùng làm mốc tham chiếu lý tưởng.
2. Fine-tune thông thường trên `Dr`.
3. Dùng negative gradient hoặc loss làm sai trên `Df`.
4. Thêm regularization bảo toàn `Dr`.
5. Thử weight-importance regularization và adversarial regularization.

## 9. Chỉ số đánh giá

### 9.1. Đánh giá open-world classifier

- Accuracy hoặc balanced accuracy trên `known`.
- Precision, recall và F1-score của lớp `unknown`.
- AUROC hoặc AUPR cho khả năng phát hiện unknown.
- Confusion matrix giữa `known` và `unknown`.
- Đánh giá riêng trên các class unknown chưa xuất hiện trong tập train.

### 9.2. Đánh giá unlearning

- Accuracy trên `Dr` trước và sau unlearning.
- Tỷ lệ quên trên `Df`, ví dụ tỷ lệ mẫu không còn được dự đoán đúng.
- Tỷ lệ mẫu `Df` chuyển từ `known` sang `unknown`, nếu đây là yêu cầu nghiệp vụ.
- Khoảng cách hiệu năng giữa `M_unlearn` và mô hình retrain từ đầu trên `Dr`.
- Khả năng chống membership inference hoặc các kiểm tra cho thấy mẫu `Df` còn bị ghi nhớ.
- Thời gian và chi phí cập nhật so với retraining từ đầu.

Một phương pháp unlearning tốt cần đạt đồng thời:

```text
Forgetting trên Df cao
Retention trên Dr cao
Chi phí thấp hơn retraining
```

### 9.3. Kết quả baseline hiện tại từ RUN_LOG.md

Run mới nhất được export vào `MLP-Classfication/code/RUN_LOG.md` ngày `2026-09-13`. Đây là baseline hiện tại tốt nhất sau khi tăng số nhãn, tăng số mẫu, dùng class weights, chọn checkpoint theo `balanced_accuracy` và sweep threshold unknown trên validation.

Runtime:

```text
PyTorch version      = 2.11.0+cu128
torch_cuda_compiled  = True
torch_cuda_available = True
CUDA device          = Tesla T4
Device đang dùng     = cuda
```

Dataset:

```text
Dataset baseline = Data/273 (200samples key)
Tổng file        = 9005
Tổng nhãn gốc    = 45
Nhãn đủ điều kiện split >= 100 samples = 45
```

Label split hiện tại:

```text
MAX_LABELS_FOR_EXPERIMENT = 24
MAX_FILES_PER_LABEL       = 160
KNOWN_LABELS              = 11 nhãn
UNKNOWN_LABELS train      = 8 nhãn
HOLDOUT_UNKNOWN_LABELS    = 5 nhãn
Tổng sample sau cap       = 3840
```

Các nhãn đã dùng:

```text
known:
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

unknown-train:
  office_depot
  qr_code_generator
  take_off_dead
  taylor_swift
  twitter
  usps
  v_for_vendetta
  xfinity_customer_service

unknown-holdout:
  quest_diagnostics
  ross
  social_security
  takeoff_shooting
  tmz
```

Data split:

```text
train      = 2128  known=1232, unknown=896
validation = 656   known=264,  unknown=392
test       = 1056  known=264,  unknown=792
```

Kết quả train/validation:

```text
epochs           = 12
batch_size       = 128
train_batches    = 17
val_batches      = 6
class_weights    = [0.8421, 1.1579]
best val balanced_accuracy = 0.6529
```

Epoch tốt nhất theo validation:

```text
train_loss         = 0.5291
val_accuracy       = 0.6296
val_known_recall   = 0.7727
val_unknown_recall = 0.5332
val_balanced_acc   = 0.6529
val_confusion      = [[204, 60], [183, 209]]
```

Threshold sweep trên validation chọn:

```text
BEST_UNKNOWN_THRESHOLD = 0.45
validation balanced_accuracy = 0.6583
validation known_recall      = 0.7197
validation unknown_recall    = 0.5969
validation unknown_precision = 0.7597
validation confusion         = [[190, 74], [158, 234]]
```

Kết quả test với threshold `0.45`:

```text
test loss                    = 0.8527
test accuracy                = 0.5691
test known_recall            = 0.6932
test unknown_recall          = 0.5278
test unknown_precision       = 0.8377
test balanced_accuracy       = 0.6105
test known_total             = 264
test unknown_total           = 792
test predicted_unknown_total = 499
test confusion_matrix        = [[183, 81], [374, 418]]
```

Diễn giải:

- Pipeline đã chạy hoàn chỉnh trên GPU Colab và lưu checkpoint.
- So với các log cũ, khả năng phát hiện unknown đã cải thiện rõ.
- `unknown_precision=0.8377` khá tốt: khi model dự đoán unknown thì đa số đúng.
- Điểm yếu chính là `unknown_recall=0.5278`: `374/792` unknown test vẫn bị đoán nhầm thành known.
- `balanced_accuracy=0.6105` vượt random baseline nhưng vẫn chưa đủ mạnh để làm baseline unlearning cuối.
- Epoch 1 mất `902s` và validation đầu mất `306.4s` do parse/cache PCAP lần đầu; từ epoch 2 trở đi chỉ khoảng `5.7s-6.0s/epoch`, chứng tỏ cache packet feature có tác dụng.

So sánh với log cũ:

```text
old test accuracy          = 0.3320
new test accuracy          = 0.5691

old unknown_recall         = 0.1632
new unknown_recall         = 0.5278

old balanced_accuracy      = 0.5445
new balanced_accuracy      = 0.6105
```

Kết luận:

- Run này là baseline tốt nhất hiện tại của project.
- Chưa nên chạy unlearning chính thức cho báo cáo cuối.
- Cần cải thiện classifier open-world trước, đặc biệt là unknown recall và balanced accuracy.

Đề xuất tiếp theo:

1. Giữ run này làm baseline mới.
2. Chạy thêm nhiều seed label split để kiểm tra độ ổn định.
3. Thử tăng `MAX_LABELS_FOR_EXPERIMENT` lên `32` hoặc dùng toàn bộ `45` nhãn nếu runtime cho phép.
4. Thử tăng `MAX_FILES_PER_LABEL` lên `200`.
5. Thêm feature packet/flow: inter-arrival time, signed packet size theo direction, protocol, TCP flags.
6. Tách `unknown-calibration` và `unknown-final-test` rõ hơn nếu cần đánh giá open-world nghiêm ngặt.
7. Chỉ bật instance-wise unlearning sau khi classifier nền ổn định hơn.

Chi tiết kết quả đã được lưu riêng trong `MLP-Classfication/results.md`.

### 9.4. Lịch sử log cũ

Log notebook ngày `2026-09-11` cho thấy pipeline đã chạy được về mặt kỹ thuật nhưng baseline open-world chưa đạt yêu cầu.

Lưu ý: log này là kết quả trước khi thêm autosplit nhãn ở cấp class. Notebook cần được chạy lại từ cell inventory nhãn để có kết quả baseline mới.

Sau khi thêm autosplit, log cho thấy split full dùng toàn bộ `45` nhãn và `9005` sample, tạo `train=5043`, `validation=1080`, `test=2882`. Cell train chạy `85` phút là hợp lý nếu dùng CPU Colab, vì mỗi epoch phải đọc/forward nhiều batch PCAP và trước đó cache packet features chưa đầy đủ. Cell test cũng có thể chạy rất lâu nếu vừa evaluate toàn bộ test vừa lưu embedding cho toàn bộ records. Do đó notebook hiện dùng cấu hình Colab free GPU với `24` nhãn, tối đa `160` PCAP mỗi nhãn, `12` epoch, evaluate toàn bộ split đã cap, sweep threshold trên validation và tắt lưu embedding toàn bộ.

Log quick experiment gần nhất trước khi sửa loss/metric cho thấy `val_acc=0.8108` nhưng `val_unknown_recall=0.0000`, `test_accuracy=0.3409` và `test_unknown_recall=0.0000`. Điều này chứng tỏ model đang lệch mạnh về dự đoán `known`. Notebook hiện đã được cập nhật để tăng số nhãn `unknown-train`, dùng weighted `CrossEntropyLoss`, in confusion matrix và chọn checkpoint theo `balanced_accuracy` thay vì accuracy thuần.

Log Colab free GPU/CPU runtime gần nhất với cấu hình `18` nhãn, tối đa `120` PCAP mỗi nhãn:

```text
records sau cap = 2160
train            = 1176
validation       = 252
test             = 732

train label counts = [756 known, 420 unknown]
class weights      = [0.7143, 1.2857]

best validation balanced_accuracy = 0.6216
test loss                         = 0.6820
test accuracy                     = 0.5246
test known_recall                 = 0.5679
test unknown_recall               = 0.5123
test unknown_precision            = 0.8066
test balanced_accuracy            = 0.5401
test confusion_matrix             = [[92, 70], [278, 292]]
```

Diễn giải log mới:

- So với log cũ `unknown_recall=0.0`, model đã bắt đầu nhận diện được unknown, nhưng `balanced_accuracy=0.5401` vẫn chỉ nhỉnh hơn ngẫu nhiên một ít.
- `loss≈0.68` gần mức phân loại nhị phân chưa tự tin, nên boundary known/unknown còn yếu.
- Confusion matrix cho thấy `278/570` unknown test vẫn bị đoán thành known, tức lỗi lớn nhất vẫn là bỏ sót unknown holdout.
- Epoch đầu mất `~637s` vì nhiều PCAP được parse/cache lần đầu; từ epoch 2 trở đi chỉ khoảng `12-18s/epoch`, chứng tỏ cache packet feature đang có tác dụng.
- Validation dao động mạnh giữa các epoch, có lúc model đoán gần như toàn known hoặc toàn unknown; đây là dấu hiệu mô hình/loss/split chưa ổn định, chưa nên dùng làm baseline cuối cho unlearning.

Điều chỉnh đã áp dụng sau log mới:

1. Giữ `CHECKPOINT_SCORE_METRIC='balanced_accuracy'`, không quay lại accuracy thuần.
2. Tăng dữ liệu lên `MAX_LABELS_FOR_EXPERIMENT = 24` và `MAX_FILES_PER_LABEL = 160`.
3. Tăng `UNKNOWN_TRAIN_LABEL_RATIO = 0.35` và giảm `KNOWN_LABEL_RATIO = 0.45`.
4. Giảm `LEARNING_RATE` từ `1e-3` xuống `5e-4` và tăng `EPOCHS` lên `12`.
5. Thêm `HOLDOUT_VALIDATION_RATIO = 0.25` và sweep threshold để giảm phụ thuộc vào ngưỡng mặc định `0.5`.
6. Nếu metric vẫn thấp sau lần chạy mới, cần xem lại feature raw flow: hiện chỉ dùng 5 feature packet cơ bản, có thể thiếu đặc trưng thống kê flow/timing để tách open-world tốt.

Cấu hình nhãn của log cũ:

```text
C_known         = {english_to_spanish, office_365}
C_unknown_train = {food_near_me}
C_unknown_test  = {twitter}
```

Số mẫu được dùng:

```text
english_to_spanish : 200 known
office_365         : 200 known
food_near_me       : 200 unknown-train
twitter            : 200 unknown-holdout
total              : 800
```

Split của log cũ:

```text
train      = 420
validation = 90
test       = 290
```

Model trong log cũ được khởi tạo đúng kiến trúc `PCAP -> encoder -> embedding 256 -> MLP -> known/unknown`, có `434,402` tham số và chạy trên CPU Colab.

Kết quả:

```text
best validation accuracy = 0.8889
test accuracy            = 0.2862
test unknown recall      = 0.1217
```

Diễn giải:

- Validation accuracy cao nhưng test accuracy thấp, chứng tỏ mô hình chưa tổng quát tốt sang class unknown holdout `twitter`.
- `unknown_recall = 0.1217` là thấp; mô hình đang bỏ sót phần lớn unknown trong test.
- `unknown-train` chỉ có một class `food_near_me`, nên lớp unknown trong train chưa đủ đa dạng để đại diện cho open-world.
- Train log dao động mạnh giữa các epoch, cho thấy mô hình còn chưa ổn định với cấu hình hiện tại.
- Chưa nên dùng model này làm mốc cuối cho unlearning. Cần có baseline classifier đáng tin hơn trước khi đánh giá `Df/Dr`.

Điều chỉnh đã cập nhật trước khi chạy unlearning:

1. Notebook đã bổ sung confusion matrix, `known_recall`, `unknown_recall` và `balanced_accuracy`.
2. Checkpoint mặc định được chọn theo `balanced_accuracy`, không còn chọn theo accuracy thuần.
3. Quick experiment tăng số class lên `24` và dùng tỷ lệ `known=0.45`, `unknown-train=0.35`, `holdout=0.20` để unknown train đa dạng hơn.
4. Train dùng weighted `CrossEntropyLoss` để giảm xu hướng đoán toàn `known`.

Tại thời điểm log cũ, các điểm cần quan sát ở lần chạy kế tiếp là:

1. Nếu `unknown_recall` vẫn gần `0`, cần tăng `MAX_FILES_PER_LABEL` hoặc tăng tiếp tỷ lệ/số nhãn `unknown-train`.
2. Nếu `known_recall` giảm mạnh nhưng `unknown_recall` tăng, cần cân bằng lại class weight hoặc dùng `open_world_score`.
3. Nếu cả hai recall đều thấp, mới nên thử giảm learning rate, giảm độ phức tạp MLP, hoặc so sánh encoder train end-to-end với phương án pretrain/freeze encoder.

## 10. Rủi ro và điểm cần quyết định

### 10.1. Unknown có thật sự là dữ liệu cần quên không?

Đây là quyết định quan trọng nhất.

- Nếu `unknown` là các class mới, chưa từng xuất hiện và chỉ dùng để kiểm tra open-world, đây là bài toán open-world classification.
- Nếu `unknown` là các flow đã từng được mô hình học nhưng nay phải xóa, chúng tạo thành `Df` và phù hợp với instance-wise unlearning.
- Nếu vừa muốn phát hiện class mới vừa muốn xóa một số flow, cần thực hiện hai thí nghiệm riêng để kết quả không bị nhập nhằng.

### 10.2. Có dùng unknown trong lúc huấn luyện không?

Nếu dùng toàn bộ unknown để train, mô hình có thể học ranh giới của các unknown cụ thể nhưng chưa chắc tổng quát được sang unknown mới. Vì vậy nên báo cáo rõ:

- unknown nào được dùng để train;
- unknown nào chỉ dùng để validation/test;
- `Df` được chọn trước hay sau khi train classifier.

### 10.3. Encoder có được cập nhật không?

Nên bắt đầu với encoder cố định. Sau đó mới so sánh:

- chỉ unlearn MLP classifier;
- unlearn cả encoder và MLP;
- fine-tune toàn bộ mô hình sau unlearning.

Điều này giúp xác định thông tin cần quên đang nằm chủ yếu ở embedding hay ở classifier head.

## 11. Kế hoạch triển khai theo thứ tự

1. Kiểm tra kích thước, phân phối và chất lượng của vector 256 chiều.
2. Xác định nhãn gốc và danh sách class thuộc `C_known`, `C_unknown`.
3. Chia dữ liệu thành train, validation và test, bảo đảm không rò rỉ class unknown.
4. Xây dựng MLP baseline với encoder được cố định.
5. Đánh giá classifier nhị phân `known/unknown`.
6. Tạo một hoặc nhiều forget set `Df` ở cấp instance.
7. Xây dựng các phương pháp unlearning baseline.
8. Thêm regularization theo instance-wise unlearning.
9. So sánh retention, forgetting, chi phí tính toán và khả năng chống ghi nhớ.
10. Phân tích riêng kết quả open-world và kết quả unlearning.

## 12. Kết luận định hướng

MLP 5 đến 7 hidden layers phù hợp với input là vector embedding 256 chiều. Mô hình có thể làm classifier nhị phân với output `known/unknown` cho kịch bản open-world.

Instance-wise unlearning có thể được áp dụng ở giai đoạn sau khi classifier đã được huấn luyện, đặc biệt khi `Df` gồm các vector cụ thể cần xóa. Cần giữ rõ hai vai trò:

- `known/unknown`: nhãn đầu ra của bài toán phân loại.
- `Df/Dr`: cách chia dữ liệu phục vụ bài toán unlearning.

Trong trường hợp muốn các mẫu bị quên được dự đoán là `unknown`, có thể đặt mục tiêu `known -> unknown`. Tuy nhiên, kết quả unlearning vẫn phải được đánh giá bằng cả khả năng quên trên `Df` và khả năng bảo toàn hiệu năng trên `Dr`, không chỉ bằng accuracy của classifier nhị phân.
