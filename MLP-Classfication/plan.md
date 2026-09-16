# Kế hoạch MLP Classification và Instance-Wise Unlearning

File này mô tả bài toán, dữ liệu đầu vào, pipeline huấn luyện, checkpoint cần lưu và hướng mở rộng cho unlearning. Tài liệu này là đặc tả độc lập cho implementation, không phụ thuộc vào trạng thái hiện tại của bất kỳ file code nào.

## 1. Bài toán

Bài toán hiện tại không bắt đầu từ một vector embedding có sẵn. Input gốc là các file network capture `.pcap`. Mỗi file `.pcap` được xem là một sample traffic hoặc một flow/session đã được thu thập.

Pipeline chính:

```text
file .pcap
  -> parse packet
  -> extract chuỗi packet feature 3 chiều
  -> encoder CNN/SupCon
  -> embedding 256 chiều
  -> MLP classifier
  -> known / unknown
```

Mục tiêu giai đoạn 1 là xây dựng classifier nhị phân cho traffic fingerprinting/open-world detection:

- `known`: flow thuộc nhóm class đã biết.
- `unknown`: flow thuộc nhóm class khác, cần nhận diện là ngoài nhóm known.

Mục tiêu giai đoạn 2 là instance-wise unlearning, tức sau khi có yêu cầu quên một số sample cụ thể, mô hình phải giảm phụ thuộc vào các sample đó nhưng vẫn giữ hiệu năng trên phần dữ liệu còn lại. Giai đoạn 2 chưa được tiến hành trong thực nghiệm hiện tại.

## 2. Dữ liệu đầu vào

Nguồn dữ liệu chính dự kiến nằm trên Google Drive sau khi mount Colab:

```text
/content/drive/MyDrive/Traffic FingerPrinting /Data/
```

Dataset baseline ưu tiên:

```text
Traffic FingerPrinting /Data/273 (200samples key)/
```

Cấu trúc dữ liệu kỳ vọng:

```text
<dataset>/<label>/<sample>.pcap
```

Trong đó:

- `<label>` là nhãn gốc của sample, ví dụ một website/query/app/service.
- `<sample>.pcap` là raw traffic cần parse.

Các dataset khác như `273 (lan 1)`, `AOL (lan 1)`, `Data iPad`, `Data iPhone` có thể dùng sau khi baseline ổn định để kiểm tra generalization/cross-device/domain shift.

Nếu một file `.pcap` chứa nhiều flow độc lập, về nguyên tắc có thể tách theo 5-tuple:

```text
(src_ip, src_port, dst_ip, dst_port, protocol)
```

Tuy nhiên pipeline hiện tại đang đi theo giả định thực dụng: mỗi file `.pcap` là một sample, nhãn lấy từ folder cha.

## 3. Packet Feature Extraction

Encoder không nhận trực tiếp raw bytes của `.pcap`. Trước tiên pipeline cần parse `.pcap` thành chuỗi packet feature.

Mỗi packet dùng 3 feature ngữ nghĩa:

```text
[relative_time, direction, packet_size]
```

Ý nghĩa:

- `relative_time`: thời gian tương đối của packet so với packet đầu tiên trong file.
- `direction`: hướng packet, ví dụ outgoing/incoming, suy luận từ IP nội bộ hoặc IP xuất hiện thường xuyên.
- `packet_size`: kích thước packet.

Trong implementation, các feature này có thể được biến đổi để train ổn định hơn:

```text
[log(1 + relative_time), direction, normalized_log_packet_size]
```

Điểm quan trọng: dù có normalize/log transform, bản chất input của encoder vẫn là 3 thông tin chính:

```text
time, direction, size
```

Mỗi file `.pcap` sau parse có dạng tensor:

```text
[MAX_PACKETS, 3]
```

Nếu flow ngắn hơn `MAX_PACKETS`, padding thêm zero. Nếu dài hơn, cắt về `MAX_PACKETS`.

## 4. Encoder

Encoder nên dùng kiến trúc DF-style raw packet encoder cho chuỗi packet:

```text
Input:  [batch, MAX_PACKETS, 3]
Output: [batch, 256]
```

Kiến trúc tham khảo:

```text
packet sequence
  -> Conv1d block 3 -> 32
  -> Conv1d block 32 -> 64
  -> Conv1d block 64 -> 128
  -> Conv1d block 128 -> 256
  -> MaxPool/Dropout sau mỗi block
  -> Flatten
  -> Linear -> embedding 256 chiều
```

Encoder output là embedding 256 chiều cho từng `.pcap`.

Cần phân biệt rõ hai khái niệm:

- **Output của encoder cho một sample**: embedding vector 256 chiều.
- **Trọng số đã học của encoder**: các tensor trong `encoder_state_dict`, được lưu trong checkpoint.

Vì vậy, không nên mô tả encoder output trực tiếp là "ma trận trọng số". Nếu cần nói theo trực giác, có thể diễn đạt:

> Encoder học một bộ trọng số để biến chuỗi packet feature thành embedding 256 chiều. Bộ trọng số này được lưu checkpoint để dùng lại, train tiếp hoặc phục vụ unlearning.

Nếu sau này cần phân tích "độ quan trọng của input/feature", đó là một bước phân tích riêng, ví dụ saliency, gradient attribution, feature importance hoặc weight importance.

## 5. SupCon Pretrain Encoder

Giai đoạn encoder chính hiện tại dùng supervised contrastive learning, chỉ trên các class `known`.

Pipeline:

```text
PCAP known-only
  -> extract [relative_time, direction, packet_size]
  -> encoder
  -> embedding 256
  -> projection head 128
  -> SupConLoss
```

Mục tiêu: encoder học representation tốt cho các known class trước khi classifier binary nhìn thấy unknown.

Sau khi SupCon pretrain xong:

1. Lưu checkpoint encoder.
2. Bỏ projection head khi đưa sang classifier.
3. Freeze encoder.
4. Train MLP binary trên embedding frozen.

`SupConPacketNet` chỉ là wrapper tạm thời gồm:

```text
encoder + projection head
```

Nó không phải encoder thứ hai nối tiếp encoder chính.

## 6. Checkpoint Trên Drive

Checkpoint cần ưu tiên lưu trên Google Drive vì runtime chính là Colab/GPU và máy local không đủ mạnh.

Output directory chính trên Drive:

```text
/content/drive/MyDrive/unlearning-artifacts/mlp-classification/
```

Nếu chạy local/VS Code, runtime có thể fallback về:

```text
artifacts/mlp-classification/
```

Nhưng plan chính vẫn ưu tiên Drive.

Các checkpoint cần có:

```text
supcon_encoder_latest.pt
supcon_encoder_final.pt
binary_training_latest.pt
binary_training_best.pt
best_model.pt
```

Ý nghĩa:

- `supcon_encoder_latest.pt`: checkpoint encoder mới nhất trong lúc SupCon đang chạy; được ghi đè định kỳ.
- `supcon_encoder_final.pt`: encoder sau khi SupCon pretrain hoàn tất.
- `binary_training_latest.pt`: model encoder + MLP classifier mới nhất trong lúc train classifier.
- `binary_training_best.pt`: model tốt nhất theo validation metric trong lúc train classifier.
- `best_model.pt`: checkpoint cuối cùng sau khi tune threshold/evaluate test.

Checkpoint encoder nên lưu tối thiểu:

```text
encoder_state_dict
projector_state_dict
optimizer_state_dict
epoch / batch progress
known_label_to_id
model_config
feature_config
label_split_path
```

### 6.1. Có thể train tiếp encoder từ checkpoint không?

Có.

Có hai chế độ:

```text
Warm-start:
  load encoder_state_dict
  train tiếp encoder với optimizer mới
```

Warm-start đơn giản và phù hợp khi muốn dùng trọng số cũ làm điểm khởi tạo tốt hơn cho lần chạy sau.

```text
Resume đúng nghĩa:
  load encoder_state_dict
  load projector_state_dict
  load optimizer_state_dict
  load epoch/batch/scheduler state
  train tiếp gần đúng từ điểm dừng
```

Resume đúng nghĩa phù hợp khi cell bị ngắt giữa chừng hoặc muốn tiếp tục SupCon training mà không mất trạng thái optimizer.

Plan khuyến nghị checkpoint encoder hỗ trợ cả hai. Nếu chưa implement resume đầy đủ, ít nhất phải lưu `encoder_state_dict` để warm-start.

## 7. MLP Classifier

Sau encoder, classifier nhận embedding 256 chiều:

```text
embedding = encoder(packet_features)
logits = MLP(embedding)
```

Output hiện tại là binary:

```text
[logit_known, logit_unknown]
```

Mặc định kiến trúc MLP:

```text
256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 2
```

Tức 5 hidden layers và output 2 lớp.

Các thành phần nên dùng:

- `Linear`
- activation như `ReLU` hoặc `GELU`
- `LayerNorm` hoặc `BatchNorm`
- `Dropout`
- `CrossEntropyLoss`

Classifier hiện tại là binary, nhưng nên viết code để dễ mở rộng multiclass. Cụ thể:

```text
output_dim = 2          # known/unknown hiện tại
output_dim = num_class  # multiclass sau này
```

Khi mở rộng multiclass, phần encoder vẫn có thể giữ nguyên. Chỉ cần thay label mapping, classifier head và metric evaluate.

## 8. Giai Đoạn 1: Classification Pipeline

Giai đoạn 1 là phần đang được triển khai chính.

Pipeline đầy đủ:

```text
1. Mount Google Drive.
2. Chọn DATA_DIR chứa các file .pcap.
3. Quét label inventory từ folder.
4. Split label ở cấp class:
   - KNOWN_LABELS
   - UNKNOWN_LABELS
   - HOLDOUT_UNKNOWN_LABELS nếu cần strict open-world
5. Parse .pcap thành chuỗi packet feature [MAX_PACKETS, 3].
6. Pretrain encoder bằng SupCon chỉ trên known train records.
7. Lưu encoder checkpoint lên Drive.
8. Freeze encoder.
9. Train MLP binary known/unknown trên embedding từ encoder.
10. Lưu checkpoint classifier lên Drive.
11. Sweep threshold unknown trên validation.
12. Evaluate test set.
13. Lưu run_summary.json/md.
```

Cấu hình mặc định đề xuất:

```text
MAX_PACKETS = 256
EMBEDDING_DIM = 256
HIDDEN_DIMS = (512, 256, 128, 64, 32)
DROPOUT = 0.20
BATCH_SIZE = 128
EPOCHS = 16
LEARNING_RATE = 2e-4

ENCODER_MODE = 'supcon'
SUPCON_EPOCHS = 8
SUPCON_LEARNING_RATE = 1e-3
SUPCON_TEMPERATURE = 0.10
SUPCON_PROJECTION_DIM = 128
FREEZE_ENCODER_AFTER_SUPCON = True

USE_CLASS_WEIGHTS = True
CHECKPOINT_SCORE_METRIC = 'balanced_accuracy'
USE_THRESHOLD_SWEEP = True
UNKNOWN_THRESHOLD = 0.50
```

Với cấu hình hiện tại, không nên chạy full trên CPU. Nên dùng Colab GPU hoặc VS Code kernel có CUDA.

## 9. Open-World Và Binary Known/Unknown

Hiện tại bài toán thực thi gần với binary known/unknown classification:

```text
known class   -> y = 0
unknown class -> y = 1
```

Tuy nhiên cần tránh rò rỉ thông tin ở cấp class. Split nên thực hiện theo label/class, không random từng PCAP trước rồi mới gán known/unknown.

Đề xuất:

```text
KNOWN_LABELS: class dùng để pretrain SupCon encoder.
UNKNOWN_LABELS: class dùng để train/validation/test binary classifier.
HOLDOUT_UNKNOWN_LABELS: class unknown không đưa vào train, dùng để đánh giá strict open-world nếu cần.
```

Trong cấu hình hiện tại, có thể dùng:

```text
KNOWN_LABEL_COUNT = 10
HOLDOUT_UNKNOWN_LABEL_COUNT = 0
```

Tức 10 known key để train encoder, các key còn lại làm unknown cho binary classifier. Sau khi pipeline ổn, có thể tăng strictness bằng cách để một số unknown class vào holdout.

## 10. Metrics

Accuracy thuần có thể gây hiểu nhầm nếu dữ liệu lệch class. Metric chính nên là:

```text
balanced_accuracy = 0.5 * (known_recall + unknown_recall)
```

Các metric cần log:

- loss
- accuracy
- known recall
- unknown recall
- unknown precision
- balanced accuracy
- confusion matrix
- best threshold trên validation

Checkpoint tốt nhất nên chọn theo:

```text
CHECKPOINT_SCORE_METRIC = 'balanced_accuracy'
```

## 11. Giai Đoạn 2: Instance-Wise Unlearning

Giai đoạn 2 chưa được tiến hành trong thực nghiệm hiện tại. Plan chỉ định nghĩa mục tiêu và hướng làm.

Unlearning cần hai tập:

```text
Df: forget set, các instance cần quên.
Dr: retain set, các instance cần giữ.
```

`Df` có thể là:

- một hoặc vài file `.pcap` cụ thể;
- một nhóm sample thuộc cùng label;
- một nhóm sample được chọn theo metadata hoặc yêu cầu xóa dữ liệu.

Mục tiêu sau unlearning:

```text
model_after_unlearning không còn phụ thuộc đáng kể vào Df
model_after_unlearning vẫn giữ hiệu năng trên Dr
```

Hướng unlearning dự kiến:

```text
1. Load checkpoint model trước unlearning.
2. Chọn Df và Dr.
3. Ước lượng importance của tham số trên Dr.
4. Fine-tune có kiểm soát:
   - tăng loss hoặc flip target trên Df
   - giữ retain loss trên Dr
   - regularization để tham số quan trọng không lệch quá xa model gốc
5. Evaluate:
   - forget behavior trên Df
   - retain accuracy trên Dr
   - test accuracy tổng thể
```

Ban đầu nên unlearn classifier head trước vì ít rủi ro hơn. Sau khi có baseline, mới thử:

- unlearn MLP classifier;
- unlearn một phần encoder;
- unlearn cả encoder và MLP.

## 12. Artifact Cần Lưu

Các artifact nên lưu trên Drive:

```text
label_inventory.json
label_split_*.json
supcon_encoder_latest.pt
supcon_encoder_final.pt
binary_training_latest.pt
binary_training_best.pt
best_model.pt
run_summary.json
run_summary.md
embeddings.pt              # optional
unlearning_result.json     # sau này
```

`embeddings.pt` là optional vì phải forward lại nhiều PCAP và có thể tốn thời gian.

## 13. Trạng Thái Mong Muốn

Trạng thái mong muốn của pipeline:

- Input chính: file `.pcap`, không phải embedding có sẵn.
- Feature chính: 3 feature packet `[relative_time, direction, packet_size]`, có thể log/normalize khi train.
- Encoder chính: DF-style CNN/SupCon encoder.
- Classifier chính: MLP 5 hidden layers, binary known/unknown.
- Checkpoint chính: lưu trên Google Drive.
- Giai đoạn 1: classification pipeline đang là trọng tâm.
- Giai đoạn 2: unlearning đã có kế hoạch nhưng chưa chạy chính thức.

## 14. Câu Hỏi Cần Chốt Sau

Các điểm có thể cần chốt thêm khi bắt đầu chạy thực nghiệm lớn:

1. Có cần strict holdout unknown class ngay từ baseline không, hay trước mắt giữ `HOLDOUT_UNKNOWN_LABEL_COUNT = 0`?
2. Khi resume SupCon, cần resume đầy đủ optimizer/scheduler hay chỉ warm-start từ encoder weights là đủ?
3. Có cần lưu embedding của toàn bộ train/val/test sau mỗi run không, hay chỉ lưu khi phân tích lỗi?
4. Giai đoạn unlearning đầu tiên sẽ quên theo file `.pcap`, theo label, hay theo một nhóm sample do người dùng chỉ định?
