# Pretrained SupCon checkpoints

Folder này chứa các checkpoint PyTorch đã train trước cho representation learning. Chúng là checkpoint của kiến trúc SupCon cũ trong `MLP-Classfication/code/supcon-model.ipynb`, không phải model phân loại binary/multiclass hoàn chỉnh.

| File | Epoch đã lưu | Cấu hình lưu trong checkpoint | Diễn giải theo tên file |
|---|---:|---|---|
| `pretrain_AOL.pth` | 300 | `max_packets=10000`, `hidden_size=256`, `embedding_size=128` | Pretrain được đặt tên AOL. |
| `pretrain_273.pth` | 500 | `max_packets=10000`, `hidden_size=256`, `embedding_size=128` | Pretrain được đặt tên 273. |

> Tên file chỉ là dấu hiệu về nguồn train. Trước khi dùng cho thí nghiệm chính thức, cần xác minh nguồn PCAP/CSV, label split, feature transform và preprocessing thực tế đã dùng để tạo checkpoint.

## Nội dung checkpoint

Mỗi `.pth` lưu một dictionary PyTorch với các khóa:

```text
model_state_dict
optimizer_state_dict
epoch
model_config
```

`model_state_dict` gồm:

```text
encoder.*  # CNN RawPacketEncoder
head.*     # SupCon projection head
```

Không có `MLPClassifier` cho output `known/unknown` hoặc original multiclass labels. Sau khi dùng encoder, vẫn phải train classifier mới trên embedding.

## Kiến trúc tương thích

Checkpoint khớp với `RawPacketEncoder` và `SupConPacketNet` trong `MLP-Classfication/code/supcon-model.ipynb`:

```text
input [10000, 3]
→ RawPacketEncoder
→ feature 256 chiều
→ SupCon head 256 → 256 → 128
```

Các tensor có tên như `encoder.batch_norm1.*`, `encoder.max_pool_1.*` và `head.*`, tương ứng với notebook này.

## Không tương thích trực tiếp với VM pipeline hiện tại

`MLP-Classfication/vm_code/train_pipeline_vm.ipynb` đang dùng:

```text
input [256, 3]
→ FlowEncoder
→ embedding 256 chiều
→ MLPClassifier
```

Do khác độ dài sequence, preprocessing, tên layer và kiến trúc, không nạp trực tiếp các checkpoint này vào `FlowModel` của VM pipeline:

```python
# Không dùng cách này với train_pipeline_vm.ipynb:
model.load_state_dict(checkpoint['model_state_dict'])
```

Đặc biệt, `encoder.fc` phụ thuộc vào `max_packets`; trọng số train với 10.000 packet không khớp model nhận 256 packet. Không dùng `strict=False` để bỏ qua lỗi mismatch, vì nó có thể bỏ qua các trọng số cần thiết và tạo cảm giác sai rằng đã transfer pretrained model.

## Cách dùng an toàn

### Resume pipeline cũ

Chỉ resume khi các yếu tố sau khớp checkpoint:

1. `RawPacketEncoder`/`SupConPacketNet` từ `supcon-model.ipynb`.
2. Input `[10000, 3]`.
3. Cùng định nghĩa ba feature: time, direction, packet size.
4. Cùng preprocessing/normalization và data format.

Khi đó có thể nạp `model_state_dict`; chỉ nạp `optimizer_state_dict` nếu muốn tiếp tục đúng optimizer state cũ. Nạp checkpoint PyTorch chỉ từ nguồn tin cậy.

### Chuyển sang VM pipeline

Muốn tái sử dụng trong pipeline VM cần một thí nghiệm migration riêng:

1. Xác minh provenance và preprocessing của checkpoint.
2. Khôi phục kiến trúc cũ cùng input 10.000 packet, hoặc thiết kế lại encoder/transfer-learning có kiểm soát.
3. Đánh giá trên validation holdout trước khi dùng cho benchmark/unlearning.
4. Train MLP classifier mới và lưu checkpoint VM theo format `best_model.pt`.

Không coi checkpoint này là checkpoint unlearning hoặc binary classifier đã sẵn sàng để deploy.

## Kiểm tra integrity

SHA-256 tại thời điểm tạo README:

```text
655d854ffa318669b36c76517e3ac21313488252d8722093a0209fffbb9a0660  pretrain_273.pth
84bcd55ffcab5b44d09d04131f9a7e81e593565b035cd969d5a73b5671b1cd7a  pretrain_AOL.pth
```
