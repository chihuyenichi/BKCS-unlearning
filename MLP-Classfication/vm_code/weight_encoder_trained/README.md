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

## Tái sử dụng trong pipeline VM hiện tại

[Notebook instance-wise](../instance_wise/train_instance_wise.ipynb) dùng `FlowEncoder` khôi phục kiến trúc encoder cũ:

```text
input [10000, 3] + packet mask
→ FlowEncoder
→ embedding 256 chiều
→ binary MLP mới
→ known / unknown
```

`load_pretrained_encoder()` kiểm tra `model_config`, lấy riêng các tensor `encoder.*`, bỏ prefix `encoder.` rồi nạp vào `model.encoder` bằng `strict=True`. SupCon projection head `head.*` không được dùng làm classifier binary. Pipeline chính dùng:

```text
MLP-Classfication/vm_code/weight_encoder_trained/pretrain_AOL.pth
→ load_pretrained_encoder(model, checkpoint_path)
→ freeze encoder, train MLP trên AOL known + icloud_100 unknown
→ base_run/base_model/best_model.pt
```

Không nạp toàn bộ SupCon `model_state_dict` trực tiếp vào `FlowModel`: `head.*` và binary `classifier.*` có vai trò khác nhau. `encoder.fc` phụ thuộc `max_packets`, nên checkpoint 10.000 packet không dùng trực tiếp với model input `[256,3]` của pipeline PCAP khác. Không dùng `strict=False` để che lỗi kiến trúc.

`embedding_size=128` trong checkpoint là chiều projection head SupCon, còn feature encoder là 256 chiều. Folder `weight_encoder_trained` chỉ chứa encoder/projection head pretrained; model gốc hoàn chỉnh để so sánh unlearning phải được tạo bằng phase `train-base`.

## Cách dùng an toàn

### Resume pipeline cũ

Chỉ resume khi các yếu tố sau khớp checkpoint:

1. `RawPacketEncoder`/`SupConPacketNet` từ `supcon-model.ipynb`.
2. Input `[10000, 3]`.
3. Cùng định nghĩa ba feature: time, direction, packet size.
4. Cùng preprocessing/normalization và data format.

Khi đó có thể nạp `model_state_dict`; chỉ nạp `optimizer_state_dict` nếu muốn tiếp tục đúng optimizer state cũ. Nạp checkpoint PyTorch chỉ từ nguồn tin cậy.

### Base training và unlearning trên VM

Notebook đã triển khai đường nạp encoder tương thích, nhưng vẫn cần xác nhận preprocessing/provenance và đánh giá holdout trên VM:

1. Xác minh provenance và preprocessing của checkpoint.
2. Giữ input `[10000,3]` với `legacy_raw_v1`, padding mask và cùng semantics của time/direction/packet size.
3. Đánh giá trên validation holdout trước khi dùng cho benchmark/unlearning.
4. Train MLP classifier mới bằng `train-base`, lưu cả encoder + MLP trong `best_model.pt`; dùng checkpoint này làm điểm xuất phát chung cho các nhánh unlearning.

Không coi checkpoint này là checkpoint unlearning hoặc binary classifier đã sẵn sàng để deploy.

Mặc định base training chỉ cập nhật MLP; `--fine-tune-encoder` mở cập nhật encoder. Notebook chưa resume optimizer/epoch cho base training hoặc unlearning; khả năng resume SupCon cũ mô tả ở trên không đồng nghĩa với resume các phase mới.

`pretrain_AOL.pth` đã được theo dõi trong Git, chuyển từ folder cũ sang `weight_encoder_trained/` và được đồng bộ qua clone/pull. `pretrain_273.pth` là checkpoint bổ sung hiện có local, chưa được theo dõi; các file `.pth` mới bị ignore theo cấu hình Git. Pipeline mặc định chỉ cần AOL. Sau clone/pull cần xác nhận checkpoint trên VM; notebook không tự tải checkpoint thay thế. Không suy ra encoder chưa từng thấy `Df` hoặc test chỉ từ tên file: cần provenance của pretraining để đánh giá điều đó.

## Kiểm tra integrity

SHA-256 tại thời điểm tạo README:

```text
655d854ffa318669b36c76517e3ac21313488252d8722093a0209fffbb9a0660  pretrain_273.pth
84bcd55ffcab5b44d09d04131f9a7e81e593565b035cd969d5a73b5671b1cd7a  pretrain_AOL.pth
```
