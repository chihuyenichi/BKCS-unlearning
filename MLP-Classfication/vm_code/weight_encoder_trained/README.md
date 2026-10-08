# Pretrained SupCon checkpoints

Folder này chứa checkpoint PyTorch pretrained cho representation learning bằng SupCon, gồm CNN encoder và projection head.

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

Kiến trúc trong checkpoint gồm `RawPacketEncoder` và SupCon projection head:

```text
input [10000, 3]
→ RawPacketEncoder
→ feature 256 chiều
→ SupCon head 256 → 256 → 128
```

Các tensor có tên như `encoder.batch_norm1.*` và `head.*`; tên module pooling gồm `max_pool_1` đến `max_pool_4`.

## Tái sử dụng trong pipeline VM hiện tại

[Notebook instance-wise](../instance_wise/train_instance_wise.ipynb) dùng `FlowEncoder` tương thích checkpoint:

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

Chỉ nạp các tensor `encoder.*` vào `model.encoder` bằng `strict=True`. Binary `classifier.*` được khởi tạo và huấn luyện trong phase `train-base`. Kích thước `encoder.fc` phụ thuộc `max_packets`, nên giữ input đúng `[10000,3]` theo checkpoint.

`embedding_size=128` trong checkpoint là chiều projection head SupCon, còn feature encoder là 256 chiều. Folder `weight_encoder_trained` chỉ chứa encoder/projection head pretrained; model gốc hoàn chỉnh để so sánh unlearning phải được tạo bằng phase `train-base`.

## Base training và unlearning trên VM

Notebook đã triển khai đường nạp encoder tương thích, nhưng vẫn cần xác nhận preprocessing/provenance và đánh giá holdout trên VM:

1. Xác minh provenance và preprocessing của checkpoint.
2. Giữ input `[10000,3]` với `legacy_raw_v1`, padding mask và cùng semantics của time/direction/packet size.
3. Đánh giá trên validation holdout trước khi dùng cho benchmark/unlearning.
4. Train MLP classifier mới bằng `train-base`, lưu cả encoder + MLP trong `best_model.pt`; dùng checkpoint này làm điểm xuất phát chung cho các nhánh unlearning.

Không coi checkpoint này là checkpoint unlearning hoặc binary classifier đã sẵn sàng để deploy.

Mặc định base training chỉ cập nhật MLP; `--fine-tune-encoder` mở cập nhật encoder. Notebook chưa resume optimizer/epoch cho base training hoặc unlearning.

`pretrain_AOL.pth` được theo dõi trong Git tại `weight_encoder_trained/` và đồng bộ qua clone/pull. `pretrain_273.pth` là checkpoint bổ sung local, chưa được theo dõi; các file `.pth` chưa được theo dõi bị ignore theo cấu hình Git. Pipeline mặc định chỉ cần AOL. Sau clone/pull cần xác nhận checkpoint trên VM; notebook không tự tải checkpoint thay thế. Cần provenance của pretraining để xác định encoder đã từng thấy `Df` hoặc test hay chưa.

## Kiểm tra integrity

SHA-256 tại thời điểm tạo README:

```text
655d854ffa318669b36c76517e3ac21313488252d8722093a0209fffbb9a0660  pretrain_273.pth
84bcd55ffcab5b44d09d04131f9a7e81e593565b035cd969d5a73b5671b1cd7a  pretrain_AOL.pth
```
