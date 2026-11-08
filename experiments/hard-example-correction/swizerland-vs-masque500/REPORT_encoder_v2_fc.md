# Báo cáo: encoder v2 + train lớp projection (`fc`)

Nguồn số liệu: `outputs/son_tung_hard_example_benchmark_encoder_v2_fc.xlsx` (kernel `tung-unfreeze-fc-enc2`, T4, 10 seed). Test BAcc là trung bình theo seed; so sánh ghép cặp theo seed trong cùng lần chạy.

## 1. Kết quả tóm tắt

| Mode | Test BAcc | Known recall | Unknown recall | Tổng lỗi test (FP+FN) | So với `uniform_finetune` |
|---|---|---|---|---|---|
| baseline (head `h256`, đóng băng) | 0,9834 | 0,9794 | 0,9874 | 332 | – |
| `uniform_finetune` (đóng băng) | 0,9859 | 0,9841 | 0,9876 | 282,5 | mốc |
| `hard_aware_mkmmd_ramp_l010` (đóng băng) | 0,9848 | 0,9863 | 0,9832 | 304,4 | −0,0011 (2/8, p=0,008) |
| **`fc_uniform_lr2e5`** | **0,9953** | **0,9953** | **0,9953** | **94,5** | **+0,0094 (10/0, p=0,002)** |
| `fc_ramp_l010_lr2e5` | 0,9929 | 0,9934 | 0,9924 | 141,5 | +0,0070 (10/0, p=0,002) |

Mode tốt nhất là **`fc_uniform_lr2e5`**: giảm số lỗi test từ 332 xuống còn 94,5 (−72%) so với baseline, cân bằng giữa hai lớp.

## 2. Định nghĩa

**Bài toán.** Phân loại nhị phân luồng mạng: KNOWN = 0 là `swizerland-extracted`, UNKNOWN = 1 là `masque-500-extracted`. Positive = UNKNOWN. Train 68.000, val 12.000 (tách ngẫu nhiên từ train), test 20.000 (file test gốc, 10.000/lớp).

**Encoder (v2).** CNN kiểu DF (`RawPacketEncoder`): 4 khối tích chập + lớp `fc` chiếu 10.240 → 256 chiều. Huấn luyện bằng SupCon chỉ trên Swizerland (70 dòng/lớp × 500 lớp, 200 epoch, batch 32, lr 0,1). Chưa từng thấy Masque. 1-NN trên 500 lớp trang web = 91,22% (encoder cũ 88,64%). SHA-256 `2cfaa8b1…`.

**Head (`BinaryMLP`).** Linear 256→256 → BatchNorm → ReLU → Dropout 0,3 → Linear 256→2 (66.818 tham số).

**Baseline.** Head được train 100 epoch trên embedding 256 chiều của encoder đóng băng (Adam lr 1e-3, batch 128 cân bằng hai lớp, dừng sớm theo val BAcc, patience 15). Ngưỡng quyết định chọn trên val (0,55) rồi giữ cố định. Baseline chỉ train một lần (seed 42).

**Mẫu khó.** Mẫu train bị dự đoán sai ở đầu mỗi vòng. Dùng để đo "tỉ lệ sửa lỗi", không dùng để chọn dữ liệu train.

**Giai đoạn sửa lỗi (chung cho mọi mode).** Khởi đầu từ head baseline, chạy tối đa 25 vòng × 2 epoch trên **toàn bộ tập train** (batch 128 cân bằng lớp, Adam, cắt gradient 5,0), dừng sớm sau 5 vòng không tiến bộ. Chọn checkpoint có **tỉ lệ sửa lỗi cao nhất** trong các vòng mà val BAcc ≥ val BAcc của baseline − 0,002.

**`uniform_finetune`.** Cross-entropy thường, mọi mẫu trọng số bằng 1 (kể cả mẫu khó). Chỉ train head (lr 3e-4, weight decay 1e-4). Encoder đóng băng hoàn toàn. Là mốc đối chứng.

**`hard_aware_mkmmd_ramp_l010`.** Như `uniform_finetune` nhưng thêm phạt MK-MMD kéo biểu diễn mẫu khó về mẫu đúng cùng lớp (λ = 0,10, tăng dần trong 5 vòng đầu; mẫu khó trọng số 5). Encoder đóng băng.

### Mode cao nhất: `fc_uniform_lr2e5`

- **Hàm loss:** cross-entropy thường, mọi mẫu trọng số 1. Không có số hạng phạt nào khác. Giống `uniform_finetune`.
- **Phần được train:** (1) head `BinaryMLP`, lr 3e-4, weight decay 1e-4, và (2) **lớp `fc` của encoder** (Linear 10.240 → 256, 2.621.696 tham số) với lr **2e-5**, không weight decay. Cả hai khởi đầu từ trọng số baseline.
- **Phần đóng băng:** bốn khối tích chập của encoder. Đầu ra của chúng (10.240 chiều sau khi làm phẳng) được tính trước và lưu cache (fp16); BatchNorm ở chế độ eval, dropout của encoder tắt.
- **Khác `uniform_finetune` đúng một điểm:** lớp `fc` được cập nhật (lr 2e-5) thay vì đóng băng. Vì vậy chênh lệch +0,0094 là tác dụng của việc cho phép lớp `fc` thay đổi.
- **Thống kê của mode:** chạy 21,7 vòng, chọn vòng 17 (trung bình); tỉ lệ sửa lỗi 0,961; giữ mẫu đúng 99,96%; lỗi mới 29,9; train accuracy 0,9991; mất 189 giây mỗi lần trên T4 (không gồm train baseline và tạo cache).

## 3. Nhận xét ngắn

- Train thêm `fc` cho +0,0094 so với đóng băng (10/10 seed), cao hơn nhiều so với mọi biến thể loss trong dự án (±0,002).
- Khi đã mở khóa `fc`, `uniform` tốt hơn `ramp_l010` (+0,0024); phạt MK-MMD không giúp thêm.
- Kernel chạy hai lần độc lập (bản 1 và 2) cho kết quả khớp: `fc_uniform` 0,9955 và 0,9953.

## 4. Lưu ý

1. Baseline chỉ là một lần train; các so sánh chắc chắn là các so sánh ghép cặp giữa mode trong cùng lần chạy. Giá trị p là kiểm định hoán vị dấu chính xác, chưa hiệu chỉnh đa so sánh.
2. Mode này **không còn là encoder đóng băng hoàn toàn**: lớp `fc` được fine-tune bằng nhãn Known/Unknown, tức là dữ liệu Masque có tham gia huấn luyện phần encoder này (encoder pretrain thì chưa thấy Masque).
3. Val không còn độc lập với encoder (encoder đã học khoảng 87% dòng train Swizerland, gồm phần val); chỉ dùng test BAcc để so sánh.
4. Chưa kiểm tra được gần trùng giữa train và test (cùng phiên/cùng trang); chỉ xác nhận 0 dòng trùng hệt và ~0,1% cùng chuỗi gói tin.
5. Đóng băng hoàn toàn với head `h256x2` đạt 0,9891 (đã kiểm tra, xem `PLAN.md`); chưa thử `fc` kết hợp `h256x2`.
