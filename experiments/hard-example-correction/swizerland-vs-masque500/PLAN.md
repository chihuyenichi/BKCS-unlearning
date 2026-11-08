# Binary hard-example correction: Swizerland (known) vs masque_500 (unknown)

Cùng bài toán với `../aol-vs-top100`, chỉ đổi input và thêm bước train encoder.

| Vai trò | Dataset Kaggle | Thư mục trong dataset |
|---|---|---|
| KNOWN (`0`) | `daohuyenchi21/swizerland-extracted` | 3 file ở gốc dataset |
| UNKNOWN (`1`) | `daohuyenchi21/masque-500-extracted` | 3 file ở gốc dataset |

Mỗi dataset gồm `train_data.csv`, `test_data.csv`, `label_mapping.csv` (500 nhãn, mỗi nhãn 100 file; train 40,000 dòng,
test 10,000 dòng; mỗi dòng 10,000 packet x `[label, time, direction, size]`). Cả hai dùng chung 500 trang web, chỉ khác
nguồn thu thập, nên bài toán nhị phân là phân biệt hai nguồn.

## Quy trình

1. **`train_encoder.ipynb`** (GPU): huấn luyện encoder DF-style bằng SupCon trên `train_data.csv` của known. Xuất
   `/kaggle/working/pretrain_swizerland.pth` và `train_log.json`.
2. **`pipeline.ipynb`** (GPU): nạp encoder đóng băng, trích embedding, train `BinaryMLP`, rồi chạy 5 mode correction
   (baseline, uniform, hard-weighted CE, + Center, + MK-MMD). Phần này giữ nguyên `../aol-vs-top100/pipeline.py`
   (xem `../aol-vs-top100/PLAN.md`); chỉ đổi tên nguồn (`SWIZERLAND`, `MASQUE_500`), cách tìm dataset, tên checkpoint và tên
   slice đánh giá (`known_swizerland`, `unknown_masque_500`).
3. `generate_benchmark_report.py` / `generate_report.py`: bản sao từ aol-vs-top100, đổi tên đường dẫn.

`pipeline.py`/`pipeline.ipynb` và `train_encoder.py`/`train_encoder.ipynb` giữ đồng bộ (ipynb gồm các ô nối lại bằng đúng
nội dung file `.py`).

## Hai mode mượn từ unlearning (thêm vào bước 2)

Cả hai chạy trên cùng baseline, cùng ngân sách (25 vòng x 2 epoch, Adam lr 3e-4), cùng tiêu chí chọn vòng như các mode khác.
Tập "hard" là các mẫu train đang bị đoán sai ở đầu mỗi vòng, đóng vai `D_f`; các mẫu đang đúng đóng vai `D_r`.

| Mode | Gốc | Cách làm trong bài toán này |
|---|---|---|
| `blindspot_kd` | Bad Teaching / Blindspot (Chundawat et al., 2023) | KL(student, target). Mẫu hard lấy target từ teacher kém (MLP khởi tạo ngẫu nhiên, không train); mẫu còn lại lấy target từ teacher giỏi (bản sao model ở đầu vòng, nên đúng trên mọi mẫu `D_r`). Student khởi tạo từ baseline |
| `boundary_shrink` | Boundary Shrink (Chen et al., CVPR 2023) | Với mỗi mẫu hard, đẩy embedding bằng bước dấu-gradient (kiểu FGSM, tối đa 20 bước) tới khi quyết định đổi sang bên đúng. Fine-tune trên mẫu hard, điểm vượt biên, và một tập `D_r` con cân bằng lớp có cùng kích thước, nhãn là lớp ở phía bên kia biên (với bài toán 2 lớp chính là nhãn thật) |

Điểm khác với bản gốc (cố ý):

- Blindspot: teacher kém nhận embedding đã chuẩn hóa theo từng chiều. Thử nghiệm cho thấy mạng ngẫu nhiên không chuẩn hóa bão hòa về một lớp khi embedding có biên độ lớn (xác suất unknown trung bình 0.98) và khi đó không còn là teacher "vô thông tin". Teacher giỏi được làm mới mỗi vòng thay vì cố định là baseline, vì các mẫu hard đã sửa được không nên bị kéo ngược về dự đoán sai của baseline.
- Boundary Shrink: bản gốc tinh chỉnh chỉ trên `D_f`. Ở đây thêm `D_r` con để tránh trôi, vì phần chọn vòng chỉ chặn bằng validation BAcc.
- Cả hai được dùng để sửa lỗi, trong khi gốc của chúng là để quên. Với Blindspot, đầu ra trên mẫu hard gần 0.5, nên "correction rate" của mode này phản ánh việc mẫu hard rơi sang bên nào của ngưỡng, chưa chắc là dự đoán đúng có chủ đích.
- `history[*].diagnostics` ghi thêm: Blindspot lưu thống kê đầu ra của teacher kém, Boundary Shrink lưu số mẫu vượt biên và biên độ nhiễu trung bình.

## Chạy trên Kaggle

**Encoder**: Add Input `swizerland-extracted`, bật GPU, chạy hết `train_encoder.ipynb`. Sau đó lấy output của notebook
làm input (hoặc tạo dataset từ output) cho bước 2.

**Pipeline**: Add Input `swizerland-extracted`, `masque-500-extracted` và output chứa `pretrain_swizerland.pth`,
bật GPU, chạy `pipeline.ipynb`. Notebook tìm thư mục theo tên (`swizerland-extracted` hoặc `swizerland`,
`masque-500-extracted` hoặc `masque_500`) và tìm duy nhất một file `pretrain_swizerland.pth` dưới `/kaggle/input`.

## Cấu hình train encoder (đã giảm cho vừa phiên Kaggle)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `N_PER_CLASS` | 40 | gốc là 150; 500 lớp x 40 = 20,000 dòng (khoảng 2.4 GB RAM) |
| `VAL_PER_CLASS` | 10 | dòng train giữ lại, chỉ dùng cho kiểm tra 1-NN, không dùng test |
| `EPOCHS` | 100 | gốc là 200; SGD lr 0.1, cosine, warm-up 10 epoch, batch 32, T=0.1 |
| `TIME_BUDGET_HOURS` | 8 | hết giờ thì dừng ở cuối epoch và vẫn lưu checkpoint |
| `CHECKPOINT_EVERY` | 10 | ghi đè `pretrain_swizerland.pth` định kỳ, nên mất phiên cũng còn kết quả |
| `KEEP_PADDING_ZERO` | True | khác bản gốc: giữ các bước đệm bằng 0 sau augmentation |

`test_data.csv` của known không bao giờ được đọc ở bước train encoder.

## Đã kiểm chứng và chưa kiểm chứng

- Đã chạy thử trên máy (CPU, dữ liệu giả rất nhỏ, 600 packet): train encoder, lưu checkpoint, nạp vào encoder của
  `pipeline.py` cho embedding giống hệt nhau (sai khác 0.0), rồi chạy trọn `pipeline.py` qua cả 5 mode.
- **Chưa chạy** trên GPU Kaggle với dữ liệu thật: thời gian mỗi epoch, bộ nhớ và chất lượng encoder là chưa biết. Dòng
  `epoch_seconds` và `one_nn_accuracy` trong log của lần chạy đầu cho biết con số thật.
- Chất lượng encoder: loss SupCon không cho biết nó có tách được known/unknown. `one_nn_accuracy` chỉ đo khả năng nhận ra
  trang web trong known, không phải đo known/unknown.

## MK-MMD "hard-aware" và lượt chạy 10 seed

Động cơ (số liệu từ lượt chạy 7 mode, 100 epoch): trong một batch cân bằng 128 mẫu chỉ có khoảng 2-3 mẫu hard mỗi lớp
(hard khoảng 3.5% train), nên ước lượng MMD không chệch của `hard_weighted_mkmmd` rất nhiễu (lớp chỉ có 1 mẫu hard thì
thành phần nguồn bằng 0). Chênh lệch của mkmmd so với `uniform_finetune` (0.0017 BAcc) nằm trong độ lệch chuẩn giữa các lần chạy.

Ba mode mới `hard_aware_mkmmd` (λ=0.1), `hard_aware_mkmmd_l05` (λ=0.5), `hard_aware_mkmmd_l10` (λ=1.0). Mọi siêu tham số
khác giống `hard_weighted_mkmmd` (25 vòng x 2 epoch, lr 2e-4, hard_weight 5, target_size 32), nên khác biệt chỉ nằm ở:

1. **Nguồn MMD.** Mỗi bước huấn luyện rút 16 mẫu hard mỗi lớp (không hoàn lại) từ toàn bộ hard pool ở đầu vòng, thay vì
   dùng mẫu hard tình cờ có trong batch. Lớp có ít hơn 2 mẫu hard bị bỏ qua; pool nhỏ hơn 16 thì dùng hết.
2. **Đi qua một lượt forward riêng ở chế độ eval.** Nhờ đó (a) running stats của BatchNorm không bị kéo về phía mẫu hard,
   (b) CE vẫn tính trên batch cân bằng như cũ nên trọng số hard (5) không bị nhân lên, (c) nguồn và bank (cũng tính ở eval)
   cùng thang đo. Bản cũ so hidden ở train-mode với bank ở eval-mode. Đây là khác biệt thứ hai ngoài việc lấy mẫu, cả hai
   cùng nằm trong "hard-aware" nên khi mode mới tốt hơn thì chưa tách được phần nào đóng góp.

Đối chứng: λ=0 chính là `hard_weighted_ce`; estimator cũ chính là `hard_weighted_mkmmd`. Lượt chạy này chỉ gồm
baseline, uniform_finetune, hard_weighted_ce, hard_weighted_mkmmd và 3 mode mới, với `REPEATS = 10` (seed ghép cặp
`SEED + repeat` giữa các mode). Ba mode center / blindspot / boundary vẫn còn trong code nhưng không nằm trong `MODES`.
Chọn λ theo validation BAcc trung bình, không theo test. `history[*].diagnostics` ghi `hard_pool_known/unknown` và
`mmd_source_known/unknown`.

Đã kiểm tra trên CPU với dữ liệu giả: các mode cũ cho kết quả giống hệt từng bit trước và sau khi vá; nguồn MMD có đúng
16 mẫu hard khác nhau mỗi lớp, running stats BN không đổi sau lời gọi, gradient chảy tới lớp Linear đầu và BN; chạy trọn
7 mode qua `main()`.

## Lượt chạy 3: λ nhỏ + warm-up + target 64, và CDD

Lượt hard-aware (10 seed) cho thấy `hard_aware_mkmmd` λ=0.1 không hơn có ý nghĩa so với estimator cũ hay `uniform_finetune`,
còn λ lớn làm hại (0.1 > 0.5 > 1.0). Lượt này thử các chỉnh sửa rút ra từ tài liệu (xem báo cáo trong hội thoại):

| Mode | λ | warm-up | target_size | đẩy lớp đối diện (μ) |
|---|---|---|---|---|
| `hard_aware_mkmmd` (đối chứng) | 0.1 | không | 32 | 0 |
| `hard_aware_mkmmd_ramp_l002/l005/l010/l020` | 0.02 / 0.05 / 0.10 / 0.20 | 5 vòng | 64 | 0 |
| `hard_aware_cdd_m05` / `_m10` | 0.1 | 5 vòng | 64 | 0.5 / 1.0 |

- **Warm-up**: hệ số nhân λ ở vòng r là `min(1, (r-1)/5)`, nên vòng 1 chỉ có CE, đến vòng 6 mới đủ λ.
- **CDD** (Kang et al., CVPR 2019, áp dụng trong cùng một miền): loss lớp c là
  `MMD(hard_c, đúng_c) - μ * MMD(hard_c, đúng_lớp_kia)`. Mẫu hard nằm phía sai của biên nên thành phần thứ hai đẩy chúng
  khỏi vùng lớp đã nhầm. Kernel Gaussian bão hòa nên thành phần này tự giới hạn, không cần hinge.
- `ramp_l010` khác `hard_aware_mkmmd` đúng ở warm-up + target 64, nên so hai mode này cho biết riêng hiệu ứng của gói
  (warm-up, target 64). So `cdd_m*` với `ramp_l010` cho biết riêng hiệu ứng của thành phần đẩy.
- **Chẩn đoán thang bandwidth** (`history[*].diagnostics`, vòng 1/5/10/20): khoảng cách bình phương trung vị giữa hidden
  của mẫu hard và mẫu đúng cùng lớp / lớp kia, và giá trị kernel trung bình ứng với từng α trong
  `0.001, 0.01, 0.1, 1.0`. Dùng để quyết định có đổi sang bandwidth theo median heuristic hay không.
- `history[*]` thêm `mmd_intra`, `mmd_inter` (MMD trung bình trong lượt epoch).
- Các mode cũ và `hard_aware_mkmmd` / `_l05` / `_l10` cho kết quả giống hệt từng bit trước và sau khi vá (đã kiểm tra).
- `MODES` của lượt này bỏ `hard_weighted_mkmmd`, `hard_aware_mkmmd_l05`, `hard_aware_mkmmd_l10`; code còn nguyên.

## Kết quả lượt chạy 3 (λ nhỏ + warm-up + target 64, và CDD)

Nguồn: `outputs/e100_cdd/`, báo cáo `outputs/son_tung_hard_example_benchmark_cdd.xlsx` (sheet "Paired"). 10 seed ghép cặp, baseline 1 lần (0.9620, trùng lượt trước; `uniform_finetune` 0.9643 và `hard_aware_mkmmd` 0.9649 cũng trùng từng chữ số => chạy tái lập được).

| Mode | Test BAcc TB | std |
|---|---|---|
| uniform_finetune | 0.9643 | 0.0006 |
| hard_weighted_ce | 0.9638 | 0.0018 |
| hard_aware_mkmmd (λ0.1, không warm-up, target 32) | 0.9649 | 0.0018 |
| ramp λ0.02 / 0.05 / 0.10 / 0.20 | 0.9657 / 0.9656 / 0.9655 / 0.9650 | 0.0014 / 0.0022 / 0.0015 / 0.0012 |
| CDD μ0.5 / μ1.0 | 0.9659 / 0.9652 | 0.0016 / 0.0012 |

- Gộp 6 mode mới - uniform_finetune: +0.0012, CI95 [+0.0002, +0.0019], 9/1, p=0.039 (gộp theo cách chọn sau khi xem kết quả; chỉ gợi ý).
- Không mode đơn lẻ nào vượt `uniform_finetune` sau hiệu chỉnh đa so sánh (tốt nhất cdd_m05: +0.0016, p=0.020).
- Warm-up + target 64 (ramp_l010 vs hard_aware_mkmmd): +0.0006, p=0.381. Thành phần đẩy CDD (cdd vs ramp_l010): +0.0004 (μ0.5, p=0.156) và -0.0003 (μ1.0, p=0.352); trung bình hai mức = +0.0000, p=0.861. Không có bằng chứng CDD giúp.
- λ trong 0.02–0.20: phẳng (chênh ≤0.0007, đều không có ý nghĩa).
- Chẩn đoán bandwidth (vòng 1): median d² ≈ 7.3 (hard Known vs correct Known), 3.4 (vs correct Unknown). Kernel trung bình: α0.001≈0.99, α0.01≈0.92, α0.1≈0.47, α1.0≈0.006. => α1.0 gần như bất hoạt, α0.001 gần hằng; α0.1 ≈ median heuristic. Tiền đề của CDD đúng: mẫu hard nằm gần lớp kia hơn lớp đúng của chúng.

## Lượt chạy 4: ba thí nghiệm song song (head, mở khóa encoder, SAM)

Mã: `experiments/` (`ext_common.py`, `ext_head.py`, `ext_unfreeze.py`, `ext_sam.py`, `build_experiments.py`). Script Kaggle sinh bằng `python experiments/build_experiments.py` = `pipeline.py` (chỉ sửa 2 chỗ, có assert) + phần mở rộng; `pipeline.py`/`pipeline.ipynb` và các lượt cũ không đổi.

| Kernel | Phần cứng | Nội dung | Ngân sách |
|---|---|---|---|
| `daohuyenchi21/tung-exp-head-sweep` | CPU | 8 cấu hình head (linear, h256, h512, 2 lớp ẩn, batch 64/256, 200 epoch) × 10 seed base khác nhau; mỗi cái thêm `uniform_finetune` | 8.5 giờ |
| `daohuyenchi21/tung-exp-unfreeze` | T4 | mở khóa `fc` (lớp chiếu) hoặc block 4 + `fc`; cache đầu ra block 1-3 (fp16); BN của phần mở khóa ở eval, lr encoder 5e-6 / 2e-5, không weight decay | 4 giờ |
| `daohuyenchi21/tung-exp-sam` | T4 | SAM ρ=0.05/0.10 trên `uniform_finetune`, ρ=0.05 trên warm-up λ=0.10 | 3.5 giờ |

- Mỗi kernel tự có `baseline`, `uniform_finetune`, `hard_aware_mkmmd_ramp_l010` để so ghép cặp trong cùng lượt. `uniform_finetune` và `ramp_l010` dùng cùng seed (SEED+repeat) nên có thể tái hiện 0.9643 / 0.9655 của lượt CDD (với B có cuDNN không tất định nên chỉ xấp xỉ).
- Vòng lặp theo repeat (repeat-major), ghi `summary.json` sau mỗi run, dừng khi hết ngân sách (`experiment.skipped_runs`, `status`). Số seed của block 4 tối đa 6.
- Phát hiện khi đọc code: baseline chỉ train 1 lần; 10 "repeat" chỉ là seed của giai đoạn sửa lỗi. Val BAcc của base vẫn tăng đến epoch 99/100 (0.9549 @30, 0.9573 @50, 0.9612 @70, 0.9630 @99), tức số epoch tối đa của base đang bị chạm; head sweep có cấu hình 200 epoch để kiểm tra.
- Kiểm tra: test đơn vị (`SAM` với ρ=0 giống từng bit bản gốc; ρ cực nhỏ ≡ không SAM nên bước 2 phát lại đúng dropout/rút mẫu; trọng số được khôi phục sau nhiễu loạn; `front+tail` tái tạo đúng encoder; mô hình mở khóa khởi đầu giống baseline sai lệch ≤1e-5; `FlexMLP((256,))` giống `BinaryMLP`, seed 42 khớp `train_base`) + chạy thử toàn bộ cả 3 script trên dữ liệu giả (CPU) + thử dừng theo ngân sách. Chưa đo thời gian thật trên T4.

## Kết quả lượt chạy 4 (head, mở khóa encoder, SAM; encoder cũ)

Nguồn: `outputs/e100_head_*`, `e100_unfreeze`, `e100_sam`; báo cáo `outputs/son_tung_hard_example_benchmark_{head10,unfreeze,sam}.xlsx` (đều có sheet "Paired"). Test BAcc trung bình, so ghép cặp theo seed trong cùng lượt.

**C - SAM (10 seed, xong 2.7 giờ): không giúp, làm xấu đi.** Đối chứng tái lập đúng lượt trước (`uniform_finetune` 0.9643, `ramp_l010` 0.9655).

| Mode | Test BAcc | So với đối chứng |
|---|---|---|
| uniform_sam ρ0.05 | 0.9639 | -0.0004 (1/7, p=0.016) |
| uniform_sam ρ0.10 | 0.9627 | -0.0016 (0/10, p=0.002) |
| ramp_l010 + SAM ρ0.05 | 0.9640 | -0.0015 vs ramp_l010 (2/8, p=0.014) |

Tỉ lệ sửa lỗi giảm (0.35 -> 0.28-0.31). Kết luận: bỏ SAM.

**A - head (đóng băng encoder cũ, 10 seed, mỗi seed train base riêng).** Lần đầu (`tung-exp-head-sweep`, 8 cấu hình) dừng theo ngân sách ở 9.25 giờ chỉ có 1-3 seed/cấu hình (CPU chậm hơn ước tính); chạy lại 4 cấu hình quan trọng thành 4 kernel song song `tung-head-{h256,h256x2,h512,h512x2}`, đủ 10 seed.

| Cấu hình | Tham số | Base | Sau uniform_finetune | Chênh uniform vs h256 |
|---|---|---|---|---|
| h256 (head hiện tại) | 67k | 0.9574 ± 0.0020 | 0.9630 ± 0.0013 | mốc |
| h512 | 134k | 0.9608 | 0.9662 | +0.0032 (9/1, p=0.004) |
| h256x2 | 133k | 0.9699 | 0.9750 | +0.0121 (10/0, p=0.002) |
| h512x2 | 397k | 0.9717 | 0.9767 | +0.0137 (10/0, p=0.002) |

- Độ sâu quan trọng hơn độ rộng (h256x2 ≈ h512 về số tham số nhưng gấp 4 lần mức tăng). h512x2 chỉ hơn h256x2 +0.0017 mà chậm gấp 5 -> chọn `h256x2`.
- Lần đầu (n nhỏ): head tuyến tính 0.79 (embedding không tách tuyến tính được); batch 64/256 và 200 epoch phẳng (best epoch seed 0 = 106/136, base không bị trần 100 epoch chặn).
- Baseline dao động giữa các seed (h256 base 0.9547-0.9606, sd 0.0020); baseline 0.9620 của các lượt trước là một lần rút khá may. So sánh ghép cặp trong cùng lượt không bị ảnh hưởng.

**B - mở khóa encoder cũ (T4, dừng theo ngân sách ở 4.06 giờ; n = 3-5 seed):**

| Mode | n | Test BAcc | Correction rate |
|---|---|---|---|
| uniform_finetune (đối chứng) | 5 | 0.9641 | 0.35 |
| fc_uniform_lr2e5 | 4 | 0.9884 | 0.96 |
| fc_ramp_l010_lr2e5 | 4 | 0.9858 | 0.94 |
| b4_uniform_lr5e6 / lr2e5 | 3 | 0.9966 / 0.9971 | 0.99 |
| b4_ramp_l010_lr2e5 | 3 | 0.9942 | 0.95 |

- Mọi seed thắng đối chứng (W/L 4/0 và 3/0; p chính xác nhỏ nhất với n=3-4 là 0.125-0.25, nên bằng chứng là độ lớn và tính nhất quán, không phải p). Khi mở khóa, `uniform` ≥ `ramp_l010` (MMD không còn vai trò).
- Train/Val/Test accuracy: baseline 0.9687 / 0.9627 / 0.9592; `b4_uniform_lr2e5` 0.9994 / 0.9973 / 0.9971. Lúc đóng băng, train cũng chỉ ~97% (underfit, không phải overfit): embedding không đủ thông tin; mở khóa cho phép học đặc trưng đúng cho bài toán.
- Kiểm tra trùng lặp train/test (băm từng dòng 4 file CSV): 0 dòng test trùng hệt với train; chỉ ~0.08-0.09% có cùng chuỗi (hướng, kích thước). Gần trùng (cùng phiên/cùng trang) KHÔNG kiểm tra được vì không có metadata; cách tách rõ nhất là chia test theo nhóm trang. Baseline kernel B là 0.9592 (cuDNN không tất định, cache fp16) nên chỉ so với đối chứng cùng kernel.

**Đánh giá encoder cũ (`outputs/encoder_train/train_log.json`):** SupCon 40 dòng/lớp, 100 epoch, 5744 giây; 1-NN (nhận ra trang trong 500 lớp Swizerland, val 10 dòng/lớp lấy từ chính train) = 82.8% @20, 85.1% @40, 86.9% @60, 88.4% @80, **88.64% @100**, còn tăng; loss 0.126. Mỗi lớp trong `train_data.csv` có đúng 80 dòng.

## Lượt chạy 5: train lại encoder (70 dòng/lớp, 200 epoch) và đánh giá lại

Encoder: `train_encoder_v2.py` (chỉ đổi `N_PER_CLASS` 40 -> 70, `EPOCHS` 100 -> 200, `TIME_BUDGET_HOURS` 10.5), kernel `tung-train-encoder-v2`, chỉ Swizerland (encoder chưa thấy Masque). Xong đủ 200 epoch trong 5.74 giờ (102 giây/epoch). 1-NN: 84.4% @20, 87.6% @100, 89.5% @140, 91.4% @180, **91.22% @200** (loss 0.110; encoder cũ 88.64%). SHA mới `2cfaa8b1...` (cũ `e88fce38...`).

Đánh giá: `tung-head-enc2` (CPU, `lin`/`h256`/`h256x2`, 10 seed, đóng băng) và `tung-unfreeze-fc-enc2` (T4, đóng băng đối chứng + mở khóa `fc`, 10 seed). Báo cáo: `outputs/son_tung_hard_example_benchmark_encoder_v2_head.xlsx`, `..._encoder_v2_fc.xlsx`. (Kernel head không lưu đủ ma trận nhầm lẫn, nên xlsx head được dựng bằng `head_to_standard.py`: confusion/precision/F1/train accuracy suy ra từ recall và số lỗi train; hàng Baseline = trung bình `h256__base`.)

| Head | Mode | Encoder cũ | Encoder mới | Chênh (ghép cặp, 10 seed) |
|---|---|---|---|---|
| h256 | base | 0.9574 | 0.9833 ± 0.0008 | +0.0258 (10/0, p=0.002) |
| h256 | uniform_finetune | 0.9630 | **0.9855** ± 0.0005 | +0.0225 (10/0, p=0.002) |
| h256x2 | base | 0.9699 | 0.9867 ± 0.0007 | +0.0168 (10/0) |
| h256x2 | uniform_finetune | 0.9750 | **0.9891** ± 0.0006 | +0.0141 (10/0) |
| lin | base | 0.7926 (3 seed) | 0.8104 ± 0.0013 | +0.018 |

Mở khóa `fc` trên encoder mới (head h256): baseline 0.9834, `uniform_finetune` 0.9859, `ramp_l010` 0.9848, **`fc_uniform_lr2e5` 0.9953** (+0.0094 vs uniform_finetune, 10/0, p=0.002), `fc_ramp_l010_lr2e5` 0.9929 (+0.0070). Kernel chạy hai lần độc lập (bản 1 và 2, do đẩy trùng): `fc_uniform` 0.9955 và 0.9953, `fc_ramp` 0.9929 và 0.9929 -> tái lập tốt.

Nhận xét:
- Train lại encoder là nguyên nhân chính: chỉ đổi encoder (vẫn đóng băng) đã +2.3 điểm ở mọi seed; đóng băng `h256x2` (0.9891) ngang mức mở khóa encoder cũ (0.9884) mà không cần mở khóa và không còn nghi vấn rò rỉ của hướng mở khóa. Known/Unknown recall cân bằng (0.988 / 0.990).
- `uniform_finetune` chỉ còn +0.002 (so với +0.005 trước đây); `ramp_l010` thấp hơn `uniform_finetune` 0.0011 (p=0.008) trên encoder mới.
- Mở khóa `fc` còn thêm ~+0.009 so với đóng băng.

Lưu ý: (1) val Known của head không còn độc lập với encoder (encoder đã học ~87% dòng train Swizerland, gồm phần val); chỉ dùng test BAcc để so sánh. (2) Chưa thử `fc` + `h256x2`, chưa thử block 4 trên encoder mới. (3) Gần trùng train/test vẫn chưa kiểm chứng được. (4) p-value chưa hiệu chỉnh đa so sánh; bản 2 của `tung-head-enc2` chưa tải (cùng mã/encoder với bản 1).
