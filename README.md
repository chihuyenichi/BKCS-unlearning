# Network Flow Unlearning

Phân loại network flow theo kịch bản open-world (binary `known` / `unknown`) và đánh giá
các kỹ thuật **instance-wise machine unlearning** trên bộ phân loại đó.

```text
Raw network flow / PCAP
        |
        v
Processed CSV: mỗi dòng = 1 instance, tối đa 10.000 packet [label, relative_time, direction, packet_size]
        |
        v
CNN encoder (SupCon pretrain, đóng băng hoặc fine-tune)  ->  embedding 256 chiều
        |
        v
MLP classifier  ->  KNOWN = 0 / UNKNOWN = 1
        |
        v
Unlearning: quên tập Df (instance AOL), giữ hiệu năng trên Dr
```

## Cấu trúc repo

```text
unlearning/
├── README.md
├── paper/                              # paper nền + bản tóm tắt tiếng Việt
├── experiments/                        # toàn bộ code thí nghiệm (xem experiments/README.md)
│   ├── instance-wise-unlearning/
│   │   ├── kaggle/                     # benchmark 12 mode trên Kaggle
│   │   └── virtual-machine/            # cùng protocol, chạy VM với dataset local
│   └── hard-example-correction/
│       ├── aol-vs-top100/              # AOL vs 100 Top Search
│       └── swizerland-vs-masque500/    # Swizerland vs masque_500 (+ train encoder riêng)
├── reference/unlearning-kf.ipynb       # notebook SupCon/unlearning gốc dùng làm tham chiếu
├── scripts/                            # tiện ích Colab (pipeline cũ) + chuẩn hóa PCAP
├── docs/
│   ├── legacy-pipeline.md              # README của pipeline giai đoạn đầu (lưu trữ)
│   ├── images/                         # ảnh minh họa
│   └── notes/ME.txt                    # ghi chú cá nhân
├── data/                               # (gitignore) dataset CSV, file tải từ Drive
└── artifacts/                          # (gitignore) cache, archive workspace cũ
```

## Thí nghiệm

| Thí nghiệm | Bài toán | Known / Unknown | Tài liệu |
|---|---|---|---|
| `instance-wise-unlearning/kaggle` | Benchmark 12 mode unlearning (baseline, only_head, full_mlp, last_encoder_block, full_model, `l2ul_*`, amnesiac, blindspot, bnd, mkmmd) | AOL / `icloud_100` | [PLAN](experiments/instance-wise-unlearning/kaggle/PLAN.md) |
| `instance-wise-unlearning/virtual-machine` | Cùng protocol, chạy trên VM, dataset local (chưa port các mode `l2ul_*`) | AOL / `icloud_100` | [PLAN](experiments/instance-wise-unlearning/virtual-machine/PLAN.md) |
| `hard-example-correction/aol-vs-top100` | Sửa mẫu đoán sai bằng fine-tune có trọng số (không phải unlearning) | AOL / `icloud_100` | [PLAN](experiments/hard-example-correction/aol-vs-top100/PLAN.md) |
| `hard-example-correction/swizerland-vs-masque500` | Như trên, thêm bước train encoder và các biến thể head/SAM/unfreeze | Swizerland / masque_500 | [PLAN](experiments/hard-example-correction/swizerland-vs-masque500/PLAN.md), [báo cáo encoder v2](experiments/hard-example-correction/swizerland-vs-masque500/REPORT_encoder_v2_fc.md) |

Mặc định benchmark unlearning quên một instance (`FORGET_COUNT = 1`) và chạy ba repeat cho mỗi non-baseline mode.
Hướng dẫn chạy Kaggle/VM, tạo báo cáo XLSX và quy tắc đồng bộ hai notebook nằm trong
[`experiments/README.md`](experiments/README.md). Báo cáo đã chọn lọc nằm ở `results/` của từng thí nghiệm.

## Tài liệu nền

Thư mục [`paper/`](paper/) chứa paper và bản tóm tắt tiếng Việt (xem [`paper/README.vi.md`](paper/README.vi.md)): Instance-Wise Unlearning,
Zero-Shot Machine Unlearning, Can Bad Teaching Induce Forgetting, và [tổng quan các phương pháp](paper/vi_Unlearning_Methods_Overview.md).

## Pipeline giai đoạn đầu

Pipeline multiclass chạy trên Colab/Drive (dữ liệu Drive, dataset baseline, cấu hình, metric, chuẩn hóa PCAP...) đã được lưu trữ tại
[`docs/legacy-pipeline.md`](docs/legacy-pipeline.md).
