# Notebook Run Log - train_pipeline.ipynb

- Exported at: `2026-09-13T10:49:17`
- Source notebook: `MLP-Classfication/code/train_pipeline.ipynb`

## Cell 1

- execution_count: `39`
- overview/source: `# OVERVIEW: Chuẩn bị dependency runtime cho Colab trước khi đọc PCAP và tạo DataLoader.`

### Output 1

```text
Các dependency runtime đã sẵn sàng.
```

## Cell 2

- execution_count: `40`
- overview/source: `# OVERVIEW: Cấu hình một mode duy nhất: Google Colab mount Google Drive, đọc PCAP baseline,`

### Output 1

```text
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Runtime: Google Colab + Google Drive
DATA_DIR đang train:
  - /content/drive/MyDrive/Traffic FingerPrinting /Data/273 (200samples key)
OUTPUT_DIR: /content/drive/MyDrive/unlearning-artifacts/notebook
Dataset notes:
  - 273 (200samples key): Baseline tốt nhất: cấu trúc <label>/*.pcap, đã xác nhận 9005 file .pcap.
  - 273 (lan 1): Cùng họ 273, phù hợp để mở rộng sau baseline và kiểm tra ổn định theo lần thu thập.
  - AOL (lan 1): Dùng được nhưng label là query phrase, nên ưu tiên cho open-world/generalization sau baseline.
  - Data iPad/Data iPad new: Cross-device iPad, có tầng trung gian và PCAP lớn hơn; dùng sau khi pipeline ổn.
  - Data iPhone/Data iPhone new: Cross-device iPhone, có tầng trung gian; phù hợp để test domain shift.
```

## Cell 3

- execution_count: `41`
- overview/source: `# OVERVIEW: Nạp thư viện dùng chung, kiểm tra CUDA và chọn DEVICE CPU/GPU an toàn cho pipeline Colab.`

### Output 1

```text
PyTorch version: 2.11.0+cu128
torch_cuda_compiled=True, torch_cuda_available=True
CUDA device: Tesla T4
Device đang dùng: cuda
```

## Cell 31

- execution_count: `68`
- overview/source: `# OVERVIEW: Quét toàn bộ nhãn gốc trong DATA_DIR trước khi lọc, lưu label_inventory.json để biết dataset có bao nhiêu class.`

### Output 1

```text
Tổng số file PCAP/CAP/TXT/CSV: 9005
Tổng số nhãn gốc: 45
Nhãn đủ điều kiện split >= 100 samples: 45
Nhãn bị loại vì quá ít sample: 0
Đã lưu inventory: /content/drive/MyDrive/unlearning-artifacts/notebook/label_inventory.json
Top nhãn theo số sample:
  united_airlines                        205
  37_200_000                             200
  english_to_spanish                     200
  food_near_me                           200
  office_365                             200
  office_depot                           200
  offset                                 200
  ohio_state_football                    200
  omegle                                 200
  paul_pelosi                            200
  phillies_game                          200
  powerball_jackpot                      200
  qr_code_generator                      200
  quavo                                  200
  quest_diagnostics                      200
  quickbooks                             200
  quizlet                                200
  rate_my_professor                      200
  ray_guy                                200
  realtor                                200
```

## Cell 32

- execution_count: `69`
- overview/source: `# OVERVIEW: Random split nhãn ở cấp class vào known/unknown-train/holdout, lưu label_split theo seed, số nhãn và ratio để tái lập thí nghiệm.`

### Output 1

```text
Tạo label split mới: /content/drive/MyDrive/unlearning-artifacts/notebook/label_split_seed42_labels24_k45_u35.json
Tóm tắt label split:
KNOWN_LABELS: 11 nhãn, 2205 samples
  office_365                             200
  phillies_game                          200
  powerball_jackpot                      200
  quavo                                  200
  realtor                                200
  soap2day                               200
  solitaire                              200
  ticketmaster                           200
  united_airlines                        205
  united_states_elections_2022           200
  us_house_elections_2022                200
UNKNOWN_LABELS train: 8 nhãn, 1600 samples
  office_depot                           200
  qr_code_generator                      200
  take_off_dead                          200
  taylor_swift                           200
  twitter                                200
  usps                                   200
  v_for_vendetta                         200
  xfinity_customer_service               200
HOLDOUT_UNKNOWN_LABELS test-only: 5 nhãn, 1000 samples
  quest_diagnostics                      200
  ross                                   200
  social_security                        200
  takeoff_shooting                       200
  tmz                                    200
Tổng nhãn được dùng: 24 / 45
Tổng sample được dùng: 4805 / 9005
Đường dẫn split: /content/drive/MyDrive/unlearning-artifacts/notebook/label_split_seed42_labels24_k45_u35.json
```

## Cell 33

- execution_count: `70`
- overview/source: `# OVERVIEW: Tạo FlowRecord sau label split và giới hạn số PCAP mỗi nhãn để quick debug không phải parse quá nhiều flow.`

### Output 1

```text
Giới hạn mỗi nhãn tối đa 160 PCAP: 4805 -> 3840 samples
Tổng số sample sau khi lọc theo label split: 3840
Số sample theo role: {'known': 1760, 'unknown-holdout': 800, 'unknown-train': 1280}
Chi tiết nhãn được dùng:
office_365                             160  known
office_depot                           160  unknown-train
phillies_game                          160  known
powerball_jackpot                      160  known
qr_code_generator                      160  unknown-train
quavo                                  160  known
quest_diagnostics                      160  unknown-holdout
realtor                                160  known
ross                                   160  unknown-holdout
soap2day                               160  known
social_security                        160  unknown-holdout
solitaire                              160  known
take_off_dead                          160  unknown-train
takeoff_shooting                       160  unknown-holdout
taylor_swift                           160  unknown-train
ticketmaster                           160  known
tmz                                    160  unknown-holdout
twitter                                160  unknown-train
united_airlines                        160  known
united_states_elections_2022           160  known
us_house_elections_2022                160  known
usps                                   160  unknown-train
v_for_vendetta                         160  unknown-train
xfinity_customer_service               160  unknown-train
```

## Cell 34

- execution_count: `71`
- overview/source: `# OVERVIEW: Split dữ liệu và tạo DataLoader lazy; validation gồm cả unknown-train và holdout-unknown để tune threshold.`

### Output 1

```text
train=2128, validation=656, test=1056
Binary counts:
  train     : {'known': 1232, 'unknown': 896}
  validation: {'known': 264, 'unknown': 392}
  test      : {'known': 264, 'unknown': 792}
Packet feature cache: /content/drive/MyDrive/unlearning-artifacts/notebook/packet-feature-cache/features
Packet tensor: (256, 5)
Mask tensor: (256,), label: 1
```

## Cell 35

- execution_count: `72`
- overview/source: `# OVERVIEW: Khởi tạo mô hình đúng pipeline trong plan: encoder 256 chiều + MLP output known/unknown.`

### Output 1

```text
FlowModel(
  (encoder): FlowEncoder(
    (network): Sequential(
      (0): Conv1d(5, 64, kernel_size=(5,), stride=(1,), padding=(2,))
      (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): GELU(approximate='none')
      (3): Conv1d(64, 128, kernel_size=(5,), stride=(1,), padding=(2,))
      (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): GELU(approximate='none')
      (6): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
      (7): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): GELU(approximate='none')
    )
    (projection): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
  )
  (classifier): MLPClassifier(
    (network): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (2): GELU(approximate='none')
      (3): Dropout(p=0.2, inplace=False)
      (4): Linear(in_features=512, out_features=256, bias=True)
      (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (6): GELU(approximate='none')
      (7): Dropout(p=0.2, inplace=False)
      (8): Linear(in_features=256, out_features=128, bias=True)
      (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (10): GELU(approximate='none')
      (11): Dropout(p=0.2, inplace=False)
      (12): Linear(in_features=128, out_features=64, bias=True)
      (13): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (14): GELU(approximate='none')
      (15): Dropout(p=0.2, inplace=False)
      (16): Linear(in_features=64, out_features=32, bias=True)
      (17): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (18): GELU(approximate='none')
      (19): Dropout(p=0.2, inplace=False)
      (20): Linear(in_features=32, out_features=2, bias=True)
    )
  )
)
Tổng số tham số: 434,402
```

## Cell 36

- execution_count: `73`
- overview/source: `# OVERVIEW: Train end-to-end encoder và MLP trên known + unknown-train; dùng class weights và balanced accuracy để giảm lệch về known.`

### Output 1

```text
Train label counts=[1232, 896], class_weights=[0.8421052098274231, 1.1578946113586426]
Train config: epochs=12, batch_size=128, train_batches=17, val_batches=6, max_train_batches_per_epoch=None, max_eval_batches=None, log_every_n_batches=2, use_class_weights=True, checkpoint_score_metric=balanced_accuracy, unknown_threshold=0.5
```

### Output 2

```text
/usr/local/lib/python3.13/dist-packages/scapy/layers/tls/crypto/groups.py:25: CryptographyDeprecationWarning: Diffie-Hellman over finite fields (FFDH) is deprecated and support will be removed in a future release. Use a more modern key exchange algorithm.
  from cryptography.hazmat.primitives.asymmetric.dh import DHParameterNumbers
```

### Output 3

```text
Epoch 001/12 | batch 0001/17 | loss=0.7522 | elapsed=35.9s
Epoch 001/12 | batch 0002/17 | loss=0.7008 | elapsed=71.1s
Epoch 001/12 | batch 0004/17 | loss=0.7065 | elapsed=150.2s
Epoch 001/12 | batch 0006/17 | loss=0.7042 | elapsed=223.6s
Epoch 001/12 | batch 0008/17 | loss=0.7032 | elapsed=293.0s
Epoch 001/12 | batch 0010/17 | loss=0.6739 | elapsed=351.7s
Epoch 001/12 | batch 0012/17 | loss=0.7208 | elapsed=424.7s
Epoch 001/12 | batch 0014/17 | loss=0.6862 | elapsed=492.8s
Epoch 001/12 | batch 0016/17 | loss=0.6805 | elapsed=567.2s
Epoch 001/12 done | train_loss=0.7008 | val_acc=0.5976 | val_known_recall=0.0000 | val_unknown_recall=1.0000 | val_balanced_acc=0.5000 | score=0.5000 | epoch_time=902.0s | val_time=306.4s | confusion=[[0, 264], [0, 392]]
Epoch 002/12 | batch 0001/17 | loss=0.7140 | elapsed=0.3s
Epoch 002/12 | batch 0002/17 | loss=0.6977 | elapsed=0.6s
Epoch 002/12 | batch 0004/17 | loss=0.6765 | elapsed=1.1s
Epoch 002/12 | batch 0006/17 | loss=0.6697 | elapsed=1.6s
Epoch 002/12 | batch 0008/17 | loss=0.6752 | elapsed=2.1s
Epoch 002/12 | batch 0010/17 | loss=0.6819 | elapsed=2.7s
Epoch 002/12 | batch 0012/17 | loss=0.6984 | elapsed=3.3s
Epoch 002/12 | batch 0014/17 | loss=0.6888 | elapsed=3.8s
Epoch 002/12 | batch 0016/17 | loss=0.6791 | elapsed=4.3s
Epoch 002/12 done | train_loss=0.6879 | val_acc=0.6037 | val_known_recall=0.0644 | val_unknown_recall=0.9668 | val_balanced_acc=0.5156 | score=0.5156 | epoch_time=5.9s | val_time=1.4s | confusion=[[17, 247], [13, 379]]
Epoch 003/12 | batch 0001/17 | loss=0.6535 | elapsed=0.3s
Epoch 003/12 | batch 0002/17 | loss=0.6675 | elapsed=0.6s
Epoch 003/12 | batch 0004/17 | loss=0.6468 | elapsed=1.1s
Epoch 003/12 | batch 0006/17 | loss=0.6612 | elapsed=1.7s
Epoch 003/12 | batch 0008/17 | loss=0.6514 | elapsed=2.2s
Epoch 003/12 | batch 0010/17 | loss=0.6738 | elapsed=2.8s
Epoch 003/12 | batch 0012/17 | loss=0.6001 | elapsed=3.3s
Epoch 003/12 | batch 0014/17 | loss=0.6424 | elapsed=3.9s
Epoch 003/12 | batch 0016/17 | loss=0.6862 | elapsed=4.4s
Epoch 003/12 done | train_loss=0.6539 | val_acc=0.6082 | val_known_recall=0.2121 | val_unknown_recall=0.8750 | val_balanced_acc=0.5436 | score=0.5436 | epoch_time=6.0s | val_time=1.4s | confusion=[[56, 208], [49, 343]]
Epoch 004/12 | batch 0001/17 | loss=0.6193 | elapsed=0.3s
Epoch 004/12 | batch 0002/17 | loss=0.6357 | elapsed=0.6s
Epoch 004/12 | batch 0004/17 | loss=0.5809 | elapsed=1.1s
Epoch 004/12 | batch 0006/17 | loss=0.7042 | elapsed=1.6s
Epoch 004/12 | batch 0008/17 | loss=0.7018 | elapsed=2.2s
Epoch 004/12 | batch 0010/17 | loss=0.6055 | elapsed=2.7s
Epoch 004/12 | batch 0012/17 | loss=0.5906 | elapsed=3.2s
Epoch 004/12 | batch 0014/17 | loss=0.6189 | elapsed=3.7s
Epoch 004/12 | batch 0016/17 | loss=0.6193 | elapsed=4.3s
Epoch 004/12 done | train_loss=0.6348 | val_acc=0.5549 | val_known_recall=0.8598 | val_unknown_recall=0.3495 | val_balanced_acc=0.6047 | score=0.6047 | epoch_time=5.8s | val_time=1.3s | confusion=[[227, 37], [255, 137]]
Epoch 005/12 | batch 0001/17 | loss=0.6429 | elapsed=0.3s
Epoch 005/12 | batch 0002/17 | loss=0.5545 | elapsed=0.5s
Epoch 005/12 | batch 0004/17 | loss=0.5770 | elapsed=1.1s
Epoch 005/12 | batch 0006/17 | loss=0.6199 | elapsed=1.6s
Epoch 005/12 | batch 0008/17 | loss=0.6047 | elapsed=2.2s
Epoch 005/12 | batch 0010/17 | loss=0.5494 | elapsed=2.7s
Epoch 005/12 | batch 0012/17 | loss=0.5931 | elapsed=3.3s
Epoch 005/12 | batch 0014/17 | loss=0.6251 | elapsed=3.8s
Epoch 005/12 | batch 0016/17 | loss=0.5692 | elapsed=4.3s
Epoch 005/12 done | train_loss=0.5955 | val_acc=0.6341 | val_known_recall=0.5152 | val_unknown_recall=0.7143 | val_balanced_acc=0.6147 | score=0.6147 | epoch_time=5.8s | val_time=1.3s | confusion=[[136, 128], [112, 280]]
Epoch 006/12 | batch 0001/17 | loss=0.6116 | elapsed=0.3s
Epoch 006/12 | batch 0002/17 | loss=0.5875 | elapsed=0.5s
Epoch 006/12 | batch 0004/17 | loss=0.6964 | elapsed=1.1s
Epoch 006/12 | batch 0006/17 | loss=0.6666 | elapsed=1.6s
Epoch 006/12 | batch 0008/17 | loss=0.5881 | elapsed=2.2s
Epoch 006/12 | batch 0010/17 | loss=0.6332 | elapsed=2.7s
Epoch 006/12 | batch 0012/17 | loss=0.5630 | elapsed=3.3s
Epoch 006/12 | batch 0014/17 | loss=0.5691 | elapsed=3.8s
Epoch 006/12 | batch 0016/17 | loss=0.5920 | elapsed=4.4s
Epoch 006/12 done | train_loss=0.6105 | val_acc=0.5579 | val_known_recall=0.8674 | val_unknown_recall=0.3495 | val_balanced_acc=0.6085 | score=0.6085 | epoch_time=5.8s | val_time=1.3s | confusion=[[229, 35], [255, 137]]
Epoch 007/12 | batch 0001/17 | loss=0.5363 | elapsed=0.3s
Epoch 007/12 | batch 0002/17 | loss=0.5914 | elapsed=0.6s
Epoch 007/12 | batch 0004/17 | loss=0.5951 | elapsed=1.1s
Epoch 007/12 | batch 0006/17 | loss=0.5808 | elapsed=1.7s
Epoch 007/12 | batch 0008/17 | loss=0.6341 | elapsed=2.2s
Epoch 007/12 | batch 0010/17 | loss=0.5717 | elapsed=2.7s
Epoch 007/12 | batch 0012/17 | loss=0.6749 | elapsed=3.3s
Epoch 007/12 | batch 0014/17 | loss=0.5876 | elapsed=3.8s
Epoch 007/12 | batch 0016/17 | loss=0.5653 | elapsed=4.4s
Epoch 007/12 done | train_loss=0.5878 | val_acc=0.6189 | val_known_recall=0.6250 | val_unknown_recall=0.6148 | val_balanced_acc=0.6199 | score=0.6199 | epoch_time=5.9s | val_time=1.3s | confusion=[[165, 99], [151, 241]]
Epoch 008/12 | batch 0001/17 | loss=0.5946 | elapsed=0.3s
Epoch 008/12 | batch 0002/17 | loss=0.5345 | elapsed=0.6s
Epoch 008/12 | batch 0004/17 | loss=0.5982 | elapsed=1.1s
Epoch 008/12 | batch 0006/17 | loss=0.5796 | elapsed=1.7s
Epoch 008/12 | batch 0008/17 | loss=0.6894 | elapsed=2.2s
Epoch 008/12 | batch 0010/17 | loss=0.5592 | elapsed=2.8s
Epoch 008/12 | batch 0012/17 | loss=0.5569 | elapsed=3.4s
Epoch 008/12 | batch 0014/17 | loss=0.5884 | elapsed=3.9s
Epoch 008/12 | batch 0016/17 | loss=0.7235 | elapsed=4.5s
Epoch 008/12 done | train_loss=0.5750 | val_acc=0.5701 | val_known_recall=0.8485 | val_unknown_recall=0.3827 | val_balanced_acc=0.6156 | score=0.6156 | epoch_time=5.9s | val_time=1.3s | confusion=[[224, 40], [242, 150]]
Epoch 009/12 | batch 0001/17 | loss=0.5356 | elapsed=0.3s
Epoch 009/12 | batch 0002/17 | loss=0.5538 | elapsed=0.5s
Epoch 009/12 | batch 0004/17 | loss=0.5529 | elapsed=1.1s
Epoch 009/12 | batch 0006/17 | loss=0.5701 | elapsed=1.6s
Epoch 009/12 | batch 0008/17 | loss=0.5978 | elapsed=2.2s
Epoch 009/12 | batch 0010/17 | loss=0.5646 | elapsed=2.7s
Epoch 009/12 | batch 0012/17 | loss=0.6524 | elapsed=3.3s
Epoch 009/12 | batch 0014/17 | loss=0.5707 | elapsed=3.9s
Epoch 009/12 | batch 0016/17 | loss=0.6156 | elapsed=4.4s
Epoch 009/12 done | train_loss=0.5906 | val_acc=0.5838 | val_known_recall=0.8220 | val_unknown_recall=0.4235 | val_balanced_acc=0.6227 | score=0.6227 | epoch_time=6.0s | val_time=1.4s | confusion=[[217, 47], [226, 166]]
Epoch 010/12 | batch 0001/17 | loss=0.6319 | elapsed=0.3s
Epoch 010/12 | batch 0002/17 | loss=0.5896 | elapsed=0.6s
Epoch 010/12 | batch 0004/17 | loss=0.6282 | elapsed=1.1s
Epoch 010/12 | batch 0006/17 | loss=0.5780 | elapsed=1.7s
Epoch 010/12 | batch 0008/17 | loss=0.5726 | elapsed=2.2s
Epoch 010/12 | batch 0010/17 | loss=0.6071 | elapsed=2.8s
Epoch 010/12 | batch 0012/17 | loss=0.4976 | elapsed=3.4s
Epoch 010/12 | batch 0014/17 | loss=0.5310 | elapsed=3.9s
Epoch 010/12 | batch 0016/17 | loss=0.5746 | elapsed=4.5s
Epoch 010/12 done | train_loss=0.5718 | val_acc=0.6341 | val_known_recall=0.4924 | val_unknown_recall=0.7296 | val_balanced_acc=0.6110 | score=0.6110 | epoch_time=6.0s | val_time=1.4s | confusion=[[130, 134], [106, 286]]
Epoch 011/12 | batch 0001/17 | loss=0.5380 | elapsed=0.3s
Epoch 011/12 | batch 0002/17 | loss=0.5496 | elapsed=0.5s
Epoch 011/12 | batch 0004/17 | loss=0.5510 | elapsed=1.0s
Epoch 011/12 | batch 0006/17 | loss=0.5856 | elapsed=1.6s
Epoch 011/12 | batch 0008/17 | loss=0.5417 | elapsed=2.1s
Epoch 011/12 | batch 0010/17 | loss=0.5688 | elapsed=2.6s
Epoch 011/12 | batch 0012/17 | loss=0.5353 | elapsed=3.1s
Epoch 011/12 | batch 0014/17 | loss=0.5207 | elapsed=3.7s
Epoch 011/12 | batch 0016/17 | loss=0.5165 | elapsed=4.2s
Epoch 011/12 done | train_loss=0.5523 | val_acc=0.5457 | val_known_recall=0.8977 | val_unknown_recall=0.3087 | val_balanced_acc=0.6032 | score=0.6032 | epoch_time=5.7s | val_time=1.3s | confusion=[[237, 27], [271, 121]]
Epoch 012/12 | batch 0001/17 | loss=0.4907 | elapsed=0.3s
Epoch 012/12 | batch 0002/17 | loss=0.5133 | elapsed=0.5s
Epoch 012/12 | batch 0004/17 | loss=0.5385 | elapsed=1.1s
Epoch 012/12 | batch 0006/17 | loss=0.5194 | elapsed=1.6s
Epoch 012/12 | batch 0008/17 | loss=0.5751 | elapsed=2.2s
Epoch 012/12 | batch 0010/17 | loss=0.5023 | elapsed=2.7s
Epoch 012/12 | batch 0012/17 | loss=0.5352 | elapsed=3.3s
Epoch 012/12 | batch 0014/17 | loss=0.5511 | elapsed=3.8s
Epoch 012/12 | batch 0016/17 | loss=0.4986 | elapsed=4.4s
Epoch 012/12 done | train_loss=0.5291 | val_acc=0.6296 | val_known_recall=0.7727 | val_unknown_recall=0.5332 | val_balanced_acc=0.6529 | score=0.6529 | epoch_time=5.8s | val_time=1.3s | confusion=[[204, 60], [183, 209]]
Train finished in 966.7s
Best validation score (balanced_accuracy): 0.6529
```

## Cell 37

- execution_count: `74`
- overview/source: `# OVERVIEW: Sweep threshold unknown trên validation để chọn ngưỡng tốt nhất theo balanced accuracy trước khi test.`

### Output 1

```text
Best unknown threshold: 0.45
Threshold sweep trên validation:
thr=0.20 | bal_acc=0.5737 | known_recall=0.3106 | unknown_recall=0.8367 | unknown_precision=0.6431 | confusion=[[82, 182], [64, 328]]
thr=0.25 | bal_acc=0.5862 | known_recall=0.3485 | unknown_recall=0.8240 | unknown_precision=0.6525 | confusion=[[92, 172], [69, 323]]
thr=0.30 | bal_acc=0.5836 | known_recall=0.3636 | unknown_recall=0.8036 | unknown_precision=0.6522 | confusion=[[96, 168], [77, 315]]
thr=0.35 | bal_acc=0.5993 | known_recall=0.4205 | unknown_recall=0.7781 | unknown_precision=0.6659 | confusion=[[111, 153], [87, 305]]
thr=0.40 | bal_acc=0.6283 | known_recall=0.5985 | unknown_recall=0.6582 | unknown_precision=0.7088 | confusion=[[158, 106], [134, 258]]
thr=0.45 | bal_acc=0.6583 | known_recall=0.7197 | unknown_recall=0.5969 | unknown_precision=0.7597 | confusion=[[190, 74], [158, 234]]
thr=0.50 | bal_acc=0.6529 | known_recall=0.7727 | unknown_recall=0.5332 | unknown_precision=0.7770 | confusion=[[204, 60], [183, 209]]
thr=0.55 | bal_acc=0.6540 | known_recall=0.8182 | unknown_recall=0.4898 | unknown_precision=0.8000 | confusion=[[216, 48], [200, 192]]
thr=0.60 | bal_acc=0.6367 | known_recall=0.8371 | unknown_recall=0.4362 | unknown_precision=0.7991 | confusion=[[221, 43], [221, 171]]
thr=0.65 | bal_acc=0.6289 | known_recall=0.8598 | unknown_recall=0.3980 | unknown_precision=0.8083 | confusion=[[227, 37], [236, 156]]
thr=0.70 | bal_acc=0.6262 | known_recall=0.8826 | unknown_recall=0.3699 | unknown_precision=0.8239 | confusion=[[233, 31], [247, 145]]
thr=0.75 | bal_acc=0.6204 | known_recall=0.9091 | unknown_recall=0.3316 | unknown_precision=0.8442 | confusion=[[240, 24], [262, 130]]
thr=0.80 | bal_acc=0.6221 | known_recall=0.9432 | unknown_recall=0.3010 | unknown_precision=0.8872 | confusion=[[249, 15], [274, 118]]
```

## Cell 38

- execution_count: `75`
- overview/source: `# OVERVIEW: Đánh giá test bằng threshold đã tune, in accuracy/balanced accuracy/confusion matrix rồi lưu checkpoint.`

### Output 1

```text
Test metrics: {'loss': 0.8527102036909624, 'accuracy': 0.5691287878787878, 'known_recall': 0.6931818181818182, 'unknown_recall': 0.5277777777777778, 'unknown_precision': 0.8376753507014028, 'balanced_accuracy': 0.610479797979798, 'unknown_threshold': 0.45, 'known_total': 264, 'unknown_total': 792, 'predicted_unknown_total': 499, 'confusion_matrix': [[183, 81], [374, 418]]}
Confusion matrix format: [[known->known, known->unknown], [unknown->known, unknown->unknown]]
BEST_UNKNOWN_THRESHOLD=0.45; MAX_TEST_BATCHES=None; đặt None nếu cần evaluate toàn bộ test set.
Đã lưu model vào /content/drive/MyDrive/unlearning-artifacts/notebook/best_model.pt
Bỏ qua lưu embeddings.pt để cell test chạy nhanh. Đặt SAVE_EMBEDDINGS_AFTER_TRAIN=True nếu cần.
```

## Cell 39

- execution_count: `76`
- overview/source: `# OVERVIEW: Smoke test 5 PCAP từ test set bằng threshold đã tune: dự đoán và đối chiếu đúng/sai.`

### Output 1

```text
Smoke test trên 5 mẫu từ test_records, threshold=0.45
Format: OK? | true_binary | pred_binary | p_unknown | confidence | original_label | path
[1] OK    | true=unknown | pred=unknown | p_unknown=0.8093 | conf=0.8093 | original=xfinity_customer_service | path=xfinity_customer_service_40.pcap
[2] OK    | true=known   | pred=known   | p_unknown=0.3747 | conf=0.6253 | original=solitaire | path=solitaire_70.pcap
[3] OK    | true=unknown | pred=unknown | p_unknown=0.8801 | conf=0.8801 | original=social_security | path=social_security_104.pcap
[4] OK    | true=unknown | pred=unknown | p_unknown=0.6798 | conf=0.6798 | original=qr_code_generator | path=qr_code_generator_34.pcap
[5] OK    | true=known   | pred=known   | p_unknown=0.3529 | conf=0.6471 | original=soap2day | path=soap2day_66.pcap
Smoke accuracy: 5/5 = 1.0000
Lưu ý: smoke test 5 mẫu chỉ để kiểm tra trực quan; metric chính vẫn là Test metrics ở cell trước.
```

## Cell 40

- execution_count: `77`
- overview/source: `# OVERVIEW: Chạy thử instance-wise unlearning theo plan: quên Df và giữ Dr bằng retain loss + regularization.`

### Output 1

```text
Bỏ qua unlearning. Đặt RUN_UNLEARNING = True để chạy cell này.
```

## Cell 41

- execution_count: `78`
- overview/source: `# OVERVIEW: Tổng hợp runtime, dataset split, train metrics, test metrics và smoke test vào run_summary.json/md cho người và AI đọc.`

### Output 1

```text
Đã lưu summary JSON: /content/drive/MyDrive/unlearning-artifacts/notebook/run_summary.json
Đã lưu summary Markdown: /content/drive/MyDrive/unlearning-artifacts/notebook/run_summary.md
File này phù hợp để gửi/đọc lại nhanh thay vì đọc toàn bộ output trong .ipynb.
```
