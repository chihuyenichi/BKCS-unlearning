# Tổng quan các method unlearning: Amnesiac, Blindspot, BND/Boundary và MK-MMD

## Mục đích của tài liệu

Tài liệu này tổng hợp các phương pháp xuất hiện trong bảng kết quả mà bạn đưa:

- `Baseline`
- `Amnesiac`
- `Blindspot`
- `BND`
- `MK-MMD (proposal)`

Các method này có thể hiểu là các chiến lược thiết kế quá trình unlearning: chọn phần nào của model được cập nhật, dùng tín hiệu/loss nào để quên `D_f`, dùng tín hiệu/loss nào để giữ `D_r`, và có thêm ràng buộc/ngưỡng/regularization nào không.

Ký hiệu chung:

```text
D = D_r ∪ D_f;  D_r ∩ D_f = ∅
```

Trong đó:

- `D_f`: forget set, dữ liệu cần quên.
- `D_r`: retain set, dữ liệu cần giữ.
- `M_0`: model gốc trước unlearning.
- `M_u`: model sau unlearning.
- `M_r`: model retrain lại chỉ trên `D_r`; đây thường được xem là chuẩn tham chiếu mạnh nhất.

## 1. Baseline

`Baseline` là model gốc hoặc phương pháp mốc dùng để so sánh.

Trong nhiều paper unlearning, baseline có thể là:

- model chưa unlearn;
- fine-tuning đơn giản trên `D_r`;
- retrain từ đầu trên `D_r`;
- hoặc một cấu hình chuẩn mà paper chọn làm mốc.

Với bảng bạn đưa, `Baseline` nhiều khả năng là model ban đầu chưa áp dụng các kỹ thuật unlearning/adaptation đặc biệt. Nó cho biết hiệu năng gốc trên dataset như `BATADAL`, `WADI`, `SWaT`.

Điểm cần nhớ:

```text
Baseline không phải một phương pháp quên, mà là mốc để so sánh.
```

## 2. Amnesiac

Nguồn chính: **Amnesiac Machine Learning**, Graves, Nagisetty, Ganesh, AAAI 2021.

Ý tưởng chính: nếu trong quá trình train ta lưu lại các update mà dữ liệu gây ra cho model, thì khi cần quên một phần dữ liệu, ta có thể loại bỏ hoặc đảo ảnh hưởng của các update liên quan đến phần dữ liệu đó.

Nói đơn giản:

```text
Train model:
  lưu lại update do từng batch/từng shard gây ra

Unlearn D_f:
  tìm các update có liên quan đến D_f
  loại bỏ/đảo tác động của chúng khỏi model
```

Về bản chất, Amnesiac gần với cách nghĩ:

```text
M_u ≈ M_0 - ảnh_hưởng_của(D_f)
```

Điểm mạnh:

- Có ý tưởng trực tiếp: xóa ảnh hưởng học được từ dữ liệu cần quên.
- Có thể nhanh hơn retrain từ đầu nếu đã lưu đủ thông tin update.
- Paper nhấn mạnh mục tiêu giảm rò rỉ qua model inversion và membership inference.

Điểm yếu:

- Cần thiết kế quá trình train từ trước để lưu thông tin cần thiết.
- Nếu pipeline hiện tại không lưu update theo batch/mẫu, rất khó áp dụng đúng nguyên bản.
- Với deep model lớn, việc lưu update chi tiết có thể tốn dung lượng.

Liên hệ với bài toán PCAP:

- Nếu muốn áp dụng Amnesiac đúng nghĩa, khi train encoder/MLP cần lưu metadata về batch/update.
- Pipeline hiện tại của mình chưa làm điều này.
- Do đó hiện tại chưa phải Amnesiac; nó là fine-tuning unlearning bằng loss mới.

## 3. Blindspot / Bad Teaching

Nguồn chính: **Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher**, Chundawat et al., AAAI 2023.

Ý tưởng chính: dùng teacher-student để làm model quên. Student là model sau unlearning. Nó học từ hai teacher:

- competent/smart teacher: model gốc đã train tốt, dùng để giữ kiến thức trên `D_r`;
- incompetent/dumb teacher: model không có tri thức hữu ích, thường khởi tạo ngẫu nhiên, dùng để làm mất tri thức trên `D_f`.

Luồng hoạt động:

```text
Với x ∈ D_r:
  student học giống smart teacher

Với x ∈ D_f:
  student học giống dumb teacher
```

Loss thường là dạng knowledge distillation:

```text
L = KL(SmartTeacher(x_r) || Student(x_r))
  + KL(DumbTeacher(x_f)  || Student(x_f))
```

Điểm quan trọng:

```text
Blindspot/Bad Teaching không nhất thiết ép D_f -> unknown.
```

Nó làm prediction trên `D_f` giống một teacher không biết dữ liệu đó. Nếu hệ thống của mình muốn mọi PCAP thuộc nhãn bị quên trả về `unknown`, thì cần biến đổi phương pháp:

```text
D_f target = unknown
D_r target = nhãn gốc hoặc output của model gốc
```

Điểm mạnh:

- Không cần lưu update trong quá trình train như Amnesiac.
- Dùng distillation nên có cơ chế giữ hành vi của model gốc trên `D_r`.
- Phù hợp cả class-level và subset-level unlearning.

Điểm yếu:

- Cần thiết kế teacher.
- Kết quả phụ thuộc chất lượng smart teacher, dumb teacher và cách cân bằng `D_f/D_r`.
- Nếu dùng dumb teacher ngẫu nhiên, output sau quên không chắc là `unknown`.

Liên hệ với bài toán PCAP:

- Có thể dùng model gốc làm smart teacher cho `Dr`.
- Có thể dùng model khởi tạo ngẫu nhiên, model yếu, hoặc target `unknown` làm blindspot signal cho `Df`.
- Đây là hướng gần với pipeline hiện tại nếu thay cross-entropy bằng distillation loss.

## 4. BND / Boundary-based unlearning

Lưu ý: trong tài liệu công khai mình tìm được, ký hiệu `BND` không phải tên viết tắt thống nhất rộng rãi như `Amnesiac` hay `MMD`. Do đó cần đối chiếu với paper gốc chứa bảng kết quả bạn đưa. Tuy vậy, trong literature unlearning gần nhất, method có vai trò tương ứng thường là nhóm **Boundary Unlearning**, gồm `Boundary Shrink` và `Boundary Expanding`.

Nguồn liên quan: **Boundary Unlearning**, Chen et al., CVPR 2023.

Ý tưởng chính: thay vì nhìn unlearning như việc “xóa ảnh hưởng trong không gian tham số”, phương pháp này nhìn vào **decision boundary**. Mục tiêu là dịch chuyển boundary để dữ liệu thuộc class cần quên không còn nằm trong vùng quyết định của class đó.

Luồng tư duy:

```text
Trước unlearning:
  D_f nằm trong vùng quyết định của class cần quên

Sau unlearning:
  boundary được dịch chuyển
  D_f bị đẩy sang vùng class khác hoặc vùng shadow/unknown
```

Hai biến thể thường gặp:

```text
Boundary Shrink:
  thu hẹp vùng quyết định của class cần quên
  bằng cách gán mẫu cần quên sang class lân cận/sai phù hợp

Boundary Expanding:
  mở rộng/đẩy vùng quyết định bằng class phụ hoặc shadow class
  để phá liên kết giữa D_f và class gốc
```

Nếu bảng dùng `BND` với nghĩa boundary/bounded method, thì có thể hiểu nó là:

```text
cập nhật trọng số theo hướng dịch decision boundary
thay vì chỉ tăng/giảm loss một cách thô
```

Điểm mạnh:

- Trực quan cho class-level unlearning.
- Không nhất thiết phải cập nhật toàn bộ tham số theo kiểu phá mạnh.
- Có thể nhanh hơn retrain.

Điểm yếu:

- Phù hợp nhất với class-wise forgetting.
- Cần biết class cần quên và có cách xác định class lân cận/target thay thế.
- Với bài toán binary `known/unknown`, nếu class cần quên là một nhãn con trong AOL nhưng classifier chỉ có output `known/unknown`, boundary-level unlearning cần thiết kế lại cẩn thận.

Liên hệ với bài toán PCAP:

- Nếu dùng binary classifier `known/unknown`, boundary method có thể được đơn giản hóa thành đẩy `Df` qua biên sang `unknown`.
- Nếu dùng multiclass classifier, boundary shrink có thể chọn target là `unknown` hoặc class gần nhất.
- Đây là hướng hợp lý nếu bài toán muốn quên trọn một hoặc nhiều nhãn folder cha.

## 5. MMD và MK-MMD

Nguồn nền tảng: **A Kernel Two-Sample Test**, Gretton et al., JMLR 2012.

MMD là viết tắt của **Maximum Mean Discrepancy**. Nó đo khoảng cách giữa hai phân phối thông qua kernel/RKHS.

Nói đơn giản:

```text
MMD(P, Q) nhỏ  -> hai phân phối P và Q giống nhau hơn
MMD(P, Q) lớn  -> hai phân phối P và Q khác nhau hơn
```

Trong deep learning, MMD thường được dùng để căn chỉnh phân phối feature giữa hai domain hoặc hai nhóm dữ liệu.

`MK-MMD` là **Multi-Kernel MMD**: thay vì dùng một kernel duy nhất, nó kết hợp nhiều kernel để đo khác biệt phân phối ở nhiều thang khác nhau. Một nguồn nổi tiếng dùng MK-MMD trong deep network là **Learning Transferable Features with Deep Adaptation Networks**, Long et al., ICML 2015.

Trong unlearning, MK-MMD có thể được dùng như một hàm phạt/regularization:

```text
loss = loss_quên + loss_giữ + λ * loss_MK-MMD
```

Tùy paper cụ thể, `loss_MK-MMD` có thể có các mục tiêu khác nhau:

- kéo feature của model sau unlearning gần phân phối retrain/retain mong muốn;
- tách phân phối feature của `D_f` khỏi class gốc;
- căn chỉnh output/feature giữa model unlearned và model tham chiếu trên `D_r`;
- giảm lệch phân phối khi có domain shift.

Điểm mạnh:

- Không chỉ sửa nhãn đầu ra, mà còn tác động lên phân phối representation.
- Có thể giúp model sau unlearning ổn định hơn, tránh phá feature quá mạnh.
- Phù hợp khi dữ liệu có đặc trưng phân phối rõ, như traffic/network/ICS time series.

Điểm yếu:

- Cần chọn feature layer để đo MMD.
- Cần chọn kernel, bandwidth, trọng số `λ`.
- Nếu chọn sai phân phối mục tiêu, model có thể “giữ nhầm” thông tin cần quên.

Liên hệ với bài toán PCAP:

- Encoder tạo vector đặc trưng cho mỗi file/flow.
- Có thể tính MK-MMD giữa feature của `D_f` sau unlearning và feature của `unknown`, để đẩy `D_f` giống `unknown`.
- Đồng thời có thể giữ feature của `D_r` gần với feature từ model gốc, để tránh làm hỏng phần retain.

Một loss gợi ý cho dự án:

```text
loss = CE(Df -> unknown)
     + α * CE(Dr -> nhãn gốc)
     + β * MKMMD(feature(Df), feature(unknown_reference))
     + γ * MKMMD(feature_new(Dr), feature_old(Dr))
```

Trong đó:

- `CE(Df -> unknown)`: ép dữ liệu cần quên sang unknown.
- `CE(Dr -> nhãn gốc)`: giữ phân loại trên retain set.
- `MKMMD(feature(Df), feature(unknown_reference))`: kéo feature của dữ liệu cần quên về phân phối unknown.
- `MKMMD(feature_new(Dr), feature_old(Dr))`: giữ representation của retain set không trôi quá xa model gốc.

## 6. So sánh nhanh các method

| Method | Cập nhật trọng số dựa trên gì? | Có cần `D_f`? | Có cần `D_r`? | Gần với pipeline hiện tại không? |
|---|---|---:|---:|---|
| Baseline | Không phải unlearning method | Không | Không | Là mốc so sánh |
| Amnesiac | Đảo/loại bỏ update liên quan `D_f` | Có | Không bắt buộc | Chưa gần, vì pipeline chưa lưu update khi train |
| Blindspot/Bad Teaching | Distillation từ smart/dumb teacher | Có | Thường có | Khá gần nếu thay CE bằng teacher-student loss |
| BND/Boundary | Dịch decision boundary khỏi class cần quên | Có | Thường có | Gần nếu xem `unknown` là vùng cần đẩy sang |
| MK-MMD | CE + ràng buộc phân phối feature/output | Có | Có hoặc cần reference distribution | Rất phù hợp để mở rộng encoder/feature pipeline |

## 7. Kết luận cho hướng triển khai hiện tại

Pipeline hiện tại của dự án đang dùng:

```text
loss = CE(Df -> unknown) + CE(Dr -> nhãn gốc)
```

Nó là baseline unlearning đơn giản, cập nhật bằng gradient descent. Nó chưa phải Amnesiac, chưa phải Blindspot nguyên bản, chưa phải Boundary Unlearning, và chưa có MK-MMD.

Nếu muốn mở rộng theo các method trong bảng, thứ tự hợp lý là:

1. Giữ baseline hiện tại làm `CE-only`.
2. Thêm `Boundary/BND-like`: ép `Df` qua biên sang `unknown`, có thể thêm margin hoặc threshold.
3. Thêm `Blindspot/Bad Teaching`: dùng model gốc làm smart teacher cho `Dr`, dùng dumb teacher hoặc unknown-target teacher cho `Df`.
4. Thêm `MK-MMD`: ràng buộc feature của `Df` giống phân phối unknown và feature của `Dr` gần model gốc.
5. Chỉ triển khai `Amnesiac` nếu thay đổi pipeline train để lưu update theo batch/mẫu từ đầu.

## 8. Tài liệu tham khảo

- Graves, Nagisetty, Ganesh. **Amnesiac Machine Learning**. AAAI 2021. <https://ojs.aaai.org/index.php/AAAI/article/view/17371>
- Chundawat, Tarun, Mandal, Kankanhalli. **Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher**. AAAI 2023. <https://arxiv.org/abs/2205.08096>
- Chen, Gao, Liu, Peng, Wang. **Boundary Unlearning**. CVPR 2023. <https://arxiv.org/abs/2303.11570>
- Gretton et al. **A Kernel Two-Sample Test**. JMLR 2012. <https://jmlr.org/papers/v13/gretton12a.html>
- Long, Cao, Wang, Jordan. **Learning Transferable Features with Deep Adaptation Networks**. ICML 2015. <https://proceedings.mlr.press/v37/long15.html>
