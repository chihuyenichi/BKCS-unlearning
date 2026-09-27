# Zero-Shot Machine Unlearning

## Thông tin nguồn

- PDF gốc: `Chundawat et al. - 2023 - Zero-Shot Machine Unlearning.pdf`
- Tác giả: Vikram S. Chundawat, Ayush K. Tarun, Murari Mandal, Mohan Kankanhalli.
- Bản arXiv trong thư mục: 2023.
- Phạm vi tài liệu này: bản tổng quan tiếng Việt, không phải bản dịch toàn văn.

## Bài toán zero-shot unlearning

Paper nghiên cứu trường hợp cực đoan: khi nhận yêu cầu xóa, tổ chức **không còn quyền truy cập cả dữ liệu retain lẫn payload của dữ liệu cần quên**. Chỉ model đã train và yêu cầu quên class `C_f` còn sẵn có.

Ký hiệu:

```text
D = D_r ∪ D_f;  theta = A(D);  theta_r = A(D_r)
```

- `D_f`: dữ liệu thuộc các forget class `C_f`.
- `D_r`: dữ liệu thuộc retain classes `C_r`.
- `M(x; theta)`: model gốc.
- `M_r(x; theta_r)`: model retrain chỉ với `D_r`, là gold model tham chiếu.
- `M_u(x; theta_u)`: model sau unlearning.

Mục tiêu hành vi:

```text
M_u(x; theta_u) ≈ M_r(x; theta_r)
```

Không đòi hỏi tham số `theta_u` phải giống `theta_r`; điều cần giống là hành vi black-box/phân phối output.

Khác biệt quyết định của zero-shot setting là thuật toán unlearn không dùng `D_r` hay `D_f` để cập nhật model. Dữ liệu gốc chỉ có thể được dùng sau đó cho evaluation, nếu tổ chức có môi trường đánh giá tách biệt.

## Hai phương pháp đề xuất

### 1. Error minimization–maximization noise (Min–Max)

Paper dùng noise/pseudo-input được tối ưu để làm tăng sai khác giữa model gốc và model đang unlearn, sau đó cập nhật model nhằm giảm sai khác đó. Cơ chế error maximization và error minimization này hướng model tới việc làm suy yếu thông tin về forget class mà không cần sample thật.

Đây là cách data-free nhưng có nguy cơ làm hỏng retain utility vì pseudo-data không chứa thông tin retain thật.

### 2. Gated Knowledge Transfer (GKT)

GKT là hướng teacher–student data-free:

```text
Model gốc          → teacher
Model khởi tạo mới → student
Noise z            → generator → pseudo-sample xp
Teacher output     → band-pass filter loại xác suất forget classes
Thông tin retain   → truyền sang student
```

Generator tạo pseudo-samples làm teacher và student bất đồng; student học từ teacher qua knowledge distillation. Band-pass filter chặn các thành phần output thuộc forget classes, để student chỉ nhận tri thức về retain classes. Đây là cơ chế trung tâm để “không truyền lại” kiến thức của class cần quên.

## Đánh giá

Paper đánh giá trên MNIST, SVHN và CIFAR-10 với nhiều backbone. Các tiêu chí chính:

- accuracy trên forget set: mong gần hành vi của retrain model;
- accuracy trên retain set: cần được giữ lại;
- so sánh với retrain model bằng black-box metrics, không so sánh trực tiếp trọng số;
- **Anamnesis Index (AIN)**: đánh giá tốc độ/khả năng relearn forget classes, nhằm phản ánh lượng thông tin còn sót;
- đánh giá model inversion và membership inference để xem thông tin forget data còn lộ không.

Paper chỉ ra rằng làm accuracy trên forget set thấp đơn thuần có thể đồng thời làm giảm chất lượng retain set; vì vậy cần đánh giá nhiều chiều.

## Ý nghĩa cho dự án PCAP

Setting này chỉ phù hợp khi sau train bạn thật sự không giữ được cache PCAP, normalized data, retain subset hay forget files. Điều đó khác với kế hoạch hiện tại, vốn lưu dữ liệu chuẩn hóa/cache trên Drive để train và unlearn.

Do đó, với pipeline hiện tại nên xem paper này như:

- một baseline/setting nghiên cứu bổ sung nếu muốn chứng minh data-free unlearning;
- nguồn ý tưởng về teacher–student và việc chặn knowledge của forget class;
- lời nhắc rằng checkpoint, label map và metadata cần được quản lý độc lập với raw PCAP nếu sau này muốn thử zero-shot.

Không nên gọi workflow hiện tại là zero-shot nếu nó vẫn dùng `D_f`, retain cache hoặc PCAP thật trong bước unlearning.

## Kết luận ngắn

Đóng góp chính của paper là formalize **unlearning không dữ liệu**: chỉ có model gốc và danh sách class cần quên. GKT là phương pháp đáng chú ý nhất vì cố truyền retain knowledge bằng pseudo-data trong khi chặn forget-class probabilities. Đổi lại, setting này khó hơn đáng kể và thường giữ retain utility kém hơn setting có `D_r` thật.
