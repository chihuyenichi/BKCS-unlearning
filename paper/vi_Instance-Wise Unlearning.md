# Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers

## Thông tin nguồn

- PDF gốc: `Instance-Wise Unlearning.pdf`
- Tác giả: Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, Moontae Lee.
- Hội nghị: AAAI 2024.
- Phạm vi tài liệu này: bản tổng quan tiếng Việt, không phải bản dịch toàn văn.

## Vấn đề paper giải quyết

Paper nghiên cứu **instance-wise unlearning**: xóa ảnh hưởng của một tập mẫu cụ thể `D_f` khỏi classifier đã train, kể cả khi các mẫu cần xóa nằm lẫn trong nhiều class. Đây khác với class-wise unlearning, nơi toàn bộ dữ liệu của một class bị xóa.

Ký hiệu:

```text
D_train = D_r ∪ D_f;  D_r = D_train \ D_f
```

- `D_f`: các mẫu được yêu cầu quên.
- `D_r`: các mẫu còn lại cần duy trì chất lượng dự đoán.
- `g_theta`: classifier trước unlearning.
- `g_hat_theta`: classifier sau unlearning.

Giả định quan trọng của paper: tại thời điểm unlearn, chỉ cần model đã train và các mẫu trong `D_f`; không cần tải lại toàn bộ `D_r`.

## Mục tiêu unlearning

Thay vì chỉ cố mô phỏng model retrain không có `D_f`, paper chọn mục tiêu hành vi mạnh hơn: các mẫu cần quên phải không còn được dự đoán đúng nhãn cũ.

Hai biến thể được nêu:

```text
g_hat_theta(x_f) != y_f
```

hoặc gán từng mẫu sang nhãn mục tiêu khác:

```text
g_hat_theta(x_f) = y_f*;  y_f* != y_f
```

Vì thế paper **không yêu cầu** output phải là `unknown`. `unknown` chỉ là một lựa chọn relabeling có chủ đích nếu hệ thống có class/reject state này.

Mục tiêu đồng thời là giữ performance trên `D_r` và tập test. Nếu chỉ tối ưu để làm sai `D_f`, model có thể mất kiến thức trên các dữ liệu còn lại.

## Phương pháp chính

### 1. Unlearning bằng misclassification/relabeling

Model được fine-tune để tăng loss với nhãn cũ của `D_f`, hoặc tối ưu prediction về nhãn thay thế. Negative-gradient thuần túy có thể làm hỏng decision boundary rộng hơn, nên paper bổ sung regularization.

### 2. Regularization ở representation level bằng adversarial examples

Với mỗi mẫu cần quên, paper tạo adversarial example có nhãn target khác nhãn gốc bằng targeted PGD. Các adversarial example này được dùng như regularizer để model vẫn giữ được decision boundary/đại diện đặc trưng của dữ liệu còn lại, trong khi vẫn ép mẫu `D_f` bị phân loại sai.

### 3. Regularization theo độ quan trọng của trọng số

Paper dùng **MAS (Memory Aware Synapses)** để ước lượng trọng số nào quan trọng cho việc dự đoán `D_f`. Khi unlearn, các trọng số liên quan mạnh tới việc nhận đúng `D_f` được cho phép thay đổi nhiều hơn; các trọng số ít liên quan bị phạt nếu thay đổi, nhằm giảm mất utility không cần thiết.

## Đánh giá

Paper kiểm tra trên CIFAR-10, CIFAR-100, ImageNet-1K và một thí nghiệm nhận dạng độ tuổi khuôn mặt. Họ đo:

- độ chính xác trên `D_f`: cần giảm, với mục tiêu misclassification là gần 0%;
- độ chính xác trên `D_r` và test set: cần được giữ lại;
- so sánh với model oracle/retrain và các baseline như negative gradient;
- mức độ lộ thông tin qua các thí nghiệm liên quan đến privacy.

Kết quả thực nghiệm cho thấy negative gradient đơn thuần có thể quên `D_f` nhưng làm giảm mạnh accuracy trên `D_r`; regularization bằng adversarial examples và MAS giữ lại utility tốt hơn.

## Ý nghĩa cho dự án PCAP

Paper gần với tình huống sau:

```text
Model PCAP đã train + danh sách file PCAP cần xóa
→ fine-tune unlearning
→ các file đó không còn được nhận đúng label cũ
→ chất lượng trên PCAP retain vẫn giữ được
```

Áp dụng trực tiếp cần thận trọng:

- Adversarial perturbation trên ảnh không thể bê nguyên sang chuỗi đặc trưng traffic `[time, direction, size]`; cần thiết kế perturbation vẫn hợp lệ về ngữ nghĩa packet traffic, hoặc bỏ thành phần này.
- Với binary `known/unknown`, đặt `known → unknown` là một ví dụ của relabeling có chủ đích.
- Với multiclass, có thể chọn một target class khác, class `unknown` riêng, hoặc chỉ đặt mục tiêu không còn dự đoán nhãn cũ. Ba lựa chọn có ý nghĩa đánh giá khác nhau.
- Cần lập retrain baseline trên `D_r` để biết unlearned model có thực sự gần với việc “chưa từng học” `D_f` hay không.

## Kết luận ngắn

Đóng góp cốt lõi là chuyển trọng tâm từ “xóa ảnh hưởng theo tham số” sang “cưỡng bức hành vi quên trên từng mẫu”, đồng thời dùng regularization để không phá hỏng tri thức còn lại. Đây là paper phù hợp nhất trong ba bài nếu dự án cần quên một danh sách PCAP cụ thể thay vì xóa trọn class.
