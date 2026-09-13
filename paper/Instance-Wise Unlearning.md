# Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers

## 1. Bài toán
Paper này bàn về **machine unlearning**: xóa thông tin của một số mẫu cụ thể khỏi mô hình đã huấn luyện sẵn mà **không cần train lại từ đầu**.

Khác với nhiều hướng trước đó chỉ xóa theo **lớp** hoặc cần truy cập lại toàn bộ dữ liệu train, bài này xét **instance-wise unlearning**:
- chỉ cần mô hình tiền huấn luyện và tập mẫu cần quên `Df`
- các mẫu cần quên có thể thuộc nhiều lớp khác nhau
- mục tiêu là làm cho các mẫu đó bị **phân loại sai** hoặc được **gán sang nhãn khác**, đồng thời giữ hiệu năng trên phần dữ liệu còn lại `Dr`

## 2. Ý tưởng chính
Tác giả cho rằng chỉ tối ưu để “quên” bằng gradient âm thì mô hình sẽ bị suy giảm mạnh trên dữ liệu còn lại. Vì vậy, họ đề xuất thêm các **regularization** để giảm hiện tượng quên ngoài ý muốn.

Hai hướng regularization:
1. **Dùng adversarial examples** sinh từ các mẫu cần quên để giữ ổn định biên quyết định.
2. **Dùng weight importance** để chỉ cập nhật mạnh những tham số thật sự liên quan đến mẫu cần quên.

## 3. Mục tiêu unlearning
Gọi mô hình sau khi unlearning là `ĝθ`.

Paper xem 2 kiểu mục tiêu:
- **Misclassification**: ép `ĝθ(xf) != yf`
- **Relabeling / correction**: ép `ĝθ(xf) = yf*`, với `yf* != yf`

Nếu chỉ tối ưu trực tiếp mục tiêu này thì mô hình rất dễ quên cả `Dr`. Vì thế họ thêm thành phần giữ kiến thức còn lại:

`L_UL = L_quên + regularization`

## 4. Hai phương pháp đề xuất

### 4.1. Regularization bằng adversarial examples
Với mỗi mẫu cần quên, tác giả sinh ra nhiều adversarial examples bằng targeted PGD. Các mẫu này mang thông tin về ranh giới quyết định của mô hình gốc.

Trong quá trình unlearning, họ thêm loss trên các adversarial examples vào bài toán tối ưu. Tác dụng chính:
- giữ lại cấu trúc quyết định đã học
- giảm việc “phá nát” biểu diễn ở các lớp ẩn
- giúp mô hình vẫn phân loại tốt trên `Dr` và `Dtest`

### 4.2. Regularization bằng weight importance
Tác giả dùng **MAS (Memory Aware Synapses)** để đo độ quan trọng của từng tham số đối với các mẫu cần quên.

Ý tưởng:
- tham số quan trọng thì cho phép thay đổi nhiều hơn
- tham số ít liên quan thì bị giữ lại chặt hơn

Cách này giúp mô hình chỉ sửa những phần thật sự cần thiết cho việc quên, thay vì cập nhật lan rộng toàn bộ mạng.

### 4.3. Kết hợp hai cách
Phương án tốt nhất của paper là kết hợp cả hai:
- adversarial examples để giữ biên quyết định
- weight importance để điều khiển vùng tham số cần cập nhật

## 5. Thiết lập thí nghiệm
Tác giả thử trên nhiều bài toán ảnh:
- **CIFAR-10**
- **CIFAR-100**
- **ImageNet-1K**
- **UTKFace**
- **ImageNet-A** cho bài toán sửa các ảnh tự nhiên dễ bị phân loại sai

Mô hình nền:
- `ResNet-18` cho CIFAR-10
- `ResNet-50` cho CIFAR-100 và ImageNet-1K

Baseline so sánh:
- **Before**: mô hình trước unlearning
- **Oracle**: retrain lại với dữ liệu còn lại
- **NegGrad**: dùng gradient âm
- **RAWP**: perturb trọng số
- **Adv**: phương pháp dùng adversarial examples
- **Adv + Imp**: kết hợp adversarial examples và weight importance

## 6. Kết quả chính
### 6.1. Với bài toán quên instance-wise
- **NegGrad** có thể quên mẫu cần xóa nhưng làm giảm rất mạnh accuracy trên `Dr` và `Dtest`
- **Adv** giữ hiệu năng tốt hơn rõ rệt so với NegGrad
- **Adv + Imp** thường là tốt nhất, nhiều trường hợp còn **vượt Oracle**

### 6.2. Với UTKFace
- Khi quên nhiều mẫu, NegGrad làm accuracy giảm rất nhanh, thậm chí thấp hơn mức ngẫu nhiên
- Hai phương pháp đề xuất vẫn giữ hiệu năng ổn hơn trên phần dữ liệu còn lại

### 6.3. Với continual unlearning
Khi dữ liệu cần quên đến theo từng đợt nhỏ liên tiếp, phương pháp của paper vẫn ổn định hơn NegGrad, cho thấy khả năng xử lý **stream of deletion requests**.

### 6.4. Với ImageNet-A
Phương pháp của paper có thể sửa các ảnh tự nhiên bị phân loại sai, đồng thời vẫn giữ được độ chính xác trên dữ liệu còn lại. Tuy nhiên, regularization quá mạnh có thể làm giảm hiệu quả khi số ảnh cần sửa quá lớn.

## 7. Phân tích định tính
Tác giả rút ra 3 quan sát quan trọng:
- Không có dấu hiệu rõ ràng rằng mô hình dùng một nhãn cố định để “giấu” các mẫu bị quên
- Adversarial examples giúp giữ lại **decision boundary** tốt hơn
- Việc quên chủ yếu diễn ra ở **high-level features**, còn đặc trưng thấp tầng vẫn gần giống mô hình gốc

## 8. Đóng góp chính
- Đề xuất hướng **instance-wise unlearning** cho mô hình tiền huấn luyện
- Chỉ cần mô hình gốc và tập mẫu cần quên, không cần truy cập toàn bộ train set
- Đưa ra 2 regularization hiệu quả để giảm quên ngoài ý muốn
- Thực nghiệm trên nhiều bộ dữ liệu cho thấy phương pháp giữ tốt hiệu năng trên dữ liệu còn lại

## 9. Kết luận ngắn
Paper này cho thấy có thể làm **unlearning theo từng mẫu** mà vẫn giữ được chất lượng mô hình nếu không chỉ “quên bằng cách phá hủy” mà còn thêm regularization hợp lý. Trong đó, adversarial examples và weight importance là hai thành phần cốt lõi giúp mô hình quên đúng phần cần quên, nhưng vẫn bảo toàn tri thức hữu ích còn lại.
