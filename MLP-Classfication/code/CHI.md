# CHI - Cell Overview

File này tổng hợp lần lượt các dòng `# OVERVIEW:` của code cell trong các notebook thuộc `MLP-Classfication/code/`.

Quy ước bảo trì:

- Không ghi số thứ tự hoặc cell index để tránh phải cập nhật hàng loạt khi thêm/xóa cell.
- Khi thêm code cell, thêm một dòng overview mới vào đúng vị trí tương ứng.
- Khi xóa code cell, xóa dòng overview tương ứng.
- Khi đổi mục đích code cell, sửa dòng `# OVERVIEW:` trong notebook và dòng tương ứng ở file này.
- Với `train_pipeline.ipynb`, mỗi code cell phải bắt đầu bằng `# OVERVIEW:`.
- Với `supcon-model.ipynb`, các cell chưa có overview chuẩn được ghi chú để chuẩn hóa sau.

## train_pipeline.ipynb

Notebook chính của project. Pipeline: raw network flow `.pcap` -> packet features -> encoder -> embedding 256 chiều -> MLP classifier `known/unknown` -> đánh giá open-world -> instance-wise unlearning.

- Chuẩn bị dependency runtime cho Colab trước khi đọc PCAP và tạo DataLoader. Trong plan, input raw là các file `.pcap`; để parse PCAP cần Scapy. Cell này tự cài package còn thiếu để các cell sau có thể chạy theo thứ tự từ trên xuống.
- Cấu hình một mode duy nhất: Google Colab mount Google Drive, đọc PCAP baseline, ánh xạ nhãn gốc của dataset thành bài toán nhị phân `known/unknown` theo plan open-world.
- Nạp thư viện dùng chung, kiểm tra CUDA và chọn `DEVICE` CPU/GPU an toàn cho pipeline Colab.
- Mỗi `FlowRecord` lưu path PCAP, nhãn gốc từ folder, và binary label `known=0/unknown=1`.
- Cố định seed để split train/validation/test và quá trình train có thể lặp lại.
- Helper đọc danh sách nhãn dạng CSV, dùng cho CLI/phần tái sử dụng ngoài notebook.
- Suy luận nhãn gốc từ folder cha trực tiếp của PCAP, vì dataset đã có nhãn theo thư mục.
- Quét đệ quy `DATA_DIR` để lấy các raw flow `.pcap/.cap`; mỗi file là một sample theo plan.
- Đọc packet IPv4/IPv6 từ PCAP để lấy timestamp, IP, port và kích thước packet thô.
- Chuyển raw flow thành tensor `[MAX_PACKETS, 5]` và mask; đây là input trực tiếp của encoder.
- Ánh xạ nhãn gốc sang `known/unknown`; không gán ngẫu nhiên nhãn, chỉ dùng cấu hình trong cell đầu.
- Chia train/validation/test; holdout unknown không vào train nhưng một phần vào validation để calibrate threshold open-world.
- Dataset PyTorch parse PCAP theo kiểu lazy và cache tensor để cell tạo DataLoader không bị treo.
- Encoder CNN 1D nén chuỗi packet thành vector đặc trưng 256 chiều như mô tả trong plan.
- MLP 5 hidden layers nhận embedding 256 chiều và xuất 2 logits: `known` và `unknown`.
- `FlowModel` ghép encoder và MLP thành pipeline end-to-end PCAP -> embedding -> `known/unknown`.
- Tạo DataLoader lazy; PCAP chưa parse hàng loạt tại bước này mà parse khi batch được đọc.
- Đưa batch packet tensor, mask và label lên CPU/GPU đã chọn trong cell cấu hình.
- Đánh giá loss, accuracy, balanced accuracy, recall từng lớp, threshold và confusion matrix cho `known/unknown`.
- Huấn luyện encoder + MLP bằng weighted `CrossEntropyLoss` và chọn checkpoint theo metric cân bằng cho open-world.
- Ước lượng độ quan trọng tham số trên retain set, dùng cho instance-wise unlearning.
- Regularization giữ tham số quan trọng gần model gốc để bảo toàn `Dr` khi unlearning.
- Instance-wise unlearning: flip target của `Df`, đồng thời giữ hiệu năng trên `Dr` bằng retain loss.
- Trích xuất embedding 256 chiều; dùng cache để tránh parse lại PCAP khi lưu feature vector.
- Lưu embedding, nhãn `known/unknown`, nhãn gốc và path PCAP; dùng lại cache packet features.
- Lưu checkpoint model cùng cấu hình dataset/nhãn để tái lập thí nghiệm unlearning.
- Nạp checkpoint đã train để đánh giá lại hoặc chạy instance-wise unlearning.
- Chọn `Df` theo file hoặc nhãn; `Df` là instance cần quên, `Dr` là phần còn lại cần giữ.
- Lưu metrics/history ra JSON để so sánh baseline, open-world và unlearning.
- Quét toàn bộ nhãn gốc trong `DATA_DIR` trước khi lọc, lưu `label_inventory.json` để biết dataset có bao nhiêu class.
- Random split nhãn ở cấp class vào `known/unknown-train/holdout`, lưu `label_split` theo seed, số nhãn và ratio để tái lập thí nghiệm.
- Tạo `FlowRecord` sau label split và giới hạn số PCAP mỗi nhãn để quick debug không phải parse quá nhiều flow.
- Split dữ liệu và tạo DataLoader lazy; validation gồm cả unknown-train và holdout-unknown để tune threshold.
- Khởi tạo mô hình đúng pipeline trong plan: encoder 256 chiều + MLP output `known/unknown`.
- Train end-to-end encoder và MLP trên known + unknown-train; dùng class weights và balanced accuracy để giảm lệch về known.
- Sweep threshold unknown trên validation để chọn ngưỡng tốt nhất theo balanced accuracy trước khi test.
- Đánh giá test bằng threshold đã tune, in accuracy/balanced accuracy/confusion matrix rồi lưu checkpoint.
- Smoke test 5 PCAP từ test set bằng threshold đã tune: dự đoán và đối chiếu đúng/sai.
- Chạy thử instance-wise unlearning theo plan: quên `Df` và giữ `Dr` bằng retain loss + regularization.
- Tổng hợp runtime, dataset split, train metrics, test metrics và smoke test vào `run_summary.json/md` cho người và AI đọc.

## supcon-model.ipynb

Notebook cũ để tham khảo xử lý traffic và thử mô hình embedding. Notebook này chưa được chuẩn hóa theo format `# OVERVIEW:` ở tất cả cell.

- Chưa có `# OVERVIEW:` chuẩn; dòng đầu là code/import.
- Chưa chuẩn `# OVERVIEW:`; comment hiện tại: `Cell 3: Extract Labels from Filenames`.
- Chưa chuẩn `# OVERVIEW:`; comment hiện tại: `Cell 5: Helper Functions - Get Local IP`.
- Chưa chuẩn `# OVERVIEW:`; comment hiện tại: `from scipy import stats`.
- Chưa có nội dung code.
- Chưa có nội dung code.
- Chưa có `# OVERVIEW:` chuẩn; dòng đầu là code/import.
- Chưa có `# OVERVIEW:` chuẩn; dòng đầu là code/import.
- Chưa chuẩn `# OVERVIEW:`; comment hiện tại là dòng phân tách.
- Chưa có `# OVERVIEW:` chuẩn; dòng đầu là code/import.
- Chưa có nội dung code.
- Chưa có nội dung code.
