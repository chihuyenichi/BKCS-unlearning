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

- Chuẩn bị dependency tối thiểu để parse PCAP và train PyTorch pipeline.
- Cấu hình dữ liệu, checkpoint Drive-first, feature extraction, encoder, MLP và training.
- Nạp thư viện, mount Drive khi cần, tạo thư mục output/cache và chọn thiết bị train.
- Định nghĩa record dữ liệu, seed, JSON helper, scan PCAP và split label ở cấp class.
- Parse PCAP thành tensor [MAX_PACKETS, 3] và cache feature theo file/version.
- Định nghĩa Dataset/DataLoader cho binary classifier và SupCon original-label training.
- Định nghĩa DF-style encoder, MLP classifier và FlowModel end-to-end.
- Định nghĩa metric, evaluate, class weights và checkpoint helper dùng chung.
- Định nghĩa SupCon loss/wrapper, checkpoint encoder và pretrain encoder known-only.
- Train MLP binary classifier, lưu latest/best checkpoint và hỗ trợ resume/warm-start.
- Quét dữ liệu, tạo label split, tạo records/DataLoader và khởi tạo model.
- Chạy giai đoạn 1: SupCon known-only, lưu encoder checkpoint, freeze encoder, train MLP binary.
- Sweep threshold trên validation, evaluate test, lưu best_model và run_summary.
- Khung giai đoạn 2 instance-wise unlearning; mặc định chưa chạy chính thức.

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
