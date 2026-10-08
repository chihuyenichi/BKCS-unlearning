# Output và báo cáo từ các lần chạy VM

Folder này lưu output và báo cáo thí nghiệm trên VM. Báo cáo `unlearning_report.md` dùng protocol group-wise, quên 1/4/8 nhãn AOL; nguồn số liệu là `vm_outputs.json`.

| File | Nội dung |
| --- | --- |
| `vm_outputs.json` | Snapshot source/output của một lần chạy trên VM; chứa log và thông tin nguồn. |
| `unlearning_report.md` | Báo cáo tiếng Việt cho group-wise unlearning với 1/4/8 nhãn, dựa trên snapshot trên. |
| `unlearning_report_readable.csv` | Định nghĩa, cấu hình, kết quả trước/sau và aggregate của cùng lần chạy; có dòng trống giữa các section. |
| `last_run_results_summary.csv` | Kết quả của thí nghiệm single-label. |
| `extract_vm_outputs.py` | Xuất output đã lưu trong một notebook thành JSON; không chạy training/unlearning. |

Đường dẫn tuyệt đối trong snapshot ghi nhận vị trí của lần chạy. Báo cáo và CSV là dữ liệu dẫn xuất: khi thấy sai khác, đối chiếu log gốc. Aggregate forget/retain-balanced được lấy từ dòng tổng hợp đã in trong log; các metric chỉ có log từng repeat được tính từ số đã làm tròn, nên độ chính xác giới hạn ở output đã lưu.

## Xuất kết quả instance-wise trên VM

Lưu notebook bằng Jupyter sau khi chạy, rồi từ repo root thực hiện:

```bash
python MLP-Classfication/vm_code/out_data/extract_vm_outputs.py
```

Mặc định script đọc `instance_wise/train_instance_wise.ipynb` và ghi `<repo>/artifacts/vm-training/instance-wise/notebook_outputs.json`. Script xác định path theo vị trí chính nó, nên cũng chạy được từ working directory khác. Mặc định này không ghi đè snapshot group-wise `out_data/vm_outputs.json`.

Có thể chỉ định notebook và output:

```bash
python MLP-Classfication/vm_code/out_data/extract_vm_outputs.py \
  --notebook MLP-Classfication/vm_code/instance_wise/train_instance_wise.ipynb \
  --output artifacts/vm-training/instance-wise/notebook_outputs.json
```

Để đánh giá model instance-wise, ưu tiên `base_run/base_model/result.json` và `base_run/instance_unlearning/<manifest>/summary.json` sinh trên VM. Notebook không có output đã lưu không chứng minh rằng VM chưa chạy hay chưa có checkpoint.
