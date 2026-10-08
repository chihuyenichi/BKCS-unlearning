# Tổng quan tiếng Việt về các paper

Ba tài liệu dưới đây là bản **tổng quan tiếng Việt**, không phải bản dịch nguyên văn. Mỗi bản tập trung vào cách paper đặt bài toán, giả định dữ liệu có sẵn khi unlearn, thuật toán và cách đánh giá.

| PDF gốc | Bản tổng quan tiếng Việt | Trọng tâm |
|---|---|---|
| `Instance-Wise Unlearning.pdf` | [Learning to Unlearn — Instance-Wise Unlearning](<vi_Instance-Wise Unlearning.md>) | Quên từng mẫu chỉ với model đã train và tập cần quên; chủ đích làm các mẫu đó bị phân loại sai. |
| `zero retrain.pdf` | [Can Bad Teaching Induce Forgetting?](<vi_Can Bad Teaching Induce Forgetting.md>) | Teacher–student: giữ kiến thức từ smart teacher trên retain set, truyền dự đoán không đáng tin trên forget set. |
| `Chundawat et al. - 2023 - Zero-Shot Machine Unlearning.pdf` | [Zero-Shot Machine Unlearning](<vi_Zero-Shot Machine Unlearning.md>) | Unlearn khi không còn giữ cả retain lẫn forget samples; chỉ còn model gốc và yêu cầu quên class. |
| Tài liệu tổng hợp từ các nguồn liên quan | [Tổng quan các method unlearning](<vi_Unlearning_Methods_Overview.md>) | Giải thích Baseline, Amnesiac, Blindspot/Bad Teaching, BND/Boundary và MK-MMD; liên hệ với pipeline PCAP hiện tại. |

## Khung chung

Gọi dữ liệu train ban đầu là `D`, tập cần quên là `D_f`, tập giữ lại là `D_r`:

```text
D = D_r ∪ D_f;  D_r ∩ D_f = ∅
```

Chuẩn tham chiếu mạnh nhất là model train lại chỉ trên `D_r`:

```text
M_r = Train(D_r);  M_u ≈ M_r
```

Do đó, accuracy trên `D_f` giảm không tự nó đủ chứng minh unlearning; model vẫn phải giữ utility trên `D_r`, và khi khả thi cần so sánh với retrain baseline hoặc thêm đánh giá privacy/memorization.

## Liên hệ nhanh với dự án PCAP

`MLP-Classfication/code/train_pipeline.ipynb` hiện phù hợp nhất với setting **có thể truy cập `D_f` và một phần/toàn bộ `D_r`**. Đây không phải zero-shot setting. Với multiclass, cần phân biệt rõ:

- *class-level unlearning*: xóa mọi PCAP của một hoặc nhiều label;
- *instance/cohort-level unlearning*: xóa một nhóm file PCAP cụ thể, kể cả khi chúng thuộc nhiều label;
- *known → unknown*: là một chính sách output/relabeling có chủ đích, không phải định nghĩa đầy đủ của unlearning.
