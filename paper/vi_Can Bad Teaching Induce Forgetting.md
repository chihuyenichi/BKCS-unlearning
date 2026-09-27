# Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher

## Thông tin nguồn

- PDF gốc: `zero retrain.pdf`
- Tác giả: Vikram S. Chundawat, Ayush K. Tarun, Murari Mandal, Mohan Kankanhalli.
- Hội nghị: AAAI 2023.
- Phạm vi tài liệu này: bản tổng quan tiếng Việt, không phải bản dịch toàn văn.

## Vấn đề paper giải quyết

Paper đề xuất unlearning cho deep neural network mà không cần retrain từ đầu và không yêu cầu thông tin đặc biệt đã được lưu trong quá trình train ban đầu.

Với tập train hoàn chỉnh `D_c`:

```text
D_c = D_r ∪ D_f;  D_r ∩ D_f = ∅
```

- `D_f` có thể là toàn bộ một/nhiều class, một cohort trong class, hoặc tập mẫu ngẫu nhiên trải trên nhiều class.
- `D_r` là phần dữ liệu phải tiếp tục được model xử lý tốt.
- Model retrain chỉ trên `D_r` là *gold model* để tham chiếu khi có điều kiện train nó.

## Ý tưởng: smart teacher, dumb teacher và student

Paper dùng knowledge distillation với ba network:

```text
Model gốc đã train đầy đủ ──→ smart/competent teacher Ts
Model cùng kiến trúc, khởi tạo ngẫu nhiên ──→ dumb/incompetent teacher Td
Student S, khởi tạo từ trọng số model gốc ──→ model sau unlearning
```

Trong quá trình unlearning:

- Trên retain samples `D_r`, student học theo phân phối dự đoán của smart teacher `T_s`. Mục đích là giữ kiến thức đúng.
- Trên forget samples `D_f`, student học theo phân phối dự đoán của dumb teacher `T_d`. Vì teacher này không có tri thức đã train, nó cung cấp target không đáng tin/ngẫu nhiên cho dữ liệu cần quên.

Loss được tạo bởi hai thành phần KL divergence:

```text
L = sum[x in D_r] KL(T_s(x) || S(x))
  + sum[x in D_f] KL(T_d(x) || S(x))
```

Do đó, mục tiêu không phải ép cố định `x_f → unknown`; prediction trên `D_f` được làm giống phân phối của dumb teacher. Paper muốn giảm tri thức đặc thù về `D_f`, đồng thời giữ hành vi của model gốc trên `D_r`.

## Metric Zero Retrain Forgetting (ZRF)

Paper cho rằng chỉ nhìn accuracy trên forget/retain set là chưa đủ: một model có thể dự đoán sai `D_f` nhưng vẫn giữ thông tin về các mẫu đó trong tham số.

Họ đề xuất **ZRF**, dựa trên Jensen–Shannon divergence giữa phân phối dự đoán của model unlearned và dumb teacher trên `D_f`. Mục đích là đánh giá mức độ prediction trên forget set đã mất tính “tri thức đã train” mà không cần luôn train một retrain model tốn kém.

Một điểm tinh tế: ZRF không nên được hiểu đơn giản là “càng gần 0/càng gần 1 càng tốt”. Giá trị phù hợp phụ thuộc model retrain lý tưởng, dataset và loại forget set.

## Đánh giá và phạm vi thực nghiệm

Paper thử nghiệm:

- class-level và random-subset unlearning;
- nhiều architecture, gồm CNN, vision transformer và LSTM;
- các domain gồm image classification, nhận dạng hoạt động người và phát hiện cơn động kinh.

Các chỉ số gồm hiệu năng trên `D_f`, `D_r`, model retrain tham chiếu khi có, ZRF, cùng đánh giá liên quan đến privacy/generalization.

## Ý nghĩa cho dự án PCAP

Đây là hướng phù hợp nếu dự án giữ được:

1. checkpoint classifier gốc;
2. các PCAP cần quên `D_f`;
3. một retain subset đại diện `D_r`.

Với `FlowModel` differentiable trong `train_pipeline.ipynb`, có thể thay Cross Entropy trong giai đoạn unlearning bằng distillation loss:

```text
retain PCAP  → ép student gần model gốc
forget PCAP  → ép student gần model khởi tạo ngẫu nhiên
```

Tuy nhiên, cần chọn kỹ target dumb teacher. Trong binary `known/unknown`, teacher ngẫu nhiên có thể làm một PCAP bị quên nghiêng về cả hai nhãn tùy seed, không bảo đảm luôn trả về `unknown`. Nếu yêu cầu sản phẩm là “mọi PCAP bị quên phải reject”, targeted `known → unknown` là policy rõ ràng hơn, nhưng không còn đúng nguyên bản phương pháp Bad Teaching.

## Kết luận ngắn

Paper đưa ra cách quên bằng **selective knowledge distillation**: giữ tri thức tốt trên retain set và thay tri thức trên forget set bằng tín hiệu từ một teacher không biết dữ liệu. Điểm mạnh là hỗ trợ class/cohort/random-subset unlearning; điểm cần chú ý là output của mẫu bị quên mang tính ngẫu nhiên, không mặc định là `unknown`.
