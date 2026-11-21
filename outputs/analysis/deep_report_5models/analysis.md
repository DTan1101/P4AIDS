# Phân tích hoàn chỉnh 5 mô hình Deep Learning

Snapshot: 2026-11-22T02:43:10.838956+07:00

## Phạm vi và xác minh

- Đủ 5 mô hình × 10 điều kiện × 3 seed = 150 lượt hoàn chỉnh, 50 JSON và 150 prediction archive.
- Manifest interrupted vì job grud_pypots thứ sáu bị dừng; không phải thất bại của 5 mô hình được phân tích.
- Đã tính lại Macro F1, PR-AUC (average precision), balanced accuracy và precision/recall/F1/support từng lớp từ prediction.
- Đã đối chiếu chỉ số mẫu/nhãn với file prepared data; cả 10 file data/mask khớp checksum lúc khởi chạy.
- Train 2006–2013, validation 2014, test 2015–2016; test 17.169 cửa sổ, gồm 1.321 Snow.
- Chung hidden_size=64, batch_size=64, lr=0.001, tối đa 50 epoch, patience=10, CUDA, deterministic; seed [42,1,2], mask seed 123.
- Tỷ lệ 0% nghĩa là không ẩn thêm; vẫn có khoảng 0.416% ô thiếu tự nhiên.

## Bảng Macro F1 theo điều kiện

Mean ± sample std giữa 3 seed mô hình.

| Case | GRU-Fill | GRU+mask/Δt | GRU-D | BRITS | SAITS |
|---|---:|---:|---:|---:|---:|
| point 0% | 0.9342 ± 0.0026 | 0.9407 ± 0.0041 | 0.9395 ± 0.0019 | 0.9312 ± 0.0125 | 0.9434 ± 0.0033 |
| point 10% | 0.9320 ± 0.0020 | 0.9331 ± 0.0085 | 0.9336 ± 0.0034 | 0.9318 ± 0.0048 | 0.9375 ± 0.0063 |
| point 30% | 0.9248 ± 0.0014 | 0.9267 ± 0.0087 | 0.9211 ± 0.0012 | 0.9236 ± 0.0052 | 0.9259 ± 0.0135 |
| point 50% | 0.9028 ± 0.0071 | 0.9121 ± 0.0068 | 0.9131 ± 0.0055 | 0.9281 ± 0.0055 | 0.9207 ± 0.0016 |
| subsequence 10% | 0.9183 ± 0.0027 | 0.9260 ± 0.0027 | 0.9217 ± 0.0006 | 0.9309 ± 0.0059 | 0.9352 ± 0.0050 |
| subsequence 30% | 0.8797 ± 0.0064 | 0.8881 ± 0.0097 | 0.8930 ± 0.0021 | 0.8996 ± 0.0039 | 0.9068 ± 0.0033 |
| subsequence 50% | 0.8377 ± 0.0107 | 0.8605 ± 0.0082 | 0.8570 ± 0.0036 | 0.8682 ± 0.0032 | 0.8636 ± 0.0120 |
| block 10% | 0.9305 ± 0.0027 | 0.9324 ± 0.0033 | 0.9282 ± 0.0028 | 0.9267 ± 0.0120 | 0.9411 ± 0.0020 |
| block 30% | 0.8972 ± 0.0040 | 0.8928 ± 0.0079 | 0.9062 ± 0.0039 | 0.9100 ± 0.0062 | 0.9172 ± 0.0007 |
| block 50% | 0.8674 ± 0.0122 | 0.8663 ± 0.0168 | 0.8705 ± 0.0178 | 0.8776 ± 0.0062 | 0.8797 ± 0.0078 |

## Tổng thể trên 9 điều kiện thiếu

Mỗi seed được lấy trung bình không trọng số trên 9 case, sau đó tính mean/std giữa 3 seed. Std này không phải std giữa 9 case.

| Model | Mean Macro F1 ± std | Mean PR-AUC ± std | Balanced accuracy | Pipeline cho 30 lượt (phút) | Trung vị khoảng epoch (giây) |
|---|---:|---:|---:|---:|---:|
| SAITS | 0.91418 ± 0.00282 | 0.94495 ± 0.00512 | 0.96146 | 126.17 | 10 |
| BRITS | 0.91073 ± 0.00246 | 0.94294 ± 0.00213 | 0.96421 | 1255.97 | 84 |
| GRU-D | 0.90494 ± 0.00159 | 0.92829 ± 0.00433 | 0.95194 | 236.48 | 18 |
| GRU+mask/Δt | 0.90422 ± 0.00617 | 0.93289 ± 0.00123 | 0.95540 | 20.45 | 2 |
| GRU-Fill | 0.89893 ± 0.00101 | 0.92234 ± 0.01368 | 0.95052 | 17.22 | 2 |

## Kết luận mô tả từ số liệu

1. SAITS có mean Macro F1 và PR-AUC cao nhất trên 9 case thiếu, đứng đầu ở 6/9 case; BRITS đứng đầu 2 case, GRU+mask/Δt đứng đầu 1 case. Đây là thứ hạng mô tả, chưa phải kiểm định ý nghĩa thống kê.
2. BRITS dẫn đầu ở point 50% (0.9281 ± 0.0055) và subsequence 50% (0.8682 ± 0.0032). BRITS còn có mức F1 thấp nhất giữa 9 case cao nhất trong 5 mô hình; lợi thế này không đồng nghĩa đứng đầu trung bình mọi điều kiện.
3. GRU+mask/Δt hơn GRU-Fill ở 7/9 case thiếu; lợi ích lớn nhất ở subsequence 50%, +0.02277 Macro F1. Block 30% và 50% là hai ngoại lệ.
4. GRU-D chỉ hơn GRU+mask/Δt khoảng 0.00072 mean Macro F1, nhưng mất khoảng 11.56 lần thời gian pipeline. PR-AUC trung bình của GRU-D còn thấp hơn GRU+mask/Δt (0.92829 so với 0.93289).
5. SAITS mất 126.17 phút cho 30 lượt, so với 1255.97 phút của BRITS: nhanh hơn khoảng 9.95 lần theo phép đo runner. SAITS cũng nhanh hơn GRU-D khoảng 1.87 lần; GRU-Fill và GRU+mask/Δt vẫn nhanh hơn SAITS đáng kể.
6. Ở cả 5 mô hình, subsequence 50% là điều kiện có Macro F1 thấp nhất; không ủng hộ giả thuyết block luôn khó nhất cho bộ mask hiện tại.
7. BRITS có balanced accuracy và recall Snow trung bình cao nhất (0.96421 và 0.95612), còn SAITS có precision Snow trung bình cao hơn (0.76183 so với 0.74688). Lựa chọn metric làm thay đổi khía cạnh mô hình nổi trội.

## Chi phí và hội tụ

| Model | Tổng epoch / 30 lượt | Trung bình epoch / lượt | Lượt chạm 50 epoch |
|---|---:|---:|---:|
| GRU-Fill | 536 | 17.87 | 0 |
| GRU+mask/Δt | 648 | 21.60 | 0 |
| GRU-D | 765 | 25.50 | 1 |
| BRITS | 844 | 28.13 | 1 |
| SAITS | 712 | 23.73 | 0 |

pipeline_seconds bao gồm tiền xử lý, train, validation/test và lưu artifact. Không dùng nó làm thời gian train thuần hoặc latency inference.
Khoảng epoch từ timestamp log chỉ chính xác đến giây. GRU-D có khoảng gián đoạn 1121 giây từ 23:53:36 tới 00:12:17 tại point_30; chưa xác định là sleep, tải hệ thống hay nguyên nhân khác. Vì vậy nên dùng trung vị khi mô tả tốc độ epoch, và giữ nguyên pipeline_seconds như phép đo riêng của runner.
Không có số đo CPU/RAM/VRAM utilization, công suất hay năng lượng; không suy ra các số liệu này từ thời gian train.

## Đoạn viết đề xuất cho báo cáo

Trên chín điều kiện thiếu dữ liệu, SAITS đạt Macro F1 trung bình cao nhất (0.91418), tiếp theo là BRITS (0.91073). Tuy nhiên, BRITS đạt kết quả tốt nhất ở hai điều kiện thiếu 50% dạng point và subsequence. Các kết quả cho thấy lợi thế kiến trúc phụ thuộc cấu trúc thiếu dữ liệu, thay vì một mô hình vượt trội đồng đều ở mọi điều kiện.

Việc bổ sung mask và Δt cho GRU cải thiện Macro F1 so với baseline forward-fill ở bảy trong chín điều kiện thiếu. GRU-D đem lại mức tăng trung bình nhỏ so với GRU+mask/Δt, trong khi thời gian pipeline lớn hơn khoảng 11.56 lần. Trong thiết lập hiện tại, GRU+mask/Δt là phương án nhẹ có hiệu năng gần GRU-D; SAITS đạt hiệu năng trung bình cao hơn cả hai và cần ít thời gian hơn GRU-D.

Cả năm mô hình đều gặp khó khăn lớn nhất ở subsequence 50%. Nhận định block luôn là dạng thiếu gây suy giảm mạnh nhất không được số liệu hiện tại ủng hộ. Các kết luận được giới hạn trong dữ liệu một trạm và bộ sinh mask đang sử dụng.

## Giới hạn và cách sử dụng

- Chỉ kết luận về 5 mô hình này; chưa so sánh ML/ROCKET hoặc UniTS.
- Một trạm thời tiết, một split thời gian và một mask seed. Std đo biến động model seed, không phải độ bất định giữa dataset hoặc mẫu thiếu.
- Các cửa sổ chồng lấn và các case dùng cùng tập test; không xem 9 case hay 3 bản prediction test là các dataset độc lập.
- Chưa chạy kiểm định ý nghĩa thống kê; không khẳng định SAITS hơn BRITS có ý nghĩa thống kê.
- Mỗi mô hình được train lại dưới cùng kiểu/tỷ lệ thiếu với test; chưa đánh giá train đầy đủ rồi test thiếu hoặc chuyển kiểu thiếu.
- Giữ cấu hình đã chốt; không dùng bảng test để tuning thêm hoặc chọn seed tốt nhất.
- TensorBoard không được tạo; history trong JSON đủ dựng đường hội tụ. Chưa đo lại inference/checkpoint quality trên GPU.

## Các file đi kèm

- summary_mean_std.csv: đủ metrics theo model/case/split, mean và std giữa seed.
- overall_comparison.csv, overall_per_seed.csv: trung bình 9 case và std giữa 3 seed của trung bình đó.
- case_winners.csv, descriptive_ranks.csv, degradation.csv: thứ hạng mô tả và mức giảm so với không ẩn thêm.
- paired_seed_differences.csv: chênh lệch ghép theo cùng case/seed; chưa suy diễn ý nghĩa thống kê.
- confusion_per_seed.csv: số TN/FP/FN/TP từng seed; không gộp ba seed như ba tập test độc lập.
- cost_and_progress.csv, timing_anomalies.csv: thời gian và khoảng log bất thường.
- macro_f1_missingness.png, pr_auc_missingness.png, macro_f1_heatmap.png, performance_cost.png: hình cho báo cáo.
- prediction_verification.csv, snapshot.json, sources/: bằng chứng đối chiếu, checksum và snapshot nguồn.
