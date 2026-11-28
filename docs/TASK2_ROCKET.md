# Task 2 — ROCKET: triển khai, thực nghiệm và bàn giao

**Phạm vi:** MiniRocket và MultiRocket cho bài toán Rain/Snow khi cảm biến mất dữ liệu. Source thực nghiệm: commit `8e3b089` (branch `task2-rocket`, PR #4). Kết quả trích từ `P4AIDS(2).zip`, không dùng kết quả Quick làm số liệu chính. Các kết luận dưới đây chỉ áp dụng cho nhánh ROCKET, không xếp hạng các nhánh khác.

## 1. Bài toán, dữ liệu và giao thức

- Input: 24 giờ lịch sử, 5 biến theo thứ tự Temperature, Humidity, Wind Speed, Pressure, Visibility; shape lưu `(N, 24, 5)`, chuyển `(N, 5, 24)` cho aeon. Output: `Precip Type` tại t+1 (Rain=0, Snow=1).
- 95.888 cửa sổ từ `outputs/data/windows.npz`. Chia theo năm của timestamp nhãn (không shuffle, không purge cửa sổ giáp ranh). Các cửa sổ chồng lấn; chỉ dùng thông tin lịch sử, không dùng nhãn test để fit/tuning.
| Tập | Năm | Số cửa sổ | Rain | Snow |
| --- | --- | --- | --- | --- |
| Train | 2006–2013 | 69962 | 61056 | 8906 |
| Validation | 2014 | 8757 | 8272 | 485 |
| Test | 2015–2016 | 17169 | 15848 | 1321 |

Test có 7,69% Snow; luôn đoán Rain đã đạt accuracy khoảng 92,31%, do đó Macro F1 là chỉ số chính. PR-AUC theo code chung là Average Precision của Snow (không phải diện tích hình thang); bổ sung Balanced Accuracy và Precision/Recall/F1 từng lớp.

Dùng 9 mask có sẵn, seed 123, True = observed. Point: điểm ngẫu nhiên; Subsequence: đoạn ngắn/vừa liên tiếp theo biến; Block: đoạn dài trên một/vài biến. Tỷ lệ danh định 10/30/50% tính cả thiếu tự nhiên, ngân sách ẩn thêm được tính theo cửa sổ. 0% nghĩa là không ẩn thêm (tập test vẫn thiếu tự nhiên khoảng 0,4872%). Train, validation và test cùng điều kiện; fit lại mô hình cho từng kịch bản. Model seed 42.

## 2. Phương pháp và cấu hình

ROCKET dùng đặc trưng từ tích chập; MiniRocket dùng tập kernel phần lớn xác định và PPV; MultiRocket mở rộng với sai phân bậc một và nhiều phép pooling (tham khảo `dempster2021minirocket`, `tan2022multirocket` trong `report/refs.bib`).

Pipeline: mask chung → nội suy tuyến tính từng biến trong cửa sổ lịch sử (đầu/cuối dùng giá trị hợp lệ gần nhất, cả kênh thiếu dùng median quan trắc train) → chuyển layout → fit transformer trên mẫu con train ngẫu nhiên → transform toàn bộ train theo batch → nối 3 đặc trưng mask/Delta mỗi kênh (tỷ lệ thiếu, đoạn thiếu dài nhất /24, độ cũ quan trắc cuối /24) → StandardScaler fit trên train → RidgeClassifier có trọng số lớp → chọn alpha và threshold trên validation Macro F1 → test. Không refit trên validation. Decision score không phải xác suất hiệu chỉnh.

RidgeClassifier dùng solver LSQR, tol 1e-3, tối đa 1000 iterations; trọng số lớp balanced = N/(2N_c). Alpha ∈ {0.1,1,10}, mỗi alpha tìm threshold trên các phân vị 0–100 của decision scores validation, thêm 0 và giá trị ngay trên max. Tie-break: threshold gần 0, sau đó alpha nhỏ hơn. Test chỉ đánh giá.

| Thông số | MiniRocket | MultiRocket |
| --- | --- | --- |
| Kernel budget | 1008 | 336 |
| Feature đầu ra (gồm 15 mask/delta) | 1023 | 2703 |
| Tham số tuyến tính (gồm intercept) | 1024 | 2704 |
| Fit transformer tối đa | 2500 | 2500 |
| Batch transform | 512 | 512 |
| CPU jobs | 2 | 2 |
| Giới hạn một ma trận feature (MiB) | 1024 | 1024 |
| Model seed | 42 | 42 |
| Missing seed | 123 | 123 |

Ngân sách kernel giảm so với mặc định aeon để phù hợp máy; không phải so sánh ngang bằng số đặc trưng. Giới hạn ma trận không phải giới hạn RAM toàn tiến trình.

## 3. Kết quả thực nghiệm chính (20 cấu hình)

### 3.1. Không giả lập thiếu thêm

| Mô hình | Macro F1 | PR-AUC | Bal. Acc. | Snow Prec. | Snow Recall | Snow F1 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniRocket | 0.9145 | 0.9178 | 0.9275 | 0.8157 | 0.8713 | 0.8426 |
| MultiRocket | 0.8914 | 0.8716 | 0.9436 | 0.7119 | 0.9182 | 0.8020 |

### 3.2. Toàn bộ kịch bản (test)

| Kịch bản | Mini F1 | Mini AP | Multi F1 | Multi AP |
| --- | --- | --- | --- | --- |
| 0% | 0.9145 | 0.9178 | 0.8914 | 0.8716 |
| Point 10% | 0.9161 | 0.9219 | 0.8865 | 0.8894 |
| Point 30% | 0.9104 | 0.9130 | 0.8970 | 0.9004 |
| Point 50% | 0.9071 | 0.9088 | 0.8988 | 0.8985 |
| Subsequence 10% | 0.9075 | 0.9017 | 0.8925 | 0.8819 |
| Subsequence 30% | 0.8921 | 0.8730 | 0.8814 | 0.8559 |
| Subsequence 50% | 0.8753 | 0.8285 | 0.8616 | 0.8227 |
| Block 10% | 0.8910 | 0.9096 | 0.9039 | 0.8985 |
| Block 30% | 0.8973 | 0.8642 | 0.8818 | 0.8875 |
| Block 50% | 0.8896 | 0.8380 | 0.8826 | 0.8343 |

MiniRocket cao hơn MultiRocket về Macro F1 ở 9/10 kịch bản (ngoại lệ Block 10%). Subsequence 50% cho Macro F1 thấp nhất ở cả hai; Point 50% vẫn đạt 0.9071/0.8988. So với 0%, Subsequence 50% giảm khoảng 3.92/2.99 điểm phần trăm (Mini/Multi). Các đường không đơn điệu vì fit/tune độc lập cho từng mask; không suy ra thiếu dữ liệu gây cải thiện mô hình.

### 3.3. Bảng chi tiết (validation và test)

| Mô hình | Kịch bản | Val F1 | Test F1 | AP | Bal. Acc. | Snow Prec. | Snow Rec. | Snow F1 | Alpha | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiniRocket | 0% | 0.9262 | 0.9145 | 0.9178 | 0.9275 | 0.8157 | 0.8713 | 0.8426 | 10.0 | 0.4769 |
| MiniRocket | Point 10% | 0.9320 | 0.9161 | 0.9219 | 0.9261 | 0.8245 | 0.8675 | 0.8454 | 10.0 | 0.4821 |
| MiniRocket | Point 30% | 0.9297 | 0.9104 | 0.9130 | 0.9106 | 0.8343 | 0.8350 | 0.8347 | 10.0 | 0.5149 |
| MiniRocket | Point 50% | 0.9320 | 0.9071 | 0.9088 | 0.9106 | 0.8210 | 0.8365 | 0.8286 | 0.1 | 0.4957 |
| MiniRocket | Subsequence 10% | 0.9342 | 0.9075 | 0.9017 | 0.9090 | 0.8258 | 0.8327 | 0.8292 | 10.0 | 0.4840 |
| MiniRocket | Subsequence 30% | 0.9285 | 0.8921 | 0.8730 | 0.9031 | 0.7782 | 0.8259 | 0.8013 | 10.0 | 0.4392 |
| MiniRocket | Subsequence 50% | 0.9022 | 0.8753 | 0.8285 | 0.8856 | 0.7486 | 0.7933 | 0.7703 | 10.0 | 0.4595 |
| MiniRocket | Block 10% | 0.9239 | 0.8910 | 0.9096 | 0.9442 | 0.7097 | 0.9198 | 0.8012 | 0.1 | 0.3069 |
| MiniRocket | Block 30% | 0.9159 | 0.8973 | 0.8642 | 0.8958 | 0.8137 | 0.8070 | 0.8103 | 10.0 | 0.5025 |
| MiniRocket | Block 50% | 0.9148 | 0.8896 | 0.8380 | 0.8930 | 0.7890 | 0.8039 | 0.7964 | 0.1 | 0.4621 |
| MultiRocket | 0% | 0.9354 | 0.8914 | 0.8716 | 0.9436 | 0.7119 | 0.9182 | 0.8020 | 10.0 | 0.2715 |
| MultiRocket | Point 10% | 0.9422 | 0.8865 | 0.8894 | 0.8672 | 0.8377 | 0.7464 | 0.7894 | 0.1 | 0.4914 |
| MultiRocket | Point 30% | 0.9375 | 0.8970 | 0.9004 | 0.9458 | 0.7264 | 0.9205 | 0.8120 | 10.0 | 0.3344 |
| MultiRocket | Point 50% | 0.9354 | 0.8988 | 0.8985 | 0.9054 | 0.7991 | 0.8282 | 0.8134 | 10.0 | 0.5172 |
| MultiRocket | Subsequence 10% | 0.9297 | 0.8925 | 0.8819 | 0.8932 | 0.8002 | 0.8032 | 0.8017 | 10.0 | 0.5336 |
| MultiRocket | Subsequence 30% | 0.9251 | 0.8814 | 0.8559 | 0.8811 | 0.7817 | 0.7805 | 0.7811 | 0.1 | 0.5409 |
| MultiRocket | Subsequence 50% | 0.9034 | 0.8616 | 0.8227 | 0.8572 | 0.7543 | 0.7343 | 0.7442 | 0.1 | 0.4923 |
| MultiRocket | Block 10% | 0.9102 | 0.9039 | 0.8985 | 0.9362 | 0.7627 | 0.8955 | 0.8238 | 0.1 | 0.3966 |
| MultiRocket | Block 30% | 0.9281 | 0.8818 | 0.8875 | 0.9343 | 0.6945 | 0.9016 | 0.7846 | 0.1 | 0.3075 |
| MultiRocket | Block 50% | 0.9091 | 0.8826 | 0.8343 | 0.8897 | 0.7686 | 0.7994 | 0.7837 | 10.0 | 0.4876 |

### 3.4. Vùng nhiệt độ và mùa (0% giả lập)

| Nhóm | N | Mini F1 | Mini AP | Multi F1 | Multi AP |
| --- | --- | --- | --- | --- | --- |
| Nhiệt độ 0–4°C | 2731 | 0.6639 | 0.2706 | 0.6033 | 0.2686 |
| Đông | 4343 | 0.8968 | 0.9285 | 0.8672 | 0.9056 |
| Xuân | 4416 | 0.8428 | 0.7217 | 0.8169 | 0.6826 |
| Hạ | 4416 | N/A | N/A | N/A | N/A |
| Thu | 3994 | 0.8867 | 0.8482 | 0.8237 | 0.5818 |

Nhiệt độ dùng quan trắc gốc tại giờ cuối cửa sổ (không phải nhiệt độ target, không cấp giá trị bị che cho mô hình); mùa tính theo timestamp nhãn. Hạ không có Snow nên không báo Macro F1/AP cho subgroup theo quy ước code. Nhóm nhiệt độ 0–4°C khó hơn toàn test. EDA trong báo cáo nhóm cho thấy nhãn cùng thời điểm tách theo dấu nhiệt độ, nên quan hệ mạnh với nhiệt độ có thể phản ánh cách sinh nhãn API; không diễn giải là bằng chứng nhân quả hay dự báo pha giáng thủy độc lập.

## 4. Ablation (19 cấu hình)

### 4.1. Thành phần pipeline — MiniRocket Block 50%

| Biến thể | Val F1 | Test F1 | AP | Snow Recall | Snow F1 | Δ F1 (điểm %) |
| --- | --- | --- | --- | --- | --- | --- |
| Đầy đủ | 0.9148 | 0.8896 | 0.8380 | 0.8039 | 0.7964 | 0.00 |
| Không class weighting | 0.9125 | 0.8832 | 0.8483 | 0.7911 | 0.7845 | -0.64 |
| Không Mask/Delta | 0.9148 | 0.8859 | 0.8442 | 0.7918 | 0.7894 | -0.37 |
| Forward-fill | 0.9034 | 0.8748 | 0.8357 | 0.7752 | 0.7691 | -1.48 |

Full pipeline cho Test Macro F1/Snow F1 cao nhất trong bốn cấu hình này. Bỏ class weighting hoặc mask/delta có AP cao hơn; không kết luận mọi metric đều cải thiện. Linear tốt hơn forward-fill tại điều kiện đã thử. Alpha/threshold được tune lại cho mỗi đối chứng, do đó so sánh pipeline hoàn chỉnh chứ không giữ cố định mọi tham số đã fit.

### 4.2. Tổ hợp biến — MiniRocket

| Tổ hợp | Val F1 0% | Test F1 0% | AP 0% | Val F1 B50 | Test F1 B50 | AP B50 |
| --- | --- | --- | --- | --- | --- | --- |
| Đầy đủ 5 biến | 0.9262 | 0.9145 | 0.9178 | 0.9148 | 0.8896 | 0.8380 |
| Chỉ Temperature | 0.9537 | 0.9219 | 0.9397 | 0.9308 | 0.9063 | 0.8739 |
| Temperature + Humidity | 0.9470 | 0.9174 | 0.9404 | 0.9274 | 0.9045 | 0.8729 |
| Temperature + Humidity + Pressure | 0.9388 | 0.9146 | 0.9172 | 0.9217 | 0.8925 | 0.8498 |
| Bỏ Temperature | 0.6021 | 0.6024 | 0.1979 | 0.5819 | 0.6177 | 0.2189 |
| Bỏ Humidity | 0.9285 | 0.9086 | 0.9119 | 0.9102 | 0.8836 | 0.8079 |
| Bỏ Wind | 0.9262 | 0.9061 | 0.9141 | 0.9148 | 0.8855 | 0.8204 |
| Bỏ Pressure | 0.9262 | 0.9041 | 0.9116 | 0.9148 | 0.8884 | 0.8143 |
| Bỏ Visibility | 0.9285 | 0.9161 | 0.9256 | 0.9123 | 0.8704 | 0.8175 |

Temperature-only có Val F1 cao nhất trong các tổ hợp đã khảo sát ở cả 0% và B50 (0.9537/0.9308), Test F1 tương ứng 0.9219/0.9063. Bỏ Temperature giảm về 0.6024/0.6177. Temperature-only vẫn có mask/delta của Temperature; không phải baseline ngưỡng nhiệt độ đơn giản. Giữ mô hình năm biến làm benchmark chính. Các mask được sinh trên 5 kênh trước khi chọn subset, tỷ lệ thiếu thực tế trên kênh còn lại có thể khác mức danh định; xem `selected_channel_missing_test`. Không suy rộng rằng các biến khác luôn vô ích.

### 4.3. Ảnh hưởng của tối ưu ngưỡng quyết định

| Mô hình | Kịch bản | Macro F1 ngưỡng mặc định (0) | Macro F1 ngưỡng chọn trên validation | Δ F1 (điểm %) |
| --- | --- | --- | --- | --- |
| MiniRocket | 0% | 0.8324 | 0.9145 | +8.21 |
| MultiRocket | 0% | 0.8457 | 0.8914 | +4.57 |
| MiniRocket | Block 50% | 0.8167 | 0.8896 | +7.29 |
| MultiRocket | Block 50% | 0.8208 | 0.8826 | +6.18 |

Nguồn: `test_metrics.macro_f1_default_threshold` và `test_metrics.macro_f1` trong JSON của benchmark chính. So sánh trên cùng decision scores, cùng classifier và alpha đã chọn; chỉ thay đổi ngưỡng quyết định. Trong bốn cấu hình minh họa, ngưỡng chọn theo Macro F1 trên validation cải thiện Test Macro F1. Đây là đối chứng threshold tuning, không phải loại bỏ toàn bộ hyperparameter tuning; không chọn lại ngưỡng từ test và không thêm lượt huấn luyện.

## 5. Chi phí tính toán

| Mô hình | Kernel budget | Features | Tham số tuyến tính | Train+tune (s) | Infer (ms/mẫu) | F1 TB | AP TB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniRocket | 1008 | 1023 | 1024 | 24.21 | 0.2169 | 0.9001 | 0.8877 |
| MultiRocket | 336 | 2703 | 2704 | 55.16 | 0.4201 | 0.8878 | 0.8741 |

Trung bình số học trên 10 kịch bản (không phải 10 lần lặp độc lập). Train+tune gồm fit transformer, scaler, classifier và validation tuning; inference không gồm chuẩn bị mask/delta; lần đầu có thể gồm Numba JIT. Số tham số chỉ tính hệ số ridge và intercept, không phải kích thước toàn bộ transformer. MiniRocket nhanh hơn theo cấu hình và máy này.

CPU Intel Core i3-1125G4 2.00 GHz, 4 nhân/8 luồng; Python 3.11.5; aeon 1.3.0; NumPy 2.2.6; pandas 2.3.3; scikit-learn 1.6.1; Numba 0.61.2; SciPy 1.15.3. Metadata platform: `Windows-10-10.0.26200-SP0` (không suy ra tên phiên bản Windows từ chuỗi này). Thông tin từ `environment.txt`: RAM hệ thống 31,77 GiB; Microsoft Windows 11 Home Single Language, version 10.0.26200, kiến trúc 64-bit. Chạy trên CPU (2 jobs), không sử dụng GPU.

## 6. Tái lập thực nghiệm

Chạy từ root `P4AIDS`, Python 3.11. Dùng dữ liệu/mask nhóm đã chuẩn bị; không sinh lại nếu đã có. Kiểm tra 9 mask và hash dữ liệu. Trên máy đã làm Task 2 có thể dùng `$python = "..\P4AIDS_Task2_ROCKET\.venv\Scripts\python.exe"` và thay `python` bằng `& $python`.

```bash
python -m pip install -r requirements_rocket.txt
python -m pip check
python -m pytest tests/test_rocket.py -v
python -m pytest tests/ -v
python tools/run_rocket.py --variant mini --full-only --tag quick --kernels 168
python tools/run_rocket.py --variant mini --missing-kind point --missing-rate 0.3 --tag quick --kernels 168
python tools/run_rocket.py --variant multi --full-only --tag quick --kernels 168
python tools/run_rocket.py --variant mini
python tools/run_rocket.py --variant multi
python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag no_weight --no-weight
python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag no_mask --no-mask-delta
python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag ffill --imputer ffill
```

Channel ablation (PowerShell; chạy sau benchmark chính; khai báo `$python` như trên hoặc `$python = "python"` nếu đã kích hoạt môi trường):

```powershell
$subsets = [ordered]@{
    temperature_only = @("temperature")
    temp_humidity = @("temperature","humidity")
    temp_humidity_pressure = @("temperature","humidity","pressure")
    drop_temperature = @("humidity","wind","pressure","visibility")
    drop_humidity = @("temperature","wind","pressure","visibility")
    drop_wind = @("temperature","humidity","pressure","visibility")
    drop_pressure = @("temperature","humidity","wind","visibility")
    drop_visibility = @("temperature","humidity","wind","pressure")
}
foreach ($name in $subsets.Keys) {
    $channels = $subsets[$name]
    & $python tools/run_rocket.py --variant mini --full-only --tag $name --channels $channels
    if ($LASTEXITCODE -ne 0) { throw "Failed: $name full" }
    & $python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag $name --channels $channels
    if ($LASTEXITCODE -ne 0) { throw "Failed: $name block" }
}
$dirs = @("outputs/results/rocket")
$dirs += @(Get-ChildItem outputs/results/rocket_extra -Directory -Recurse | Where-Object {
    $_.FullName -notmatch '[\\/]quick([\\/]|$)' -and
    @(Get-ChildItem $_.FullName -Filter "rocket_*.json").Count -gt 0
} | ForEach-Object { $_.FullName })
& $python tools/plot_rocket_results.py --input $dirs
```

Runner bỏ qua kết quả cũ chỉ khi fingerprint khớp và file predictions tồn tại. Không đổi code/config giữa các lần chạy; dùng thư mục/tag mới nếu thay cấu hình. Nếu muốn tái lập từ đầu, dùng thư mục output riêng. Không xóa kết quả gốc.

## 7. Bàn giao, kiểm chứng và giới hạn

- `outputs/results/rocket/`: 20 JSON + 20 predictions, settings theo variant. `rocket_extra/`: 19 ablation + 3 quick (tách riêng). `rocket_analysis/`: 6 CSV, 73 PNG và environment.txt.
- `metrics.csv`: 39 dòng; `summary.csv`: 20; `full_data_comparison.csv`: 2; `subgroup_metrics.csv`: 195; `validation_trials.csv`: 117; `ablation.csv`: 19. Std giữa seed không xác định vì n=1, không coi thanh sai số 0 là bằng chứng ổn định.
- 42 file predictions có cùng sample_idx, timestamp và y_true theo test của `windows.npz`; hash dataset, mask và source được đối chiếu với file thật; đã tính lại Macro F1/AP/Balanced Accuracy/confusion matrix và đối chiếu JSON. Không huấn luyện lại trong lần review này. 77 unit tests PASS trên máy thực nghiệm theo ảnh Terminal trước đó.
- Bàn giao predictions để nhóm chạy McNemar nếu cần. Không khẳng định ý nghĩa thống kê; cửa sổ chồng lấn và các mask dùng cùng tập test không độc lập. Không tự chạy Friedman liên nhánh khi chưa có kết quả nhóm.
- Chỉ một bộ dữ liệu, nhãn từ API có thể phụ thuộc nhiệt độ; một model seed và một mask seed; kernel budget khác nhau; 0% còn thiếu tự nhiên; mỗi kịch bản train lại, không đánh giá một model cố định trên nhiều missing rate. Channel ablation chỉ gồm MiniRocket ở 0% và B50.
- Git: không commit `.venv`, cache, ZIP kết quả hoặc `outputs/results` (đã ignore). Chỉ commit tài liệu, code đã kiểm tra và hình báo cáo chọn lọc theo thống nhất với nhóm. PR #4 đã mở; trưởng nhóm review/merge.

## 8. Kết luận riêng Task 2

Trong cấu hình đã đo, MiniRocket đạt Macro F1 trung bình 0.9001 so với 0.8878 của MultiRocket và chi phí thấp hơn. Subsequence 50% là trường hợp khó nhất theo Macro F1; vùng 0–4°C cũng có hiệu năng thấp đáng kể. Temperature là biến nổi bật trong ablation, và Temperature-only là ứng viên tốt theo validation. Kết quả này không thay thế benchmark năm biến đã đăng ký, không khẳng định ROCKET tốt nhất toàn nhóm. Hướng mở rộng: so sánh ngân sách feature tương đương, nhiều seed và dữ liệu có nhãn độc lập; đánh giá train sạch/test missing như giao thức bổ sung nếu nhóm yêu cầu.
