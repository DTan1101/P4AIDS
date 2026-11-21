# Kết quả thực nghiệm

Các artifact của chiến dịch `deep_report` được lưu trên Google Drive, không đưa vào Git.

**Google Drive:** [Tải bộ kết quả Deep Learning](https://drive.google.com/drive/folders/11Xgq0jSXECO7x_Fgol10JUy3CtKfav91?usp=sharing)

## Bộ kết quả `deep_report`

Kết quả hoàn chỉnh của **5 mô hình**: GRU-Fill, GRU+mask/Δt, GRU-D tự cài,
BRITS và SAITS. Mỗi mô hình chạy 10 điều kiện dữ liệu với 3 seed `[42, 1, 2]`,
tổng cộng **150 lượt huấn luyện**.

- Điều kiện dữ liệu: không ẩn thêm và point/subsequence/block ở tỷ lệ 10%, 30%, 50%.
- Seed mặt nạ: `123`; điều kiện không ẩn thêm vẫn giữ phần thiếu tự nhiên.
- Chia theo thời gian: train 2006–2013, validation 2014, test 2015–2016.
- Giao thức: `deep-v2-observed-zscore-macro-f1`.
- Chọn checkpoint theo Macro F1 trên toàn validation; đánh giá test sau khi
  khôi phục checkpoint tốt nhất.

Chiến dịch ban đầu có thêm `grud_pypots`, nhưng lượt này bị dừng và không thuộc
bộ so sánh 5 mô hình. Vì vậy, manifest có thể ghi `interrupted` ở cấp chiến dịch;
hãy xem trạng thái từng mục trong `jobs` và các JSON kết quả khi kiểm tra độ đầy đủ.

## Cấu trúc sau khi tải về

Đặt hoặc giải nén bộ kết quả để có đường dẫn sau trong repository:

```text
outputs/experiments/deep_report/
├── manifest.json
├── configs/                    # Cấu hình của từng mô hình
├── logs/                       # Log huấn luyện
└── results/deep/
    ├── config_*.yaml           # Cấu hình đã resolve
    ├── deep_*.json             # Metrics, history và đường dẫn artifact
    └── artifacts/
        └── <protocol>/<model>/<case>/seed<seed>/
            ├── checkpoint.pt
            └── predictions.npz
```

`checkpoint.pt` lưu trọng số tốt nhất và metadata, gồm mean/std để inference;
chưa lưu đầy đủ trạng thái optimizer để resume training. `predictions.npz`
chứa chỉ số mẫu, nhãn, dự đoán và xác suất Snow trên validation/test.

Các đường dẫn trong JSON/manifest có thể là đường dẫn tuyệt đối của máy chạy;
cần điều chỉnh khi xử lý trên máy khác. Dữ liệu đã chuẩn bị và mask nằm riêng
ở `outputs/data/` và `outputs/masks/`, không mặc định nằm trong gói `deep_report`.

## Kiểm tra và tái lập

Đối chiếu checksum dữ liệu/mask, phiên bản thư viện và cấu hình trong
`manifest.json` trước khi so sánh hoặc chạy lại. Dùng thư mục đầu ra mới để
giữ nguyên bộ kết quả đã tải về. Hướng dẫn chạy CLI và tổng hợp kết quả nằm
trong [README của dự án](../../README.md).
