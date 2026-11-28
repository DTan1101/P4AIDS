# P4AIDS — Rain/Snow Classification under Sensor Missingness

Benchmark so sánh ba hướng phương pháp — học máy cổ điển (XGBoost), ROCKET
(MiniRocket/MultiRocket), và học sâu cho dữ liệu thiếu (GRU-D, BRITS, SAITS) —
cho bài toán phân loại pha giáng thủy Rain/Snow trên bộ dữ liệu thời tiết
Szeged (2006–2016), dưới điều kiện cảm biến mất số đo được giả lập theo ba
kiểu (point/subsequence/block) ở nhiều tỷ lệ (10/30/50%). Xem đầy đủ phương
pháp luận và kết quả trong [report/](report/).

Pipeline ảnh nấm cũ (deep mutual learning) đã được chuyển sang
`legacy/mushroom_classification/` khi dự án chuyển hướng sang bài toán này —
xem `legacy/mushroom_classification/README.md`.

## Cấu trúc

```
P4AIDS/
├── datasets/
│   ├── archive/weatherHistory.csv  # dữ liệu Szeged gốc
│   ├── szeged.py                   # load_raw() / clean()
│   ├── windows.py                  # cắt cửa sổ 24h -> nhãn t+1
│   ├── splits.py                   # chia theo năm + fold rolling-origin
│   ├── missingness.py              # giả lập point/subsequence/block, cache mặt nạ
│   ├── features.py                 # LiteStatTemp, mask/Δt, catch22/tsfresh (tuỳ chọn)
│   ├── dl_dataset.py                # Dataset PyTorch (value/mask/delta) cho nhánh DL
│   └── cache.py                     # lưu/nạp dữ liệu đã chuẩn bị (.npz)
├── models/
│   ├── classic.py          # XGBoost (NaN gốc), median+mask, MissForest
│   ├── rocket.py            # MiniRocket/MultiRocket + RidgeClassifier
│   ├── deep.py               # GRU-Fill, GRU+mask/Δt, GRU-D (tự cài)
│   ├── deep_pypots.py        # BRITS/SAITS/GRU-D qua thư viện PyPOTS
│   └── units_stub.py         # UniTS -- CHƯA triển khai (xem docstring)
├── trainers/
│   ├── classic_trainer.py
│   ├── rocket_trainer.py
│   └── deep_trainer.py
├── evaluation/
│   ├── metrics.py       # Macro F1, PR-AUC, balanced accuracy
│   ├── baselines.py     # lớp đa số, persistence, ngưỡng nhiệt độ, logreg, k-NN
│   ├── stats_tests.py   # McNemar, Friedman, biểu đồ critical difference
│   └── explain.py       # SHAP (cây), Integrated Gradients (DL) -- tuỳ chọn
├── options/
│   └── experiment_options.py   # CLI dùng chung cho mọi tools/run_*.py
├── tools/
│   ├── prepare_data.py     # làm sạch + cắt cửa sổ + chia tập -> outputs/data/
│   ├── build_masks.py      # sinh + cache mặt nạ thiếu dữ liệu -> outputs/masks/
│   ├── run_baselines.py
│   ├── run_classic.py
│   ├── run_rocket.py
│   ├── run_deep.py
│   └── aggregate_results.py   # gộp kết quả -> bảng Chương 6 + CD diagram
├── configs/
│   ├── rain_snow_data.yaml
│   ├── rain_snow_missing.yaml
│   ├── rain_snow_classic.yaml
│   ├── rain_snow_rocket.yaml
│   └── rain_snow_deep.yaml
├── utils/                 # seed/device/logging/config/checkpoint (dùng chung)
├── report/                # báo cáo LaTeX (không nằm trong git, xem .gitignore)
└── outputs/                # dữ liệu/mặt nạ/kết quả cache theo từng lần chạy
```

## Cài đặt

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` chia theo nhóm (core / ROCKET / Deep Learning / giải thích
mô hình) — đọc comment trong file để chỉ cài nhánh bạn cần. Nhánh Deep
Learning kéo theo `torch` + `pypots` (tải lớn); nhánh ROCKET cần `aeon`.

## Quy trình chạy

```bash
# 1. Làm sạch dữ liệu, cắt cửa sổ, chia theo năm (một lần)
python tools/prepare_data.py

# 2. Sinh + cache mặt nạ thiếu dữ liệu dùng chung cho mọi nhánh (một lần)
python tools/build_masks.py

# 3. Baseline (Mục 5.2)
python tools/run_baselines.py

# 4. Classic ML (Mục 4.4) -- có thể lọc theo một cấu hình cụ thể
python tools/run_classic.py --missing-kind point --missing-rate 0.3
python tools/run_classic.py --opts classic.impute_strategy=missforest classic.tune=true

# 5. ROCKET (Mục 4.5)
python tools/run_rocket.py --variant mini
python tools/run_rocket.py --variant multi

# 6. Deep Learning (Mục 4.6) -- chọn mô hình qua deep.model
python tools/run_deep.py --opts deep.model=gru_d
python tools/run_deep.py --opts deep.model=brits   # qua PyPOTS

# 7. Gộp toàn bộ kết quả thành bảng cho Chương 6 + biểu đồ critical-difference
python tools/aggregate_results.py
```

Mỗi `tools/run_*.py` nhận chung các cờ: `--config`, `--seed`,
`--missing-kind {point,subsequence,block}`, `--missing-rate {0.1,0.3,0.5}`
(không truyền = chạy toàn bộ 9 cấu hình + 0% thiếu), và `--opts key.path=value`
để override bất kỳ khoá nào trong YAML, theo đúng quy ước cũ của repo
(`utils/utils.py:apply_overrides`).

### Giới hạn bộ nhớ đã biết (ROCKET)

Nhánh ROCKET dùng `RidgeClassifier` (LSQR), chọn alpha và threshold theo
Macro F1 trên validation 2014. Cấu hình mặc định của nhánh đã triển khai:
MiniRocket 1008, MultiRocket 336 kernel budget; batch 512, 2 CPU jobs,
fit transformer tối đa 2500 mẫu train và giới hạn một ma trận feature 1024 MiB
(không phải giới hạn tổng RAM). Giảm ngân sách khi cần và dùng output/tag mới
để tránh trộn kết quả khác cấu hình.

Cài riêng nhánh: `python -m pip install -r requirements_rocket.txt`.
Khi dữ liệu và 9 mask dùng chung đã có, không cần chạy lại bước 1–2.
Hai lệnh ROCKET trên tạo 20 JSON chính và predictions tương ứng; Quick và
ablation lưu riêng. `python tools/plot_rocket_results.py` tổng hợp benchmark chính.
Xem [tài liệu Task 2](docs/TASK2_ROCKET.md) để tái lập 19 ablation, xuất đầy đủ bảng/hình
và đọc kết quả, giới hạn. Kết quả nằm trong `outputs/results/` (Git ignore),
bàn giao bằng ZIP riêng. Script tổng hợp toàn nhóm chỉ chạy khi có đủ các nhánh.

## Kiểm thử

```bash
pytest tests/ -v
```

`tests/` kiểm tra các bất biến quan trọng của pipeline dữ liệu và đánh giá
(`datasets/szeged|windows|splits|missingness|features.py`,
`evaluation/metrics|baselines|stats_tests|explain.py`): chống rò rỉ dữ liệu
khi chia theo năm, tỷ lệ mặt nạ thiếu dữ liệu đúng mục tiêu, nhãn cửa sổ đúng
vị trí t+1, các độ đo khớp với sklearn, McNemar/Friedman/critical-difference
đúng trên các case biết trước kết quả. Không phụ thuộc `datasets/archive/`
(dùng dữ liệu tổng hợp dựng trong fixture) nên chạy trong vài giây.

## License

For educational purposes only.
