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
│   ├── rocket.py            # MiniRocket/MultiRocket + RidgeClassifierCV
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
python tools/run_rocket.py --opts rocket.variant=multi

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

### Giao thức Deep Learning và artifact

Nhánh DL dùng giao thức `deep-v2-observed-zscore-macro-f1`:

- Cùng split thời gian, mask và bộ seed. Mean/std chỉ tính từ các ô quan sát
  được của train sau khi áp mask; kênh train hoàn toàn thiếu dùng mean=0/std=1.
- Chuẩn hoá trước khi điền thiếu: GRU tự cài dùng forward-fill, đầu cửa sổ
  còn thiếu điền 0 trong không gian chuẩn hoá; PyPOTS nhận NaN đã chuẩn hoá.
- Cùng weighted cross-entropy cho phần phân loại, Adam và learning rate từ
  `deep.lr`; giữ các loss phụ đặc trưng của từng kiến trúc PyPOTS.
- Chọn checkpoint theo Macro F1 trên **toàn validation**, scheduler
  ReduceLROnPlateau (factor=0.5, patience=3), early stopping theo `deep.patience`.
  Khi hoà điểm giữ epoch sớm nhất; `best_epoch` bắt đầu từ 0. Test chỉ được
  đánh giá sau khi khôi phục checkpoint tốt nhất.
- Adapter cố định `pypots==2.0`, sửa BRITS.fit dùng đúng `val_set` và thay vòng
  chọn checkpoint theo metric từng batch bằng Macro F1 của prediction cuối cùng.

`--seed` chọn một seed và ghi đè `experiment.seeds`. Không truyền cờ này thì
chạy danh sách seed trong YAML. `deep.hidden_size` áp dụng cho GRU/BRITS/GRUD
và là `d_model` mặc định của SAITS. `deep.pypots` chỉ chứa tuỳ chọn riêng của
mô hình; giá trị riêng như `rnn_hidden_size` hoặc `d_model` được ưu tiên.
Tuỳ chọn không được constructor hỗ trợ sẽ báo lỗi.

```bash
rtk proxy python tools/run_deep.py --model brits --hidden-size 32 --lr 0.0005 --seed 7 --missing-kind block --missing-rate 0.5
rtk proxy python tools/run_deep.py --model saits --opts deep.pypots.n_layers=1 deep.pypots.d_model=32 deep.pypots.n_heads=4
rtk proxy python tools/aggregate_results.py
```

Mỗi lần chạy CLI ghi JSON metrics có tên chứa phiên bản giao thức và YAML cấu
hình đã resolve. Trong `outputs/results/deep/artifacts/<protocol>/<model>/<case>/seed<seed>/`:

- `checkpoint.pt`: state_dict tốt nhất, mean/std, kích thước đầu vào, tham số
  tạo mô hình, trọng số lớp, seed, best_epoch và phiên bản thư viện. Đây là
  checkpoint để inference, chưa lưu trạng thái optimizer để resume training.
- `predictions.npz`: `val_idx/test_idx`, `val_y_true/test_y_true`,
  `val_y_pred/test_y_pred`, `val_y_score/test_y_score` (xác suất Snow).
  `idx` là vị trí cửa sổ trong file dữ liệu đã chuẩn bị.

Nạp bằng `utils.utils.load_checkpoint`; tạo lại mô hình từ metadata bằng
`build_custom_deep_model` hoặc `build_pypots_model`. Với PyPOTS, nạp state_dict
vào thuộc tính `.model` và tạo lại `class_weight` từ metadata. Khi inference,
dùng mean/std trong checkpoint; không fit lại thống kê. Các test trong
`tests/test_deep_protocol.py` minh hoạ việc khôi phục cả hai nhánh.

`summary.csv/.md` giữ từng seed; `summary_mean.csv/.md` chứa mean, sample std
(`ddof=1`, một lần chạy ghi std=0) và số lần chạy `n_seeds`. CD dùng mean giữa
seed và bỏ qua khi các kết quả DL trộn giao thức hoặc khác bộ seed. File cũ
được đánh dấu `legacy`; không trộn với kết quả DL mới. Giao thức này thay đổi
đầu vào và chọn checkpoint nên cần chạy lại benchmark để so sánh chất lượng.
Chạy lại cùng model/case/seed trong cùng thư mục sẽ ghi đè artifact tương ứng;
dùng `experiment.output_dir` khác để giữ các thí nghiệm có siêu tham số khác.

### Giới hạn bộ nhớ đã biết (ROCKET)

Trên máy không có swap, `RidgeClassifierCV` trên ma trận đặc trưng MiniRocket
mặc định (~10 nghìn chiều) với toàn bộ ~70 nghìn cửa sổ train có thể vượt RAM.
`configs/rain_snow_rocket.yaml` đã đặt sẵn `n_jobs`, `max_fit_samples`,
`transform_batch_size` ở mức an toàn; hạ thêm `rocket.n_kernels` (ví dụ
1000–2000) nếu vẫn gặp lỗi hết bộ nhớ trên máy yếu hơn.

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
