# Classic ML - Nhat ky thuc nghiem

## Pham vi

Thuc nghiem duoc thuc hien cho nhánh Classic ML va tap trung vao XGBoost, Random Forest,
LightGBM, CatBoost, feature engineering va xu ly du lieu thieu. ROCKET va Deep Learning
khong thay doi trong dot chay nay.

## Du lieu va chia

- Du lieu: Szeged Weather 2006-2016.
- Dau vao: 5 bien tren cua so 24 gio.
- Dau ra: Rain=0/Snow=1 tai thoi diem t+1.
- Sau lam sach va cat cua so: 95,888 cua so.
- Train: 69,962 mau, nam 2006-2013.
- Validation: 8,757 mau, nam 2014.
- Test co dinh: 17,169 mau, nam 2015-2016.
- Test chi dung mot lan sau khi chon cau hinh.

## Missingness

Sinh va cache bang `tools/build_masks.py` voi seed 123:

- 0%.
- Point: 10%, 30%, 50%.
- Subsequence: 10%, 30%, 50%.
- Block: 10%, 30%, 50%.

Tat ca model trong ma tran chinh dung cung X, split va mask.

## Dependency va moi truong

Moi truong chay:

- Python 3.12.5.
- Windows 11.
- CPU: Intel64 Family 6 Model 186 Stepping 2.
- Commit: `f209fd4440df197873c597d100c73a195758b867`.
- XGBoost 3.4.1.
- LightGBM 4.7.0.
- CatBoost 1.2.10.
- scikit-learn 1.9.1.
- Optuna 5.1.0.
- imbalanced-learn 0.14.2.
- pycatch22 1.0.2.
- tsfresh 0.21.2.

Chi tiet duoc luu tai `outputs/results/classic_full/environment.json`.

## Ma tran da chay

Ma tran chính gom 50 run:

| Model | Feature | Missing handling | Imbalance |
|---|---|---|---|
| XGBoost | LiteStatTemp + mask/delta | Native NaN | scale_pos_weight |
| XGBoost | LiteStatTemp + mask/delta | Median + indicator | scale_pos_weight |
| Random Forest | LiteStatTemp + mask/delta | Median + indicator | scale_pos_weight |
| LightGBM | LiteStatTemp + mask/delta | Native NaN | scale_pos_weight |
| CatBoost | LiteStatTemp + mask/delta | Native NaN | scale_pos_weight |

Ma tran nay chay tren 0% va 9 cau hinh missing.

Ablation va bo sung:

- 12 run feature: raw lag, catch22, tsfresh tren 0%, point50 va block50.
- 3 run XGBoost + MissForest.
- 2 run SMOTE tren point30 va block50.
- 2 run tuning Optuna tren point30 va block50.
- Tong so: 66 run, seed 42.

Luu y:

- Catch22 va tsfresh la ablation option va khong thay the LiteStatTemp trong ma tran chinh.
- Catch22 duoc dien khi toan bo mot kenh bi thieu.
- Tsfresh truoc tien duoc forward/interpolation de thoa yeu cau khong NaN cua thu vien.
- Cac feature ablation khong dung feature selection theo nhãn.

## Tuning

Tuning XGBoost dung 12 trial Optuna cho moi cau hinh dai dien:

- Point 30%.
- Block 50%.

Khong gian tim kiem:

- max_depth: 3-9.
- learning_rate: 0.01-0.20.
- n_estimators: 150-500.
- subsample: 0.7-1.0.
- colsample_bytree: 0.7-1.0.
- min_child_weight: 1-8.

Muc tieu la Macro F1 tren validation.

## Lenh chay

```powershell
python tools/prepare_data.py --config configs/rain_snow_data.yaml
python tools/build_masks.py --config configs/rain_snow_missing.yaml
python tools/run_classic_full.py
python tools/run_classic_full_continue.py
python tools/analyze_classic_full.py
```

## Dau ra

Moi run duoc luu thanh JSON trong `outputs/results/classic_full/`:

- Metrics validation/test.
- Macro F1, PR-AUC, Balanced Accuracy.
- Precision/Recall/F1 tung lop.
- Du doan test va xac suat theo tung mau.
- Hyperparameter va thoi gian.

Tong hop:

- `classic_full_runs.csv`: moi run.
- `classic_full_summary.csv`: bang tom tat don gian.
- `classic_full_aggregate.csv`: mean theo model/feature/imputation/imbalance/kind/rate.
- `classic_mcnemar.csv`: McNemar theo tung cau hinh.
- `classic_friedman.json`: Friedman tren ma tran model chinh.
- `report/figs/classic_missingness_curves.pdf`.
- `report/figs/classic_critical_difference.png`.
