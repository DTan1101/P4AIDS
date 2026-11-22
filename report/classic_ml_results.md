# Classic ML - Ket qua va phan tich

## Tom tat

Tong so 66 run Classic ML da chay tren CPU. Tren du lieu day du, XGBoost dung raw lag
24x5 dat 0.9457 Macro F1 va 0.9769 PR-AUC Snow, cao hon ro rang so voi LiteStatTemp
XGBoost 0.8510/0.8454. Tuy nhien, raw lag chi duoc thu tren 0%, point50 va block50.

Trong ma tran LiteStatTemp chinh:

- LightGBM la model tot nhat tren point va subsequence ở 0-50%.
- XGBoost va LightGBM co Macro F1 ngang nhau ở mot so cau hinh block.
- XGBoost median va native gan nhu giong nhau do LiteStatTemp da xu ly NaN va bo sung mask/delta.
- MissForest khong tao loi ich ro re tren LiteStatTemp.
- Random Forest phai xet hau het kem XGBoost/LightGBM ve PR-AUC.

## Du lieu day du

| Model/feature | Macro F1 | PR-AUC | Snow Recall | Thoi gian train (s) |
|---|---:|---:|---:|---:|
| XGBoost raw lag + mask/delta | 0.9457 | 0.9769 | 0.9644 | 9.13 |
| LightGBM LiteStatTemp + mask/delta | 0.8529 | 0.8405 | 0.8312 | 1.88 |
| XGBoost LiteStatTemp + mask/delta | 0.8510 | 0.8454 | 0.9016 | 2.60 |
| CatBoost LiteStatTemp + mask/delta | 0.8446 | 0.8357 | 0.9175 | 10.46 |
| Random Forest LiteStatTemp + mask/delta | 0.8368 | 0.8019 | 0.8433 | 9.27 |
| XGBoost tsfresh + mask/delta | 0.8038 | 0.7736 | 0.8357 | 3.23 |
| XGBoost catch22 + mask/delta | 0.5651 | 0.1596 | 0.4656 | 5.43 |

## Ket qua ma tran LiteStatTemp theo missing

### Point

| Model | 0% | 10% | 30% | 50% |
|---|---:|---:|---:|---:|
| XGBoost native | 0.8510 | 0.8486 | 0.8428 | 0.8366 |
| XGBoost median | 0.8510 | 0.8486 | 0.8428 | 0.8367 |
| Random Forest median | 0.8368 | 0.8335 | 0.8314 | 0.8340 |
| LightGBM native | 0.8529 | 0.8519 | 0.8497 | 0.8434 |
| CatBoost native | 0.8446 | 0.8445 | 0.8350 | 0.8282 |

LightGBM giam tu 0.8529 con 0.8434 tai point50, suy giam nho hon XGBoost 0.0144.

### Subsequence

| Model | 10% | 30% | 50% |
|---|---:|---:|---:|
| XGBoost native | 0.8410 | 0.8370 | 0.8202 |
| XGBoost median | 0.8410 | 0.8350 | 0.8212 |
| Random Forest median | 0.8327 | 0.8251 | 0.8160 |
| LightGBM native | 0.8487 | 0.8415 | 0.8328 |
| CatBoost native | 0.8359 | 0.8322 | 0.8120 |

Subsequence50 la kịch bản suy giam manh nhat cho XGBoost va CatBoost. LightGBM van dat
Macro F1 cao nhat 0.8328.

### Block

| Model | 10% | 30% | 50% |
|---|---:|---:|---:|
| XGBoost native | 0.8541 | 0.8486 | 0.8447 |
| XGBoost median | 0.8541 | 0.8553 | 0.8464 |
| Random Forest median | 0.8434 | 0.8427 | 0.8492 |
| LightGBM native | 0.8582 | 0.8629 | 0.8546 |
| CatBoost native | 0.8493 | 0.8439 | 0.8406 |

Khac voi truc gia thiet ban dau, block missing trong du lieu nay khong gay suy giam lon hon
point/subsequence. Mot so model thậm chí tang PR-AUC tai block30/50. Nguyên nhân co the la
mask va cach sinh block lam thay doi phan bo mau tren test ma khong xoa duoc toan bo tinh
nang LiteStatTemp.

## Ablation feature

Raw lag 24x5 + mask/delta la feature set tot nhat trong cac ablation da chay:

| Cau hinh | LiteStatTemp | Raw lag | Delta |
|---|---:|---:|---:|
| Full | 0.8510 | 0.9457 | +0.0947 |
| Point50 | 0.8366 | 0.9035 | +0.0669 |
| Block50 | 0.8447 | 0.8915 | +0.0468 |

Tsfresh dat 0.8038 tren full va 0.8066 point50. Catch22 dat 0.5651 tren full va thay doi
rat manh o point/block50; ket qua nay duoc ghi ro la can xem lai phuong phap dien va don
vi dang xay dung, khong nen coi la hieu nang thật cua catch22 tren benchmark nay.

## Imputation va imbalance

- Native NaN va median+mask co Macro F1 gan nhu giong nhau tren cac cau hinh LiteStatTemp.
- MissForest tren full va point30 cho ket qua gan XGBoost median/native.
- MissForest block50 dat 0.8470 Macro F1, tuong duong median 0.8464 va native 0.8447.
- SMOTE tang Macro F1 block50 len 0.8736 nhung Snow Recall giam tu xap xi 0.915-0.932
  ve 0.8077. Tren point30 SMOTE dat 0.8473 va Snow Recall 0.6646.
- Tuning Optuna 12 trial dat 0.8392 Macro F1 point30 va 0.8677 block50.

Ket qua SMOTE cho thay oversampling co thể cai thien Macro F1 o block50 nhưng lam xau
Snow Recall, nen can lua chon metric tong hop khi cân bang lop.

## Thong ke

Friedman tren 9 cau hinh missing va 5 bien the model chính cho:

- Statistic: 11.2376.
- p-value: 0.04687.
- Ket luan: co khac biệt có ý nghĩa thống kê giua cac bien the ở muc alpha=0.05.

McNemar da duoc tinh cho cac cap model tren tung cau hinh; cac file chi tiet tai
`outputs/results/classic_full/classic_mcnemar.csv`.

## Chi phi tinh toan

Tong hop thoi gian train/inference trong run:

- LightGBM: khoang 1.6-2.3 giay cho LiteStatTemp.
- XGBoost: khoang 2.5-3.3 giay.
- CatBoost: khoang 7-10 giay.
- Random Forest: khoang 9-10 giay.
- Raw lag XGBoost: khoang 7.4-9.1 giay.
- Catch22: khoang 5-6 giay.
- TsFresh: khoang 3-4 giay.
- MissForest và tuning ton nhieu thoi gian hon do imputation va nhieu fit.

## Gioi han

- Run chinh moi model moi cau hinh chi dung mot seed (42); chua co std nhieu seed.
- Raw lag, catch22, tsfresh, MissForest, SMOTE va tuning moi la ablation vai cau hinh.
- Catch22 can kiem tra lai co che dien va api pycatch22, vi ket qua rat thap.
- Friedman/CD tren cau hinh huyen (kind/rate) khong thay the cho valid rolling-origin folds.
- SMOTE va tuning can mo rong nhieu seed va nhieu cau hinh truoc khi dua vao ket luan cuoi.

## Ket luan

Trong Classic ML hien tai:

1. LightGBM la lua chon can bang tot nhat giua toc do va Macro F1 cho LiteStatTemp.
2. XGBoost la baseline tot va on dinh giua native/median/MissForest.
3. Raw lag 24x5 co loi the lon va nen la feature set uu tien neu cho phep train te hon.
4. Subsequence50 la truong hop suk suy giam ro nhất.
5. Block missing khong gay suy giam lon nhu point/subsequence trong LiteStatTemp matrix.
6. SMOTE can duoc danh gia bang Snow Recall, khong chi Macro F1.
