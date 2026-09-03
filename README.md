# P4AIDS — Phân loại ảnh nấm (OAI Challenge 2025)

Hệ thống phân loại ảnh nấm bằng **deep mutual learning**: nhiều backbone được
huấn luyện đồng thời, vừa học từ nhãn thật vừa học từ phân phối dự đoán của
nhau, rồi kết hợp thành ensemble khi suy luận.

Code cũ nằm ở `legacy/` (chỉ để tham khảo, đã được `.gitignore`).

## Cấu trúc

```
P4AIDS/
├── configs/                # Cấu hình YAML
│   ├── train.yaml
│   └── eval.yaml
├── datasets/               # Dữ liệu
│   ├── dataset.py          # MushroomDataset + các wrapper augment
│   ├── preprocess_data.py  # Transform, tính mean/std
│   └── dataloader.py       # Chia train/val, dựng DataLoader
├── models/                 # Mô hình
│   ├── backbone.py         # Backbone torchvision, bỏ lớp phân loại gốc
│   ├── head.py             # Head MLP phân loại
│   └── model.py            # Classifier = backbone + head, các factory
├── trainers/
│   └── mutual_trainer.py   # Vòng huấn luyện deep mutual learning
├── options/                # Tham số dòng lệnh
│   ├── train_options.py
│   └── eval_option.py
├── tools/                  # Điểm vào
│   ├── train.py
│   └── eval.py
├── utils/
│   ├── metrics.py          # AverageMeter, ConfusionMatrix, accuracy
│   └── utils.py            # Seed, device, config, logging, checkpoint
└── outputs/                # Kết quả mỗi lần chạy
```

## Cài đặt

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

```
data/
├── train/                  # mỗi lớp một thư mục con
│   ├── bào ngư xám + trắng/
│   ├── linh chi trắng/
│   ├── nấm mỡ/
│   └── Đùi gà Baby (cắt ngắn)/
└── test/                   # ảnh phẳng, không nhãn
    ├── 001.jpg
    └── ...
```

Số lớp được lấy tự động từ `data/train`, không cần sửa code.

## Huấn luyện

```bash
python tools/train.py --config configs/train.yaml
```

Mọi khoá trong YAML đều ghi đè được từ dòng lệnh:

```bash
python tools/train.py --epochs 20 --batch-size 32 --lr 3e-4 --no-amp
```

```bash
python tools/train.py --backbones efficientnet_b6 resnet101 --opts train.scheduler.type=cosine
```

Kết quả nằm ở `outputs/<experiment.name>/`: `checkpoints/`, `train.log`,
`config.yaml`, `history.json`, `training_history.png`.

## Suy luận & xuất kết quả

```bash
python tools/eval.py --name dml_mushroom --test-dir data/test
```

Script tự dò checkpoint trong `outputs/<name>/checkpoints/`, dựng lại đúng
kiến trúc và tiền xử lý từ metadata đã lưu, rồi ghi CSV. Nếu thư mục test
được chia theo lớp (có nhãn thật), sẽ in thêm báo cáo accuracy / F1 / ma trận
nhầm lẫn.

```bash
python tools/eval.py --checkpoints outputs/run/checkpoints/best_model1_efficientnet_b6.pth --tta
```

## Cấu hình

Xem chú thích trực tiếp trong `configs/train.yaml`. Vài khoá đáng chú ý:

| Khoá | Ý nghĩa |
| --- | --- |
| `model.backbones` | Danh sách backbone tham gia mutual learning (tên theo torchvision) |
| `model.head` / `model.head_overrides` | Kiến trúc head chung và head riêng cho từng backbone |
| `train.mutual_weight` | Trọng số KL với **mỗi** peer (0 = tắt mutual learning) |
| `train.temperature` | Nhiệt độ làm mềm phân phối khi chưng cất |
| `data.augmentations` | Các transform dùng cho multi-augment |
| `data.class_to_label` | Ánh xạ tên thư mục lớp sang mã nhãn nộp bài |

> **Lưu ý về `data.class_to_label`:** hiện đang để trùng với chỉ số lớp.
> Hãy sửa lại theo đúng bảng mã nhãn của ban tổ chức trước khi nộp bài.

## Khác biệt so với `legacy/`

| Vấn đề ở bản cũ | Cách xử lý |
| --- | --- |
| Chia train/val **sau** khi nhân bản augment, cùng một ảnh gốc nằm ở cả hai tập | Chia trước, phân tầng theo lớp, rồi mới augment |
| `clip_grad_norm_` gọi trên gradient còn đang scale bởi AMP nên gần như vô hiệu | `scaler.unscale_()` trước khi clip |
| `calculate_mean_std` lấy std của std theo batch | Tính theo `sqrt(E[x²] − E[x]²)` trên toàn bộ pixel, có cache |
| `num_classes=3` cố định trong khi dữ liệu có 4 lớp | Suy ra từ số thư mục lớp thật |
| Ảnh lỗi trả về tensor `3×32×32` làm vỡ batch | Thay bằng ảnh trắng đúng kích thước |
| `MultiAugmentDataset` ghi đè `dataset.transform` khi lấy mẫu | Wrapper tự áp transform, an toàn với nhiều worker |
| Ba class model gần như trùng nhau; siêu tham số nằm rải rác trong code | `Classifier` dùng chung + config YAML |
| KL tính tay kèm `+1e-6` | `F.kl_div` trên log-softmax, ổn định số học |
| `main.py` import `libs.setup_datav2`, `test.py` gọi `trf.get_test_transforms()` — đều không tồn tại | Đã bỏ, thay bằng `tools/train.py` và `tools/eval.py` |

## Giấy phép

Chỉ dùng cho mục đích học tập.
