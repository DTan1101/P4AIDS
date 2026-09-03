# P4AIDS — Mushroom Image Classification (OAI Challenge 2025)

This system performs mushroom image classification using deep mutual learning: multiple backbones are trained simultaneously, learning both from ground-truth labels and from each other's predicted distributions, and then combined into an ensemble during inference.

The legacy code is in `legacy/` (for reference only, and already excluded via `.gitignore`).

## Structure

```
P4AIDS/
├── configs/                # YAML configuration
│   ├── train.yaml
│   └── eval.yaml
├── datasets/               # Data
│   ├── dataset.py          # MushroomDataset + augmentation wrappers
│   ├── preprocess_data.py  # Transform, mean/std computation
│   └── dataloader.py       # Train/val split, DataLoader creation
├── models/                 # Model
│   ├── backbone.py         # torchvision backbone with classifier head removed
│   ├── head.py             # MLP classification head
│   └── model.py            # Classifier = backbone + head, factory helpers
├── trainers/
│   └── mutual_trainer.py   # Deep mutual learning training loop
├── options/                # Command-line parameters
│   ├── train_options.py
│   └── eval_option.py
├── tools/                  # Entry points
│   ├── train.py
│   └── eval.py
├── utils/
│   ├── metrics.py          # AverageMeter, ConfusionMatrix, accuracy
│   └── utils.py            # Seed, device, config, logging, checkpoint
└── outputs/                # Results for each run
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

```
data/
├── train/                  # one subdirectory per class
│   ├── bào ngư xám + trắng/
│   ├── linh chi trắng/
│   ├── nấm mỡ/
│   └── Đùi gà Baby (cắt ngắn)/
└── test/                   # flat images, unlabeled
    ├── 001.jpg
    └── ...
```

The number of classes is inferred automatically from `data/train`, so no code changes are required.

## Training

```bash
python tools/train.py --config configs/train.yaml
```

Any key in the YAML can be overridden from the command line:

```bash
python tools/train.py --epochs 20 --batch-size 32 --lr 3e-4 --no-amp
```

```bash
python tools/train.py --backbones efficientnet_b6 resnet101 --opts train.scheduler.type=cosine
```

Results are stored under `outputs/<experiment.name>/`: `checkpoints/`, `train.log`, `config.yaml`, `history.json`, and `training_history.png`.

## Inference and result export

```bash
python tools/eval.py --name dml_mushroom --test-dir data/test
```

The script automatically finds checkpoints in `outputs/<name>/checkpoints/`, rebuilds the correct architecture and preprocessing from the saved metadata, and writes the CSV output. If the test directory is split by class (with ground-truth labels), it also prints an accuracy / F1 / confusion matrix report.

```bash
python tools/eval.py --checkpoints outputs/run/checkpoints/best_model1_efficientnet_b6.pth --tta
```

## Configuration

Refer directly to the comments in `configs/train.yaml`. A few notable keys:

| Key | Meaning |
| --- | --- |
| `model.backbones` | List of backbones participating in mutual learning (names from torchvision) |
| `model.head` / `model.head_overrides` | Shared head architecture and per-backbone head overrides |
| `train.mutual_weight` | KL weight with each peer (**0 = disable mutual learning**) |
| `train.temperature` | Temperature used to soften the distribution during distillation |
| `data.augmentations` | Transforms used for multi-augment training |
| `data.class_to_label` | Mapping from class directory names to submission label IDs |

> **Note on `data.class_to_label`:** it is currently set to match the class index. Please update it to the exact label mapping from the organizing committee before submitting.

## Differences from `legacy/`

| Issue in the old version | Fix |
| --- | --- |
| Train/val split performed after augmentation duplication, so the same original image appeared in both sets | Split first, stratify by class, then augment |
| `clip_grad_norm_` was called on gradients still scaled by AMP, making it nearly ineffective | Call `scaler.unscale_()` before clipping |
| `calculate_mean_std` computed the std of std across each batch | Compute using `sqrt(E[x²] − E[x]²)` over all pixels with caching |
| `num_classes=3` was fixed while the dataset actually had 4 classes | Infer from the real class directories |
| Corrupt images returned a tensor of shape `3×32×32`, breaking the batch | Replace with a correctly sized white image |
| `MultiAugmentDataset` overwrote `dataset.transform` when sampling | Wrapper applies transforms internally, safe for multiple workers |
| Three class models were nearly identical and hyperparameters were scattered across the code | Unified `Classifier` + YAML config |
| KL was computed manually with `+1e-6` | Use `F.kl_div` on log-softmax for numerically stable behavior |
| `main.py` imported `libs.setup_datav2`, and `test.py` called `trf.get_test_transforms()` — both missing | Removed and replaced with `tools/train.py` and `tools/eval.py` |

## License

For educational purposes only.
