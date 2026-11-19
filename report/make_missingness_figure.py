"""Sinh hình minh hoạ 3 kiểu mặt nạ thiếu dữ liệu giả lập (Mục 2 báo cáo), trên
cùng một cửa sổ thật (không có NaN tự nhiên, để mặt nạ hiện rõ).

Chạy một lần: python report/make_missingness_figure.py
Xuất: report/figs/missingness_mask_example.pdf
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from datasets.missingness import MissingnessConfig, generate_mask
from datasets.szeged import FEATURE_COLUMNS, clean, load_raw
from datasets.windows import build_windows

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
SHORT_NAMES = ["Temp.", "Humidity", "Wind", "Pressure", "Visib."]


def main() -> None:
    df = clean(load_raw())
    w = build_windows(df, window=24)

    natural_nan = np.isnan(w.X).any(axis=(1, 2))
    sample_idx = np.flatnonzero(~natural_nan)[100]
    X_sample = w.X[sample_idx: sample_idx + 1]

    kinds = [("point", "Point (10%)", 0.10), ("subsequence", "Subsequence (30%)", 0.30),
             ("block", "Block (50%)", 0.50)]
    cmap = ListedColormap(["#C44E52", "#DDDDDD"])  # đỏ = thiếu, xám nhạt = quan sát được

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (kind, title, rate) in zip(axes, kinds):
        config = MissingnessConfig(kind, rate)
        mask = generate_mask(X_sample, config, seed=7)[0]  # (24, 5) bool
        actual_rate = 1.0 - mask.mean()

        ax.imshow(mask.T.astype(int), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{title}\n(thực tế {actual_rate*100:.0f}% thiếu)")
        ax.set_xlabel("Giờ trong cửa sổ (0..23)")
        ax.set_yticks(range(5))
        ax.set_yticklabels(SHORT_NAMES)
        ax.set_xticks([0, 6, 12, 18, 23])

    fig.suptitle("Ba kiểu mất dữ liệu giả lập trên cùng một cửa sổ thật (ô đỏ = bị ẩn)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "missingness_mask_example.pdf"))
    plt.close(fig)
    print("Đã lưu report/figs/missingness_mask_example.pdf")


if __name__ == "__main__":
    main()
