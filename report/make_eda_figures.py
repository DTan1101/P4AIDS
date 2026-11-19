"""Sinh các hình EDA cho Mục 2 của báo cáo, từ dữ liệu Szeged thật.

Chạy một lần: python report/make_eda_figures.py
Xuất 3 hình PDF vào report/figs/: eda_snow_by_year.pdf, eda_snow_by_season.pdf,
eda_temperature_by_label.pdf.
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

from datasets.szeged import clean, load_raw

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
plt.rcParams.update({"font.size": 11})


def season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "Đông"
    if month in (3, 4, 5):
        return "Xuân"
    if month in (6, 7, 8):
        return "Hạ"
    return "Thu"


def main() -> None:
    df = clean(load_raw())
    df = df.dropna(subset=["label"]).copy()
    df["year"] = df.index.year
    df["season"] = df.index.month.map(season_of)

    # --- Hình 1: tỷ lệ Snow theo năm ---------------------------------------
    by_year = df[df["year"] >= 2006].groupby("year")["label"].mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(by_year.index, by_year.values, color="#4C72B0")
    ax.set_xlabel("Năm")
    ax.set_ylabel("Tỷ lệ Snow (%)")
    ax.set_title("Tỷ lệ Snow theo năm (2006-2016)")
    ax.set_xticks(by_year.index)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "eda_snow_by_year.pdf"))
    plt.close(fig)

    # --- Hình 2: tỷ lệ Snow theo mùa ----------------------------------------
    order = ["Xuân", "Hạ", "Thu", "Đông"]
    by_season = df.groupby("season")["label"].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    bars = ax.bar(by_season.index, by_season.values, color="#55A868")
    ax.set_xlabel("Mùa")
    ax.set_ylabel("Tỷ lệ Snow (%)")
    ax.set_title("Tỷ lệ Snow theo mùa (gộp 2006-2016)")
    for bar, value in zip(bars, by_season.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "eda_snow_by_season.pdf"))
    plt.close(fig)

    # --- Hình 3: phân bố nhiệt độ theo nhãn, kèm phóng to dải -4..4C --------
    rain_temp = df.loc[df["label"] == 0, "Temperature (C)"]
    snow_temp = df.loc[df["label"] == 1, "Temperature (C)"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bins_full = np.linspace(-25, 40, 66)
    axes[0].hist(rain_temp, bins=bins_full, alpha=0.6, label="Rain", color="#C44E52", density=True)
    axes[0].hist(snow_temp, bins=bins_full, alpha=0.6, label="Snow", color="#4C72B0", density=True)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Nhiệt độ (°C)")
    axes[0].set_ylabel("Mật độ")
    axes[0].set_title("Toàn bộ dải nhiệt độ")
    axes[0].legend()

    bins_zoom = np.linspace(-8, 8, 33)
    axes[1].hist(rain_temp, bins=bins_zoom, alpha=0.6, label="Rain", color="#C44E52", density=True)
    axes[1].hist(snow_temp, bins=bins_zoom, alpha=0.6, label="Snow", color="#4C72B0", density=True)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Nhiệt độ (°C)")
    axes[1].set_title("Phóng to dải -8..8°C")
    axes[1].legend()

    fig.suptitle("Phân bố nhiệt độ theo nhãn Rain/Snow")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "eda_temperature_by_label.pdf"))
    plt.close(fig)

    print("Đã lưu 3 hình vào", FIGS_DIR)


if __name__ == "__main__":
    main()
