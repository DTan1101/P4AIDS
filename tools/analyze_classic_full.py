"""Tổng hợp kết quả Classic ML: bảng mean/std, suy giảm missing, McNemar và biểu đồ."""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.stats_tests import (
    critical_difference_diagram,
    friedman_test,
    mcnemar_test,
)


RESULT_DIR = "outputs/results/classic_full"
FIG_DIR = "report/figs"


def load_runs() -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULT_DIR, "*.json"))):
        if os.path.basename(path) == "environment.json":
            continue
        with open(path, "r", encoding="utf-8") as fp:
            run = json.load(fp)
        if "model" in run and "test_metrics" in run:
            runs.append(run)
    return runs


def aggregate(runs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run in runs:
        test = run["test_metrics"]
        snow = test["per_class"]["Snow"]
        rows.append({
            "model": run["model"],
            "feature_set": run["feature_set"],
            "imputation": run["imputation"],
            "imbalance": run["imbalance"],
            "kind": run["missing_kind"],
            "rate": run["missing_rate"],
            "seed": run["seed"],
            "val_macro_f1": run["val_metrics"]["macro_f1"],
            "test_macro_f1": test["macro_f1"],
            "test_pr_auc": test["pr_auc"],
            "test_balanced_accuracy": test["balanced_accuracy"],
            "test_snow_precision": snow["precision"],
            "test_snow_recall": snow["recall"],
            "test_snow_f1": snow["f1"],
            "train_seconds": run["train_seconds"],
        })
    df = pd.DataFrame(rows)
    grouped = [
        "model", "feature_set", "imputation", "imbalance", "kind", "rate"
    ]
    metrics = [
        "val_macro_f1", "test_macro_f1", "test_pr_auc",
        "test_balanced_accuracy", "test_snow_precision",
        "test_snow_recall", "test_snow_f1", "train_seconds",
    ]
    agg = df.groupby(grouped, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") if c[1] else c[0] for c in agg.columns]
    return df, agg


def plot_missingness(df: pd.DataFrame) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    main = df[
        (df["feature_set"] == "lite_mask_delta")
        & (df["kind"].isin(["point", "subsequence", "block"]))
    ]
    main = main.groupby(["model", "kind", "rate"], as_index=False)["test_macro_f1"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, kind in zip(axes, ("point", "subsequence", "block")):
        sub = main[main["kind"] == kind]
        for model, g in sub.groupby("model"):
            ax.plot(g["rate"], g["test_macro_f1"], marker="o", label=model)
        ax.set_title(kind)
        ax.set_xlabel("Missing rate")
        ax.set_xticks([0.1, 0.3, 0.5])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test Macro F1")
    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "classic_missingness_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def mcnemar_pairs(runs: List[Dict]) -> pd.DataFrame:
    candidates = defaultdict(list)
    for run in runs:
        key = (run["missing_kind"], run["missing_rate"])
        candidates[key].append(run)
    rows = []
    for key, run_list in candidates.items():
        by_model = {run["model"] + "__" + run["feature_set"] + "__" +
                    run["imputation"] + "__" + run["imbalance"]: run
                    for run in run_list}
        names = sorted(by_model)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                run_a, run_b = by_model[name_a], by_model[name_b]
                if len(run_a["test_true"]) == len(run_b["test_true"]):
                    y_true = np.asarray(run_a["test_true"])
                    pred_a = np.asarray(run_a["test_predictions"])
                    pred_b = np.asarray(run_b["test_predictions"])
                    if np.array_equal(y_true, np.asarray(run_b["test_true"])):
                        statistic, p_value = mcnemar_test(y_true, pred_a, pred_b)
                        rows.append({
                            "kind": key[0],
                            "rate": key[1],
                            "model_a": name_a,
                            "model_b": name_b,
                            "statistic": statistic,
                            "p_value": p_value,
                        })
    return pd.DataFrame(rows)


def friedman_main_matrix(df: pd.DataFrame) -> Dict[str, float]:
    main = df[
        (df["feature_set"] == "lite_mask_delta")
        & (df["kind"].isin(["point", "subsequence", "block"]))
    ].copy()
    main["model_variant"] = main["model"] + "__" + main["imputation"]
    pivot = main.pivot_table(
        index=["kind", "rate"],
        columns="model_variant",
        values="test_macro_f1",
        aggfunc="mean",
    ).dropna()
    if pivot.shape[1] < 2 or pivot.shape[0] < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}
    statistic, p_value = friedman_test(pivot.to_numpy())
    critical_difference_diagram(
        list(pivot.columns),
        pivot.to_numpy(),
        os.path.join(FIG_DIR, "classic_critical_difference.png"),
    )
    return {"statistic": statistic, "p_value": p_value}


def main() -> int:
    runs = load_runs()
    if not runs:
        print("Không có kết quả Classic ML.")
        return 1
    df, agg = aggregate(runs)
    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULT_DIR, "classic_full_runs.csv"), index=False)
    agg.to_csv(os.path.join(RESULT_DIR, "classic_full_aggregate.csv"), index=False)
    plot_missingness(df)

    pairwise = mcnemar_pairs(runs)
    if not pairwise.empty:
        pairwise.to_csv(os.path.join(RESULT_DIR, "classic_mcnemar.csv"), index=False)
    friedman = friedman_main_matrix(df)
    with open(os.path.join(RESULT_DIR, "classic_friedman.json"), "w", encoding="utf-8") as fp:
        json.dump(friedman, fp, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(runs)} runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
