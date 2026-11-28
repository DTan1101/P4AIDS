"""Generate ROCKET tables and report figures."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve


def save_figure(fig, path):

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", nargs="+",
        default=["outputs/results/rocket"]
    )

    parser.add_argument(
        "--out",
        default="outputs/results/rocket_analysis"
    )

    args = parser.parse_args()

    out = Path(args.out)
    figures = out / "figures"

    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    groups = []
    trials = []

    for folder in args.input:

        for path in sorted(
            Path(folder).glob("rocket_*.json")
        ):

            r = json.loads(
                path.read_text(encoding="utf-8")
            )

            m = r["test_metrics"]
            meta = r["metadata"]

            snow = m["per_class"]["Snow"]
            rain = m["per_class"]["Rain"]

            row = {
                "variant": r["variant"],
                "kind": r["config"]["kind"],
                "rate": r["config"]["rate"],
                "seed": r["seed"],
                "tag": meta.get("tag", "main"),

                "channels": "|".join(r["channels"]),
                "mask_delta": r["use_mask_delta"],
                "class_weight": meta["class_weight"],
                "imputer": meta["imputer"],

                "n_kernels": meta["n_kernels"],
                "n_features": meta["n_features"],
                "best_alpha": meta["best_alpha"],
                "threshold": meta["threshold"],

                "val_macro_f1": r["val_metrics"]["macro_f1"],
                "macro_f1": m["macro_f1"],
                "pr_auc": m["pr_auc"],
                "balanced_accuracy": m["balanced_accuracy"],

                "snow_precision": snow["precision"],
                "snow_recall": snow["recall"],
                "snow_f1": snow["f1"],
                "snow_support": snow["support"],

                "rain_precision": rain["precision"],
                "rain_recall": rain["recall"],
                "rain_f1": rain["f1"],

                "training_seconds": meta[
                    "training_tuning_seconds"
                ],

                "inference_ms": meta[
                    "inference_ms_per_sample"
                ],

                "linear_parameters": meta[
                    "linear_parameters"
                ],

                "actual_missing_test": meta[
                    "actual_missing_test"
                ],

                "selected_channel_missing_test": meta[
                    "selected_channel_missing_test"
                ],

                "mask_sha256": meta["mask_sha256"],
                "data_sha256": meta["data_sha256"]
            }

            rows.append(row)

            for name, g in r.get(
                "subgroup_metrics", {}
            ).items():

                groups.append({
                    **row,
                    "subgroup": name,
                    "subgroup_n": g.get("n_samples"),
                    "subgroup_macro_f1": g.get("macro_f1"),
                    "subgroup_pr_auc": g.get("pr_auc")
                })

            for t in r.get("validation_trials", []):

                trials.append({
                    "variant": row["variant"],
                    "kind": row["kind"],
                    "rate": row["rate"],
                    "seed": row["seed"],
                    "tag": row["tag"],
                    **t
                })

            if row["rate"] not in (0.0, 0.5):
                continue

            prefix = (
                f'{row["tag"]}_{row["seed"]}_{path.stem}'
            )

            cm = np.asarray(m["confusion_matrix"])

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")

            for (i, j), value in np.ndenumerate(cm):

                ax.text(
                    j, i, str(value),
                    ha="center", va="center",
                    color=(
                        "white"
                        if value > cm.max() / 2
                        else "black"
                    )
                )

            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["Rain", "Snow"],
                yticklabels=["Rain", "Snow"],
                xlabel="Predicted",
                ylabel="Actual",
                title=prefix
            )

            save_figure(
                fig, figures / f"cm_{prefix}.png"
            )

            pred_path = (
                path.parent / "predictions"
                / f"{path.stem}.csv.gz"
            )

            if pred_path.exists():

                pred = pd.read_csv(pred_path)

                precision, recall, _ = precision_recall_curve(
                    pred["y_true"],
                    pred["decision_score"]
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                ax.step(
                    recall, precision, where="post",
                    label=f'AP={row["pr_auc"]:.4f}'
                )

                ax.axhline(
                    pred["y_true"].mean(),
                    linestyle="--",
                    label="Snow prevalence"
                )

                ax.set(
                    xlabel="Recall",
                    ylabel="Precision",
                    title=prefix
                )

                ax.legend()

                save_figure(
                    fig, figures / f"pr_{prefix}.png"
                )

    if not rows:
        raise SystemExit("No completed ROCKET results")

    df = pd.DataFrame(rows)

    keys = [
        "variant", "kind", "rate", "seed", "tag"
    ]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate experiments")

    if df["data_sha256"].nunique() != 1:
        raise ValueError("Different datasets")

    for _, g in df.groupby(["kind", "rate"]):

        if g["mask_sha256"].nunique() != 1:
            raise ValueError("Different missing masks")

    df.to_csv(out / "metrics.csv", index=False)

    pd.DataFrame(groups).to_csv(
        out / "subgroup_metrics.csv", index=False
    )

    pd.DataFrame(trials).to_csv(
        out / "validation_trials.csv", index=False
    )

    main = df[df["tag"] == "main"].copy()

    metrics = [
        "macro_f1", "pr_auc",
        "balanced_accuracy",
        "snow_precision", "snow_recall", "snow_f1",
        "training_seconds", "inference_ms"
    ]

    settings = [
        "channels", "mask_delta", "class_weight",
        "imputer", "n_kernels"
    ]

    for _, g in main.groupby("variant"):

        if len(g[settings].drop_duplicates()) > 1:
            raise ValueError("Inconsistent main settings")

    if not main.empty:

        grouped = main.groupby([
            "variant", "kind", "rate"
        ])

        summary = grouped[metrics].agg(["mean", "std"])

        summary.columns = [
            "_".join(c) for c in summary.columns
        ]

        summary["n_seeds"] = grouped.size()

        summary.reset_index().to_csv(
            out / "summary.csv", index=False
        )

        baseline = main[main["rate"] == 0]

        baseline.to_csv(
            out / "full_data_comparison.csv",
            index=False
        )

        for metric in [
            "macro_f1", "pr_auc",
            "balanced_accuracy",
            "snow_recall", "snow_f1"
        ]:

            for kind in [
                "point", "subsequence", "block"
            ]:

                fig, ax = plt.subplots(figsize=(7, 5))

                for variant, g in main.groupby("variant"):

                    selected = g[
                        (g["rate"] == 0)
                        | (g["kind"] == kind)
                    ]

                    score = selected.groupby("rate")[metric]

                    mean = score.mean()
                    std = score.std().fillna(0)

                    ax.errorbar(
                        mean.index * 100,
                        mean.values,
                        yerr=std.values,
                        marker="o",
                        capsize=3,
                        label=variant
                    )

                ax.set(
                    xlabel="Missing rate (%)",
                    ylabel=metric,
                    title=f"{metric} | {kind}",
                    ylim=(0, 1.04)
                )

                ax.set_xticks([0, 10, 30, 50])
                ax.grid(alpha=0.25)
                ax.legend()

                save_figure(
                    fig,
                    figures / f"{metric}_{kind}.png"
                )

        if not baseline.empty:

            for metric in [
                "training_seconds", "inference_ms"
            ]:

                fig, ax = plt.subplots(figsize=(6, 4))

                baseline.groupby(
                    "variant"
                )[metric].mean().plot.bar(ax=ax, rot=0)

                ax.set_ylabel(metric)

                save_figure(
                    fig, figures / f"{metric}.png"
                )

    ablation = df[df["tag"] != "main"].copy()

    if not ablation.empty:

        match_keys = [
            "variant", "kind", "rate", "seed"
        ]

        reference = main[
            match_keys + ["macro_f1", "val_macro_f1"]
        ].rename(columns={
            "macro_f1": "reference_macro_f1",
            "val_macro_f1": "reference_val_macro_f1"
        })

        ablation = ablation.merge(
            reference,
            on=match_keys,
            how="left",
            validate="many_to_one"
        )

        ablation["delta_macro_f1"] = (
            ablation["macro_f1"]
            - ablation["reference_macro_f1"]
        )

        ablation.to_csv(
            out / "ablation.csv", index=False
        )

        for (kind, rate), g in ablation.groupby(
            ["kind", "rate"]
        ):

            g = g.dropna(subset=["reference_macro_f1"])

            if g.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 5))

            g.groupby("tag")[
                "delta_macro_f1"
            ].mean().plot.barh(ax=ax)

            ax.axvline(0, color="black")
            ax.set_xlabel("Macro F1 change vs main run")

            save_figure(
                fig,
                figures / f"ablation_{kind}_{rate}.png"
            )

    expected = {("point", 0.0)}

    expected.update(
        (kind, rate)
        for kind in ["point", "subsequence", "block"]
        for rate in [0.1, 0.3, 0.5]
    )

    for variant in ["mini", "multi"]:

        g = main[
            (main["variant"] == variant)
            & (main["seed"] == 42)
        ]

        missing = expected - set(zip(
            g["kind"], g["rate"]
        ))

        if missing:
            print(
                f"WARNING {variant}: missing {sorted(missing)}"
            )

    print(f"Exported {len(df)} experiments to {out}")


if __name__ == "__main__":
    main()
