"""Generate SAITS-only result rows and figure from the verified experiment snapshot."""
from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT = Path(__file__).resolve().parent
data = json.loads((REPORT / "data/saits_verified_results.json").read_text(encoding="utf-8"))
assert data["verification"]["runs_checked"] == 30
assert not data["protocol"]["smoke"]
results = {row["scenario"]: row for row in data["summary"]}
labels = {
    "point_00": "Không ẩn thêm",
    **{f"{kind}_{rate}": f"{kind.title()} {rate}\\%"
       for kind in ("point", "subsequence", "block") for rate in (10, 30, 50)},
}
baseline = results["point_00"]["test_macro_f1_mean"]
summary_rows, seed_rows = [], []
for scenario, label in labels.items():
    runs = sorted((run for run in data["runs"] if run["scenario"] == scenario), key=lambda run: run["seed"])
    assert [run["seed"] for run in runs] == [1, 2, 42]
    scores = [run["macro_f1_test"] for run in runs]
    mean, std = statistics.mean(scores), statistics.stdev(scores)
    assert abs(mean - results[scenario]["test_macro_f1_mean"]) < 1e-12
    assert abs(std - results[scenario]["test_macro_f1_sample_std"]) < 1e-12
    delta = mean - baseline
    delta_text = "0.0000" if scenario == "point_00" else f"{delta:+.4f}"
    summary_rows.append(f"{label} & ${mean:.4f} \\pm {std:.4f}$ & ${delta_text}$ \\\\")
    epochs = "/".join(str(run["best_epoch"]) for run in runs)
    seed_rows.append(f"{label} & " + " & ".join(f"${score:.4f}$" for score in scores) + f" & {epochs} \\\\")

header = "% Generated from report/data/saits_verified_results.json; SAITS only.\n"
for name, rows in (("saits_results_rows.tex", summary_rows), ("saits_seed_rows.tex", seed_rows)):
    # Keep the closing booktabs rule in the same input as the final row.
    (REPORT / "section" / name).write_text(
        header + "\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8"
    )

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 11,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.0), sharey=True, constrained_layout=True)
colors = {"point": "#2464a8", "subsequence": "#ad4c34", "block": "#207556"}
for ax, kind in zip(axes, ("point", "subsequence", "block")):
    keys = ["point_00", *[f"{kind}_{rate}" for rate in (10, 30, 50)]]
    means = [results[key]["test_macro_f1_mean"] for key in keys]
    stds = [results[key]["test_macro_f1_sample_std"] for key in keys]
    ax.errorbar([0, 10, 30, 50], means, yerr=stds, marker="o", markersize=4,
                capsize=3, linewidth=1.5, color=colors[kind])
    ax.set(title=kind.title(), xlabel="Mức thiếu mục tiêu (%)", xlim=(-4, 54), ylim=(0.70, 1.00))
    ax.set_xticks([0, 10, 30, 50])
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Macro-F1 trên test")
fig.savefig(REPORT / "figs/saits_macro_f1_missingness.png", dpi=300, facecolor="white")
plt.close(fig)
print("Generated 10 summary rows, 10 per-seed rows, and the SAITS Macro-F1 figure.")
