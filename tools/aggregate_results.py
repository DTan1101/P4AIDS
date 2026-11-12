"""Tổng hợp outputs/results/*.json thành bảng cho Chương 6 + kiểm định thống kê.

    python tools/aggregate_results.py

Đọc toàn bộ kết quả đã lưu bởi ``run_baselines.py``/``run_classic.py``/
``run_rocket.py``/``run_deep.py``, gộp thành một bảng dài (Mục 6.1/6.2), rồi:
  - ghi ``outputs/results/summary.csv`` + ``summary.md`` (tất cả mô hình x cấu hình);
  - vẽ biểu đồ critical-difference (Mục 6.5) từ Macro F1 trên tập test, coi mỗi
    cấu hình (kind, rate) là một "đơn vị so sánh" lặp lại giữa các mô hình -- đây
    là một cách diễn giải hợp lý cho Friedman/Nemenyi khi không có nhiều fold
    thật, cần nêu rõ khi viết Mục 6.5 của báo cáo.

LƯU Ý QUAN TRỌNG: McNemar's test (Mục 6.5) cần dự đoán theo từng mẫu (không chỉ
chỉ số tổng hợp) của hai mô hình trên CÙNG tập test. Các file .json ở đây chỉ lưu
chỉ số tổng hợp (để file nhẹ), nên script này KHÔNG tự chạy McNemar. Khi cần, gọi
``evaluation.stats_tests.mcnemar_test(y_true, pred_a, pred_b)`` trực tiếp sau khi
có hai mảng dự đoán theo từng mẫu từ ``tools/run_*.py`` (có thể nạp lại mô hình
đã lưu hoặc chạy lại inference).
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from evaluation.stats_tests import critical_difference_diagram, friedman_test  # noqa: E402
from utils.utils import ensure_dir, setup_logger  # noqa: E402

RESULTS_DIR = "outputs/results"
SUMMARY_COLUMNS = ("branch", "model", "kind", "rate", "split", "macro_f1", "pr_auc", "balanced_accuracy")


def _row(branch: str, model: str, kind: str, rate: float, split: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "branch": branch, "model": model, "kind": kind, "rate": rate, "split": split,
        "macro_f1": metrics.get("macro_f1"), "pr_auc": metrics.get("pr_auc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
    }


def load_baselines(results_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "baselines", "baselines_*.json"))):
        key = os.path.basename(path)[len("baselines_"):-len(".json")]
        kind, rate = key.rsplit("_", 1)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for model_name, splits in data.items():
            for split in ("val", "test"):
                if split in splits:
                    rows.append(_row("baseline", model_name, kind, int(rate) / 100, split, splits[split]))
    return rows


def load_single_result_files(results_dir: str, branch: str, pattern: str, model_field: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, branch, pattern))):
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        model_name = data.get(model_field, branch)
        kind, rate = data["config"]["kind"], data["config"]["rate"]
        for split in ("val", "test"):
            metrics_key = f"{split}_metrics"
            if metrics_key in data:
                rows.append(_row(branch, model_name, kind, rate, split, data[metrics_key]))
    return rows


def load_classic(results_dir: str) -> List[Dict[str, Any]]:
    rows = load_single_result_files(results_dir, "classic", "classic_*.json", "impute_strategy")
    # tên mô hình đầy đủ: impute_strategy (dễ phân biệt các biến thể trong bảng)
    for row in rows:
        row["model"] = f"xgboost_{row['model']}"
    return rows


def load_rocket(results_dir: str) -> List[Dict[str, Any]]:
    rows = load_single_result_files(results_dir, "rocket", "rocket_*.json", "variant")
    for row in rows:
        row["model"] = f"rocket_{row['model']}"
    return rows


def load_deep(results_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "deep", "deep_*.json"))):
        with open(path, "r", encoding="utf-8") as fp:
            runs = json.load(fp)
        for run in runs:
            kind, rate = run["config"]["kind"], run["config"]["rate"]
            for split in ("val", "test"):
                metrics_key = f"{split}_metrics"
                if metrics_key in run:
                    rows.append(_row("deep", run["model_name"], kind, rate, split, run[metrics_key]))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in SUMMARY_COLUMNS})


def write_markdown(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    header = "| " + " | ".join(SUMMARY_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        values = [str(row.get(c, "")) for c in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def build_cd_diagram(rows: List[Dict[str, Any]], save_path: str) -> None:
    """Macro F1 trên test, mỗi (kind, rate) là một đơn vị so sánh lặp lại (Mục 6.5)."""
    test_rows = [r for r in rows if r["split"] == "test" and r["macro_f1"] is not None]
    models = sorted({r["model"] for r in test_rows})
    configs = sorted({(r["kind"], r["rate"]) for r in test_rows})
    if len(models) < 2 or len(configs) < 2:
        logging.getLogger("p4aids").warning(
            "Không đủ mô hình/cấu hình để chạy Friedman/CD (models=%d, configs=%d).",
            len(models), len(configs),
        )
        return

    score_matrix = np.full((len(configs), len(models)), np.nan)
    lookup = {(r["kind"], r["rate"], r["model"]): r["macro_f1"] for r in test_rows}
    for i, config in enumerate(configs):
        for j, model in enumerate(models):
            score_matrix[i, j] = lookup.get((*config, model), np.nan)

    if np.isnan(score_matrix).any():
        logging.getLogger("p4aids").warning(
            "Một số (model, config) thiếu kết quả -- bỏ qua Friedman/CD (ma trận không đầy đủ)."
        )
        return

    statistic, p_value = friedman_test(score_matrix)
    logging.getLogger("p4aids").info("Friedman: statistic=%.4f, p=%.6f", statistic, p_value)
    critical_difference_diagram(models, score_matrix, save_path)


def main() -> int:
    setup_logger()
    logger = logging.getLogger("p4aids")

    rows = (load_baselines(RESULTS_DIR) + load_classic(RESULTS_DIR) +
            load_rocket(RESULTS_DIR) + load_deep(RESULTS_DIR))
    if not rows:
        logger.warning("Không tìm thấy kết quả nào trong '%s'. Chạy tools/run_*.py trước.", RESULTS_DIR)
        return 1

    write_csv(rows, os.path.join(RESULTS_DIR, "summary.csv"))
    write_markdown(rows, os.path.join(RESULTS_DIR, "summary.md"))
    logger.info("Đã ghi %d dòng kết quả vào '%s/summary.{csv,md}'.", len(rows), RESULTS_DIR)

    build_cd_diagram(rows, os.path.join(RESULTS_DIR, "critical_difference.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
