"""Kiểm định ý nghĩa thống kê (Mục 5.4 / 6.5): McNemar, Friedman, biểu đồ CD."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger("p4aids")

#: Giá trị tới hạn Nemenyi q_alpha=0.05 cho k=2..10 mô hình (Demšar, 2006, Table 5).
NEMENYI_Q_ALPHA_005: Dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int],
                  pred_b: Sequence[int]) -> Tuple[float, float]:
    """So sánh hai mô hình trên cùng tập test bằng McNemar's test (McNemar, 1947).

    Trả về ``(statistic, p_value)``. Dùng hiệu chỉnh liên tục (``correction=True``)
    và xấp xỉ chi-square (``exact=False``) vì số mẫu bất đồng thường lớn với tập
    test ~17 nghìn mẫu của báo cáo.
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true

    both_correct = int(np.sum(correct_a & correct_b))
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    if only_a + only_b == 0:
        # Không có cặp bất đồng nào (hai mô hình đúng/sai giống hệt nhau trên mọi
        # mẫu): thống kê hiệu chỉnh liên tục của statsmodels chia cho (only_a+only_b)
        # = 0, cho p-value vô nghĩa (0.0) thay vì đúng ra phải là "không có bằng
        # chứng khác biệt" (p=1.0). Trả thẳng kết quả này, không gọi statsmodels.
        return 0.0, 1.0

    table = [[both_correct, only_a], [only_b, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)
    return float(result.statistic), float(result.pvalue)


def friedman_test(score_matrix: np.ndarray) -> Tuple[float, float]:
    """Kiểm định Friedman trên ma trận điểm số ``(n_folds, n_models)``."""
    score_matrix = np.asarray(score_matrix)
    if score_matrix.ndim != 2 or score_matrix.shape[1] < 2:
        raise ValueError("score_matrix phải có shape (n_folds, n_models >= 2).")
    columns = [score_matrix[:, i] for i in range(score_matrix.shape[1])]
    statistic, p_value = scipy_stats.friedmanchisquare(*columns)
    return float(statistic), float(p_value)


def compute_average_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """Hạng trung bình mỗi mô hình qua các fold (hạng 1 = tốt nhất, điểm cao hơn)."""
    # rankdata xếp hạng tăng dần nên đảo dấu để điểm cao hơn có hạng nhỏ hơn.
    ranks = np.apply_along_axis(scipy_stats.rankdata, 1, -np.asarray(score_matrix))
    return ranks.mean(axis=0)


def critical_difference(n_models: int, n_folds: int, alpha: float = 0.05) -> float:
    """Critical Difference (CD) của kiểm định Nemenyi (Demšar, 2006, công thức 6)."""
    if alpha != 0.05:
        raise NotImplementedError("Chỉ hỗ trợ alpha=0.05 (bảng NEMENYI_Q_ALPHA_005).")
    if n_models not in NEMENYI_Q_ALPHA_005:
        raise ValueError(f"Chưa có giá trị q_alpha cho {n_models} mô hình (hỗ trợ 2-10).")
    q_alpha = NEMENYI_Q_ALPHA_005[n_models]
    return float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_folds)))


def critical_difference_diagram(model_names: Sequence[str], score_matrix: np.ndarray,
                                  save_path: str, alpha: float = 0.05) -> str:
    """Vẽ biểu đồ critical difference (hạng trung bình + thanh CD) và lưu ra file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_ranks = compute_average_ranks(score_matrix)
    n_models, n_folds = len(model_names), score_matrix.shape[0]
    cd = critical_difference(n_models, n_folds, alpha)

    order = np.argsort(avg_ranks)
    sorted_names = [model_names[i] for i in order]
    sorted_ranks = avg_ranks[order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * n_models + 1.5))
    y_positions = np.arange(n_models)[::-1]
    ax.scatter(sorted_ranks, y_positions, color="black", zorder=3)
    for y, rank, name in zip(y_positions, sorted_ranks, sorted_names):
        ax.text(rank, y + 0.15, f"{name} ({rank:.2f})", ha="center", fontsize=9)

    best_rank = sorted_ranks[0]
    for y, rank in zip(y_positions, sorted_ranks):
        if rank <= best_rank + cd:
            ax.plot([best_rank, best_rank + cd], [y, y], color="red", linewidth=4, alpha=0.4, zorder=1)

    ax.set_yticks([])
    ax.set_xlabel("Hạng trung bình (thấp hơn = tốt hơn)")
    ax.set_title(f"Critical Difference diagram (CD={cd:.3f}, Nemenyi α={alpha})")
    ax.invert_xaxis()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Đã lưu biểu đồ CD vào '%s' (CD=%.3f).", save_path, cd)
    return save_path
