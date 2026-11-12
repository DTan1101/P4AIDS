"""Giải thích mô hình (Mục 4.7 / 6.7): SHAP cho mô hình cây, Integrated Gradients cho DL.

Cả hai hàm đòi hỏi thư viện tuỳ chọn (``shap``, ``captum``) không được cài trong
lần refactor này (nặng, chỉ cần khi thực sự chạy bước giải thích mô hình ở
Chương 6) -- import được bọc trong try/except với thông báo rõ cách cài.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import numpy as np


def shap_summary(model, X: np.ndarray, feature_names: Sequence[str],
                  save_path: Optional[str] = None, max_display: int = 20) -> np.ndarray:
    """SHAP values cho một mô hình dạng cây (ví dụ XGBoost) bằng ``TreeExplainer``.

    Trả về ma trận shap values ``(n_samples, n_features)``; nếu ``save_path``
    được cung cấp, vẽ thêm summary plot và lưu ra file (headless, dùng backend Agg).
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap_summary() cần thư viện 'shap': pip install shap") from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X, feature_names=list(feature_names),
                           max_display=max_display, show=False)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return np.asarray(shap_values)


def integrated_gradients(forward_fn: Callable[["torch.Tensor"], "torch.Tensor"],  # noqa: F821
                          inputs: "torch.Tensor", target: int = 1,  # noqa: F821
                          baseline: Optional["torch.Tensor"] = None,  # noqa: F821
                          n_steps: int = 50) -> "torch.Tensor":  # noqa: F821
    """Integrated Gradients (Sundararajan et al., 2017) cho một mô hình PyTorch.

    ``forward_fn`` nhận một tensor đầu vào đã được người gọi chuẩn bị sẵn (ví
    dụ ghép value/mask/delta thành một tensor) và trả về logits -- hàm này
    không phụ thuộc kiến trúc cụ thể (GRU-D/BRITS/SAITS), chỉ bọc captum.
    """
    try:
        import torch
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError(
            "integrated_gradients() cần 'captum' (và torch): pip install captum"
        ) from exc

    if baseline is None:
        baseline = torch.zeros_like(inputs)

    ig = IntegratedGradients(forward_fn)
    attributions, _ = ig.attribute(inputs, baselines=baseline, target=target,
                                    n_steps=n_steps, return_convergence_delta=True)
    return attributions
