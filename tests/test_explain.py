"""Kiểm tra evaluation/explain.py báo lỗi rõ ràng khi thiếu thư viện tuỳ chọn.

Trong môi trường test này ``shap``/``captum`` CHƯA được cài (xem requirements.txt
-- hai thư viện này là tuỳ chọn cho Mục 4.7/6.7), nên đúng ra gọi các hàm này
phải báo ``RuntimeError`` kèm hướng dẫn cài đặt, không phải ``ImportError`` trần
hay một lỗi khó hiểu khác.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.explain import integrated_gradients, shap_summary


def test_shap_summary_raises_clear_error_without_shap():
    try:
        import shap  # noqa: F401
        pytest.skip("shap đã được cài trong môi trường này -- bỏ qua test cảnh báo thiếu lib.")
    except ImportError:
        pass

    with pytest.raises(RuntimeError, match="pip install shap"):
        shap_summary(model=None, X=np.zeros((2, 2)), feature_names=["a", "b"])


def test_integrated_gradients_raises_clear_error_without_captum():
    try:
        import captum  # noqa: F401
        pytest.skip("captum đã được cài trong môi trường này -- bỏ qua test cảnh báo thiếu lib.")
    except ImportError:
        pass

    with pytest.raises(RuntimeError, match="pip install captum"):
        integrated_gradients(forward_fn=None, inputs=None)
