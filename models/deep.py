"""GRU-Fill, GRU+mask/Δt, GRU-D cài từ đầu bằng PyTorch (Mục 4.6.2 báo cáo).

Cả ba đều nhận cùng một chữ ký ``forward(value, mask, delta) -> logits`` để
:mod:`trainers.deep_trainer` dùng chung một vòng lặp huấn luyện.
"""

from __future__ import annotations

import torch
from torch import nn


class GRUFill(nn.Module):
    """GRU chuẩn trên dữ liệu đã forward-fill; không dùng mask/delta (baseline DL)."""

    def __init__(self, n_features: int, hidden_size: int = 64, n_classes: int = 2,
                 num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, value: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        del mask, delta  # không dùng, giữ lại để đồng nhất chữ ký forward()
        _, h_n = self.gru(value)
        return self.classifier(h_n[-1])


class GRUMaskDelta(nn.Module):
    """GRU-Fill mở rộng: nối thêm kênh mask và delta vào đầu vào (Mục 4.6.2)."""

    def __init__(self, n_features: int, hidden_size: int = 64, n_classes: int = 2,
                 num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features * 3, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, value: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([value, mask, delta], dim=-1)
        _, h_n = self.gru(x)
        return self.classifier(h_n[-1])


class GRUD(nn.Module):
    """GRU-D (Che et al., 2018), bản đơn giản hoá: decay giá trị đầu vào và decay
    trạng thái ẩn theo delta, dùng :class:`torch.nn.GRUCell` lặp theo từng bước thời
    gian (Mục 4.6.2). Giả định ``value`` đã chuẩn hoá z-score (nên dùng x̄=0 học được
    thay vì trung bình thực nghiệm cố định).
    """

    def __init__(self, n_features: int, hidden_size: int = 64, n_classes: int = 2) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.x_mean = nn.Parameter(torch.zeros(n_features))
        self.decay_x = nn.Linear(n_features, n_features)
        self.decay_h = nn.Linear(n_features, hidden_size)
        self.gru_cell = nn.GRUCell(n_features * 3, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, value: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps, _ = value.shape
        h = value.new_zeros(batch_size, self.hidden_size)
        x_last = value.new_zeros(batch_size, self.n_features)

        for t in range(n_steps):
            x_t, m_t, d_t = value[:, t, :], mask[:, t, :], delta[:, t, :]
            gamma_x = torch.exp(-torch.relu(self.decay_x(d_t)))
            gamma_h = torch.exp(-torch.relu(self.decay_h(d_t)))

            x_last = m_t * x_t + (1 - m_t) * x_last
            x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_last + (1 - gamma_x) * self.x_mean)

            h = gamma_h * h
            gru_input = torch.cat([x_hat, m_t, d_t], dim=-1)
            h = self.gru_cell(gru_input, h)

        return self.classifier(h)


CUSTOM_DEEP_MODELS = {"gru_fill": GRUFill, "gru_mask_delta": GRUMaskDelta, "gru_d": GRUD}


def build_custom_deep_model(name: str, n_features: int, hidden_size: int = 64,
                             n_classes: int = 2, **kwargs) -> nn.Module:
    if name not in CUSTOM_DEEP_MODELS:
        raise ValueError(f"Mô hình không hợp lệ: '{name}'. Hỗ trợ: {list(CUSTOM_DEEP_MODELS)}")
    cls = CUSTOM_DEEP_MODELS[name]
    if name == "gru_d":
        return cls(n_features, hidden_size, n_classes)
    return cls(n_features, hidden_size, n_classes, **kwargs)
