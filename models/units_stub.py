"""Chỗ dành cho UniTS (Mục 4.6.2/4.6.4 báo cáo) -- CHƯA triển khai.

Report mô tả UniTS là hướng mở rộng "nếu đủ thời gian/tài nguyên" (fine-tune
một mô hình nền tảng chuỗi thời gian đa nhiệm cho tác vụ phân loại). Việc này
đòi hỏi tải checkpoint pretrained và một pipeline fine-tuning riêng, nằm ngoài
phạm vi của lần refactor này. Mục 7.3 của báo cáo cũng liệt kê các hướng phát
triển khác (Neural CDE, nhiều trạm, Edge) theo cùng tinh thần "chưa làm, có
chỗ để làm tiếp".
"""

from __future__ import annotations


def build_units_model(*args, **kwargs):
    raise NotImplementedError(
        "UniTS chưa được tích hợp. Cần: (1) chọn checkpoint pretrained phù hợp, "
        "(2) viết pipeline fine-tune riêng (Mục 4.6.4), (3) adapter chuyển đổi "
        "(value, mask, delta) của pipeline này sang định dạng đầu vào của UniTS."
    )
