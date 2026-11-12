"""Định nghĩa Dataset cho bộ ảnh nấm (OAI Challenge).

- :class:`MushroomDataset` đọc ảnh từ thư mục, tự nhận biết tập train
  (mỗi lớp một thư mục con) hay tập test (ảnh nằm phẳng, không nhãn).
- :class:`TransformDataset` / :class:`MultiAugmentDataset` là các wrapper
  áp transform lên một dataset trả về ảnh PIL, nhờ vậy việc chia train/val
  diễn ra trên ảnh gốc trước khi augment (tránh rò rỉ dữ liệu).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, Subset

IMG_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
TEST_LABEL = -1  # nhãn giả cho ảnh không có ground-truth

logger = logging.getLogger("p4aids")


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMG_EXTENSIONS)


class MushroomDataset(Dataset):
    """Dataset ảnh nấm.

    Args:
        root_dir: thư mục dữ liệu.
        transform: transform áp lên ảnh PIL; để ``None`` sẽ trả về ảnh PIL gốc.
        is_test: ép chế độ test. Mặc định ``None`` = tự phát hiện.
        fallback_size: kích thước ảnh trắng thay thế khi file lỗi, giúp
            batch không bị lệch shape lúc collate.
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 is_test: Optional[bool] = None, fallback_size: int = 224) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.fallback_size = fallback_size
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: dict = {}

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: '{root_dir}'")

        entries = sorted(os.listdir(root_dir))
        if is_test is None:
            is_test = not any(os.path.isdir(os.path.join(root_dir, e)) for e in entries)
        self.is_test = is_test

        if is_test:
            self._scan_test_dir(entries)
        else:
            self._scan_train_dir(entries)

        if not self.samples:
            raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong '{root_dir}'")

    # ------------------------------------------------------------------ #
    def _scan_test_dir(self, entries: Sequence[str]) -> None:
        self.classes = []
        self.class_to_idx = {}
        for name in entries:
            path = os.path.join(self.root_dir, name)
            if is_image_file(name) and os.path.isfile(path):
                self.samples.append((path, TEST_LABEL))

    def _scan_train_dir(self, entries: Sequence[str]) -> None:
        self.classes = sorted(e for e in entries if os.path.isdir(os.path.join(self.root_dir, e)))
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in sorted(os.listdir(class_dir)):
                image_path = os.path.join(class_dir, image_name)
                if is_image_file(image_name) and os.path.isfile(image_path):
                    self.samples.append((image_path, self.class_to_idx[class_name]))

    # ------------------------------------------------------------------ #
    @property
    def targets(self) -> List[int]:
        return [label for _, label in self.samples]

    @property
    def image_paths(self) -> List[str]:
        return [path for path, _ in self.samples]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # ảnh hỏng: thay bằng ảnh trắng để không vỡ batch
            logger.warning("Lỗi khi đọc ảnh '%s': %s", image_path, exc)
            image = Image.new("RGB", (self.fallback_size, self.fallback_size))

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __repr__(self) -> str:  # pragma: no cover - chỉ để log
        mode = "test" if self.is_test else "train"
        return (f"MushroomDataset(root='{self.root_dir}', mode={mode}, "
                f"samples={len(self.samples)}, classes={len(self.classes)})")


class TransformDataset(Dataset):
    """Áp một transform cố định lên dataset gốc trả về ảnh PIL."""

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MultiAugmentDataset(Dataset):
    """Nhân bản dataset với nhiều transform khác nhau.

    Với ``n`` transform, dataset dài gấp ``n`` lần: mẫu thứ ``i`` của bản sao
    thứ ``k`` là ảnh gốc ``i`` được augment bằng ``transforms_list[k]``.
    Wrapper tự áp transform (không ghi đè ``dataset.transform`` như bản cũ),
    nên an toàn khi dùng nhiều worker.
    """

    def __init__(self, dataset: Dataset, transforms_list: Sequence[Callable],
                 num_copies: Optional[int] = None) -> None:
        if not transforms_list:
            raise ValueError("transforms_list không được rỗng.")
        if num_copies is None:
            num_copies = len(transforms_list)
        if num_copies > len(transforms_list):
            raise ValueError(
                f"num_copies={num_copies} lớn hơn số transform ({len(transforms_list)})."
            )

        self.dataset = dataset
        self.transforms_list = list(transforms_list)[:num_copies]
        self.num_copies = num_copies

    def __len__(self) -> int:
        return len(self.dataset) * self.num_copies

    def __getitem__(self, index: int):
        base_size = len(self.dataset)
        image, label = self.dataset[index % base_size]
        transform = self.transforms_list[index // base_size]
        if transform is not None:
            image = transform(image)
        return image, label


def get_targets(dataset: Dataset) -> List[int]:
    """Lấy danh sách nhãn của dataset, hỗ trợ cả ``Subset`` lồng nhau."""
    if isinstance(dataset, Subset):
        parent = get_targets(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    if isinstance(dataset, (TransformDataset, MultiAugmentDataset)):
        targets = get_targets(dataset.dataset)
        if isinstance(dataset, MultiAugmentDataset):
            return targets * dataset.num_copies
        return targets
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    return [label for _, label in dataset]


def get_image_paths(dataset: Dataset) -> List[str]:
    """Lấy đường dẫn ảnh theo đúng thứ tự index của dataset (để ghi submission)."""
    if isinstance(dataset, Subset):
        parent = get_image_paths(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    if isinstance(dataset, (TransformDataset, MultiAugmentDataset)):
        paths = get_image_paths(dataset.dataset)
        if isinstance(dataset, MultiAugmentDataset):
            return paths * dataset.num_copies
        return paths
    if hasattr(dataset, "image_paths"):
        return list(dataset.image_paths)
    raise AttributeError(f"{type(dataset).__name__} không cung cấp được đường dẫn ảnh.")
