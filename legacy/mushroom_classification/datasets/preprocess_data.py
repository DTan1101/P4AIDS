"""Tiền xử lý dữ liệu: bộ transform và thống kê mean/std của tập train."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.dataset import MushroomDataset, TransformDataset
from utils.utils import ensure_dir, load_json, save_json

# mean/std đã tính sẵn trên tập train của cuộc thi, dùng làm giá trị mặc định.
DEFAULT_MEAN: Tuple[float, float, float] = (0.439673, 0.394798, 0.360261)
DEFAULT_STD: Tuple[float, float, float] = (0.184119, 0.177669, 0.170195)

logger = logging.getLogger("p4aids")


class MushroomTransforms:
    """Tập hợp các transform dùng cho train/eval.

    Mỗi transform train là một "khẩu vị" augment khác nhau; khi bật
    multi-augment, mỗi ảnh gốc sẽ được nhân bản qua toàn bộ danh sách này.
    """

    def __init__(self, mean: Sequence[float] = DEFAULT_MEAN,
                 std: Sequence[float] = DEFAULT_STD, image_size: int = 224) -> None:
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.image_size = int(image_size)

    # -- khối dùng lại ------------------------------------------------- #
    @property
    def _size(self) -> Tuple[int, int]:
        return (self.image_size, self.image_size)

    def _to_float_tensor(self) -> List[Callable]:
        return [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)]

    def _normalize(self) -> Callable:
        return transforms.Normalize(self.mean, self.std)

    # -- eval ----------------------------------------------------------- #
    def eval_transform(self) -> Callable:
        """Transform tất định dùng cho validation và test."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self._size),
            self._normalize(),
        ])

    # -- train ---------------------------------------------------------- #
    def flip_crop(self) -> Callable:
        return transforms.Compose(self._to_float_tensor() + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=self._size, scale=(0.8, 1.0)),
            self._normalize(),
        ])

    def color_jitter(self) -> Callable:
        return transforms.Compose(self._to_float_tensor() + [
            transforms.Resize(self._size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            self._normalize(),
        ])

    def random_erasing(self) -> Callable:
        return transforms.Compose(self._to_float_tensor() + [
            transforms.Resize(self._size),
            self._normalize(),
            transforms.RandomErasing(p=1.0),
        ])

    def elastic(self) -> Callable:
        return transforms.Compose(self._to_float_tensor() + [
            transforms.Resize(self._size),
            self._normalize(),
            transforms.ElasticTransform(),
        ])

    def plain(self) -> Callable:
        return transforms.Compose(self._to_float_tensor() + [
            transforms.Resize(self._size),
            self._normalize(),
        ])

    def rotation(self) -> Callable:
        return transforms.Compose(self._to_float_tensor() + [
            transforms.Resize(self._size),
            self._normalize(),
            transforms.RandomRotation(degrees=(0, 90)),
        ])

    # -- truy xuất ------------------------------------------------------ #
    AVAILABLE: Tuple[str, ...] = (
        "flip_crop", "color_jitter", "random_erasing", "elastic", "plain", "rotation",
    )

    def get(self, name: str) -> Callable:
        if name == "eval":
            return self.eval_transform()
        if name not in self.AVAILABLE:
            raise KeyError(f"Transform '{name}' không tồn tại. Hợp lệ: {self.AVAILABLE}")
        return getattr(self, name)()

    def build(self, names: Optional[Sequence[str]] = None) -> List[Callable]:
        """Danh sách transform train theo tên; ``None`` = lấy toàn bộ."""
        return [self.get(name) for name in (names or self.AVAILABLE)]


def compute_mean_std(dataset: MushroomDataset, image_size: int = 224, batch_size: int = 32,
                     num_workers: int = 0) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Tính mean/std theo từng kênh trên toàn bộ dataset.

    Dùng công thức ``std = sqrt(E[x^2] - E[x]^2)`` trên tổng thể pixel nên
    kết quả chính xác và không phụ thuộc batch size (bản cũ lấy std của std
    theo batch nên bị lệch).
    """
    resize = transforms.Compose([transforms.Resize((image_size, image_size)),
                                 transforms.ToTensor()])
    loader = DataLoader(TransformDataset(dataset, resize), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for images, _ in loader:
        images = images.double()
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        pixel_count += images.size(0) * images.size(2) * images.size(3)

    if pixel_count == 0:
        raise RuntimeError("Dataset rỗng, không tính được mean/std.")

    mean = channel_sum / pixel_count
    variance = (channel_sq_sum / pixel_count) - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return tuple(mean.tolist()), tuple(std.tolist())


def resolve_normalization(data_cfg: Mapping, train_dir: Optional[str] = None,
                          cache_path: Optional[str] = None
                          ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Xác định mean/std sẽ dùng, theo thứ tự ưu tiên:

    1. giá trị khai báo tường minh trong config;
    2. cache đã tính ở lần chạy trước;
    3. tính lại từ tập train (nếu ``normalize.auto`` bật);
    4. hằng số mặc định :data:`DEFAULT_MEAN` / :data:`DEFAULT_STD`.
    """
    norm_cfg: Mapping = data_cfg.get("normalize") or {}
    mean, std = norm_cfg.get("mean"), norm_cfg.get("std")
    if mean and std:
        logger.info("Dùng mean/std từ config.")
        return tuple(float(v) for v in mean), tuple(float(v) for v in std)

    if not norm_cfg.get("auto", True) or not train_dir:
        logger.info("Dùng mean/std mặc định của bộ dữ liệu.")
        return DEFAULT_MEAN, DEFAULT_STD

    image_size = int(data_cfg.get("image_size", 224))
    if cache_path and os.path.isfile(cache_path) and not norm_cfg.get("recompute", False):
        cached = load_json(cache_path)
        if cached.get("train_dir") == train_dir and cached.get("image_size") == image_size:
            logger.info("Dùng mean/std từ cache: %s", cache_path)
            return tuple(cached["mean"]), tuple(cached["std"])

    logger.info("Đang tính mean/std trên tập train (chỉ chạy lần đầu)...")
    dataset = MushroomDataset(train_dir, transform=None, is_test=False)
    mean, std = compute_mean_std(
        dataset,
        image_size=image_size,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    logger.info("mean=%s std=%s", [round(v, 6) for v in mean], [round(v, 6) for v in std])

    if cache_path:
        ensure_dir(os.path.dirname(cache_path))
        save_json({"train_dir": train_dir, "image_size": image_size,
                   "mean": list(mean), "std": list(std)}, cache_path)
    return mean, std


def build_transforms(data_cfg: Mapping, mean: Sequence[float], std: Sequence[float]
                     ) -> Dict[str, object]:
    """Tạo sẵn transform train/eval từ config."""
    factory = MushroomTransforms(mean=mean, std=std,
                                 image_size=int(data_cfg.get("image_size", 224)))
    names = data_cfg.get("augmentations")
    return {
        "factory": factory,
        "train": factory.build(names),
        "eval": factory.eval_transform(),
    }
