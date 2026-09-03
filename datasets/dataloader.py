"""Lắp ráp DataLoader cho train / validation / test."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.dataset import (
    MultiAugmentDataset,
    MushroomDataset,
    TransformDataset,
    get_targets,
)
from datasets.preprocess_data import build_transforms, resolve_normalization
from utils.utils import seed_worker

logger = logging.getLogger("p4aids")


@dataclass
class DataBundle:
    """Gói dữ liệu trả về cho tools/trainers, kèm metadata cần cho inference."""

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    classes: List[str] = field(default_factory=list)
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    label_mapping: Dict[int, int] = field(default_factory=dict)
    mean: Tuple[float, ...] = ()
    std: Tuple[float, ...] = ()

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def stratified_split(targets: Sequence[int], val_split: float, seed: int = 42
                     ) -> Tuple[List[int], List[int]]:
    """Chia index thành train/val, giữ nguyên tỉ lệ mỗi lớp."""
    rng = np.random.default_rng(seed)
    targets_np = np.asarray(targets)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for label in np.unique(targets_np):
        indices = np.where(targets_np == label)[0]
        rng.shuffle(indices)
        num_val = int(round(len(indices) * val_split))
        # Luôn chừa lại ít nhất 1 mẫu train cho mỗi lớp.
        num_val = min(num_val, max(len(indices) - 1, 0))
        val_indices.extend(indices[:num_val].tolist())
        train_indices.extend(indices[num_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def random_split_indices(size: int, val_split: float, seed: int = 42
                         ) -> Tuple[List[int], List[int]]:
    """Chia ngẫu nhiên không phân tầng."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    num_val = int(round(size * val_split))
    return indices[num_val:].tolist(), indices[:num_val].tolist()


def resolve_label_mapping(classes: Sequence[str],
                          mapping_cfg: Optional[Mapping] = None) -> Dict[int, int]:
    """Ánh xạ chỉ số lớp nội bộ -> nhãn nộp bài.

    ``mapping_cfg`` nhận cả hai dạng:
    - theo tên thư mục lớp: ``{"nấm mỡ": 0, "linh chi trắng": 3}`` (khuyến nghị,
      không phụ thuộc thứ tự sắp xếp thư mục);
    - theo chỉ số lớp: ``{0: 1, 1: 2, 2: 0}`` (giữ tương thích code cũ).
    Lớp nào không khai báo sẽ giữ nguyên chỉ số.
    """
    mapping = {index: index for index in range(len(classes))}
    if not mapping_cfg:
        return mapping

    class_to_idx = {name: index for index, name in enumerate(classes)}
    for key, value in mapping_cfg.items():
        if key in class_to_idx:
            mapping[class_to_idx[key]] = int(value)
            continue
        try:
            index = int(key)
        except (TypeError, ValueError):
            logger.warning("Bỏ qua ánh xạ nhãn cho lớp không tồn tại: '%s'", key)
            continue
        if 0 <= index < len(classes):
            mapping[index] = int(value)
        else:
            logger.warning("Bỏ qua ánh xạ nhãn cho chỉ số ngoài phạm vi: %s", key)
    return mapping


def _make_loader(dataset, batch_size: int, shuffle: bool, data_cfg: Mapping,
                 generator: Optional[torch.Generator] = None) -> DataLoader:
    num_workers = int(data_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        drop_last=bool(data_cfg.get("drop_last", False)) and shuffle,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(cfg: Mapping, cache_path: Optional[str] = None,
                      with_test: bool = True) -> DataBundle:
    """Tạo train/val/test loader từ config.

    Khác biệt quan trọng so với bản cũ: dữ liệu được **chia train/val trước**
    rồi mới augment. Bản cũ nhân bản augment rồi mới chia nên cùng một ảnh gốc
    có thể nằm ở cả hai tập, khiến val accuracy bị thổi phồng.
    """
    data_cfg: Mapping = cfg["data"]
    train_dir = data_cfg.get("train_dir")
    batch_size = int(data_cfg.get("batch_size", 32))
    val_split = float(data_cfg.get("val_split", 0.1))
    split_seed = int(data_cfg.get("split_seed", 42))

    mean, std = resolve_normalization(data_cfg, train_dir=train_dir, cache_path=cache_path)
    transform_set = build_transforms(data_cfg, mean, std)
    eval_transform = transform_set["eval"]
    train_transforms = transform_set["train"]

    bundle = DataBundle(mean=mean, std=std)
    generator = torch.Generator().manual_seed(split_seed)

    if train_dir:
        base_dataset = MushroomDataset(train_dir, transform=None, is_test=False)
        bundle.classes = base_dataset.classes
        bundle.class_to_idx = base_dataset.class_to_idx
        bundle.label_mapping = resolve_label_mapping(base_dataset.classes,
                                                     data_cfg.get("class_to_label"))
        logger.info("Các lớp phát hiện được: %s", base_dataset.classes)

        if val_split > 0 and data_cfg.get("stratified", True):
            train_indices, val_indices = stratified_split(get_targets(base_dataset),
                                                          val_split, split_seed)
        elif val_split > 0:
            train_indices, val_indices = random_split_indices(len(base_dataset),
                                                              val_split, split_seed)
        else:
            train_indices, val_indices = list(range(len(base_dataset))), []

        train_subset = Subset(base_dataset, train_indices)
        if data_cfg.get("use_multi_augment", True):
            num_copies = data_cfg.get("num_augment_copies") or len(train_transforms)
            train_dataset = MultiAugmentDataset(train_subset, train_transforms,
                                                num_copies=int(num_copies))
            logger.info("Multi-augment: %d bản sao / ảnh gốc.", train_dataset.num_copies)
        else:
            train_dataset = TransformDataset(train_subset, train_transforms[0])

        bundle.train_loader = _make_loader(train_dataset, batch_size, True, data_cfg, generator)
        logger.info("Số mẫu train (sau augment): %d", len(train_dataset))

        if val_indices:
            val_dataset = TransformDataset(Subset(base_dataset, val_indices), eval_transform)
            bundle.val_loader = _make_loader(val_dataset, batch_size, False, data_cfg)
            logger.info("Số mẫu validation: %d", len(val_dataset))

    if with_test and data_cfg.get("test_dir"):
        bundle.test_loader = build_test_loader(data_cfg, eval_transform)

    return bundle


def build_test_loader(data_cfg: Mapping, eval_transform) -> DataLoader:
    """Loader cho tập test.

    ``is_test=None`` để dataset tự nhận diện: thư mục ảnh phẳng thì không nhãn,
    thư mục chia theo lớp thì lấy nhãn thật (dùng để chấm điểm offline).
    """
    test_dataset = MushroomDataset(data_cfg["test_dir"], transform=eval_transform, is_test=None)
    logger.info("Số mẫu test: %d", len(test_dataset))
    return _make_loader(test_dataset, int(data_cfg.get("batch_size", 32)), False, data_cfg)
