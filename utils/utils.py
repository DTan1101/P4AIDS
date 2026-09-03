"""Tiện ích dùng chung cho toàn bộ project.

Gom các thao tác lặp lại ở nhiều nơi: seeding, chọn device, đọc/ghi config,
logging, lưu & nạp checkpoint và vẽ biểu đồ lịch sử huấn luyện.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # PyYAML là dependency bắt buộc để đọc configs/*.yaml
    import yaml
except ImportError:  # pragma: no cover - chỉ xảy ra khi thiếu dependency
    yaml = None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class ConfigDict(dict):
    """dict cho phép truy cập theo thuộc tính: ``cfg.data.batch_size``."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"Config không có khoá '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = to_config(value)

    def __delattr__(self, key: str) -> None:
        del self[key]

    def copy(self) -> "ConfigDict":
        return to_config(json.loads(json.dumps(self, default=str)))


def to_config(obj: Any) -> Any:
    """Chuyển đệ quy dict/list thường thành :class:`ConfigDict`."""
    if isinstance(obj, ConfigDict):
        return obj
    if isinstance(obj, Mapping):
        return ConfigDict({key: to_config(value) for key, value in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [to_config(value) for value in obj]
    return obj


def load_yaml(path: str) -> ConfigDict:
    """Đọc file YAML thành :class:`ConfigDict`."""
    if yaml is None:
        raise ImportError("Cần cài PyYAML để đọc file config: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config '{path}' phải là một mapping ở cấp cao nhất.")
    return to_config(data)


def dump_config(cfg: Mapping, path: str) -> None:
    """Lưu lại config đã resolve để tái lập thí nghiệm."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fp:
        if yaml is not None:
            yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), fp,
                           allow_unicode=True, sort_keys=False)
        else:  # pragma: no cover
            json.dump(cfg, fp, ensure_ascii=False, indent=2, default=str)


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` vào ``base`` theo chiều sâu (in-place)."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_update(base[key], value)
        else:
            base[key] = to_config(value)
    return base


def parse_value(text: str) -> Any:
    """Ép chuỗi từ CLI về kiểu Python (int/float/bool/list/None/str)."""
    if yaml is not None:
        try:
            return to_config(yaml.safe_load(text))
        except Exception:
            return text
    return text


def get_nested(cfg: Mapping, dotted_key: str, default: Any = None) -> Any:
    """Lấy giá trị theo khoá lồng nhau, ví dụ ``"train.optimizer.lr"``."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Gán giá trị theo khoá lồng nhau, tự tạo các cấp trung gian."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), Mapping):
            node[part] = ConfigDict()
        node = node[part]
    node[parts[-1]] = to_config(value)


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Áp dụng các override dạng ``key.path=value`` từ dòng lệnh."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' phải có dạng key.path=value")
        key, raw_value = item.split("=", 1)
        set_nested(cfg, key.strip(), parse_value(raw_value.strip()))
    return cfg


def override_from_args(cfg: Dict[str, Any], args: Any,
                      mapping: Mapping[str, str]) -> Dict[str, Any]:
    """Đẩy các tham số CLI đã khai báo vào config.

    ``mapping`` ánh xạ tên thuộc tính của argparse sang khoá lồng nhau trong
    config; giá trị ``None`` (người dùng không truyền) sẽ được bỏ qua để giữ
    nguyên thiết lập trong file YAML.
    """
    for arg_name, dotted_key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            set_nested(cfg, dotted_key, value)
    return cfg


# --------------------------------------------------------------------------- #
# Reproducibility & device
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Cố định seed cho random / numpy / torch để kết quả tái lập được."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed cho từng worker của DataLoader (dùng với ``worker_init_fn``)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Trả về device thực dùng; ``auto`` sẽ ưu tiên CUDA nếu khả dụng."""
    if preferred in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.getLogger("p4aids").warning("CUDA không khả dụng, chuyển sang CPU.")
        return torch.device("cpu")
    return device


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def ensure_dir(path: Optional[str]) -> str:
    """Tạo thư mục nếu chưa tồn tại và trả lại đường dẫn."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path or "."


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def setup_logger(name: str = "p4aids", log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Logger ghi ra stdout và (tuỳ chọn) ra file, không nhân đôi handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file:
        log_file = os.path.abspath(log_file)
        already = any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file
                      for h in logger.handlers)
        if not already:
            ensure_dir(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
_STATE_DICT_KEYS = ("model", "model_state_dict", "state_dict")


def save_checkpoint(path: str, model: torch.nn.Module,
                    meta: Optional[Mapping[str, Any]] = None, **extra: Any) -> str:
    """Lưu checkpoint kèm metadata (backbone, danh sách lớp, mean/std...)."""
    ensure_dir(os.path.dirname(path))
    payload: Dict[str, Any] = {"model": model.state_dict(), "meta": dict(meta or {})}
    payload.update(extra)
    torch.save(payload, path)
    return path


def _torch_load(path: str, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 chưa có tham số weights_only
        return torch.load(path, map_location=map_location)


def load_checkpoint(path: str, model: Optional[torch.nn.Module] = None,
                    map_location: Any = "cpu", strict: bool = True) -> Dict[str, Any]:
    """Nạp checkpoint; hỗ trợ cả file chỉ chứa ``state_dict`` kiểu cũ."""
    checkpoint = _torch_load(path, map_location)

    if isinstance(checkpoint, Mapping):
        state_dict = next((checkpoint[k] for k in _STATE_DICT_KEYS if k in checkpoint), None)
        if state_dict is None:  # file chỉ chứa state_dict thuần
            state_dict, checkpoint = checkpoint, {"model": checkpoint}
    else:
        raise ValueError(f"Checkpoint '{path}' không đúng định dạng.")

    if model is not None:
        model.load_state_dict(state_dict, strict=strict)

    result = dict(checkpoint)
    result["model"] = state_dict
    result.setdefault("meta", {})
    return result


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.nn.Parameter] = model.parameters()
    return sum(p.numel() for p in params if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def plot_history(history: Mapping[str, Sequence[float]], save_path: str) -> Optional[str]:
    """Vẽ loss/accuracy theo epoch và lưu ra ảnh (chạy được ở môi trường headless)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - thiếu matplotlib thì bỏ qua
        logging.getLogger("p4aids").warning("Không có matplotlib, bỏ qua vẽ biểu đồ.")
        return None

    ensure_dir(os.path.dirname(save_path))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    for key, label in (("train_loss", "Train Loss"), ("val_loss", "Val Loss")):
        if history.get(key):
            axes[0].plot(history[key], label=label)
    axes[0].set_title("Loss vs Epochs")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    for key, label in (("train_acc", "Train Accuracy"), ("val_acc", "Val Accuracy")):
        if history.get(key):
            axes[1].plot(history[key], label=label)
    axes[1].set_title("Accuracy vs Epochs")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    return save_path
