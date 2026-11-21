"""Chạy một chiến dịch DL tuần tự, lưu cấu hình/provenance/log và tổng hợp seed.

    python tools/run_deep_experiments.py --output-dir outputs/experiments/deep_report --dry-run
    python tools/run_deep_experiments.py --models gru_d brits --output-dir outputs/experiments/deep_pair
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from datasets.cache import load_prepared_data
from datasets.missingness import FULL_DATA_CONFIG, mask_cache_path
from options.experiment_options import missing_configs_from_args, parse_experiment_options
from tools.run_deep import ALL_MODEL_NAMES, ARG_TO_CONFIG, DEFAULT_CONFIG, add_training_arguments, training_kwargs
from trainers.deep_trainer import CUSTOM_MODEL_NAMES, DEEP_PROTOCOL
from utils.utils import apply_overrides, dump_config, save_json


def _add_arguments(parser):
    add_training_arguments(parser)
    parser.add_argument("--models", nargs="+", choices=["all", *ALL_MODEL_NAMES], default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", required=True, help="Thư mục chiến dịch mới, chưa tồn tại.")
    parser.add_argument("--model-opts", nargs="*", default=[],
                        help="Override riêng: brits.hidden_size=32 saits.pypots.n_heads=4")
    parser.add_argument("--dry-run", action="store_true", help="In kế hoạch JSON, không tạo file/train.")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(cfg, cases):
    path = Path(cfg["data"].get("prepared_path", cfg["data"].get("output_path", "outputs/data/windows.npz"))).resolve()
    data = load_prepared_data(str(path))
    X = data["X"]
    missing = cfg.get("missing", {})
    mask_dir = Path(missing.get("cache_dir", "outputs/masks")).resolve()
    mask_seed = int(missing.get("seed", 123))
    masks = []
    for case in cases:
        if case.rate == 0:
            continue
        mask_path = Path(mask_cache_path(case, mask_seed, len(X), str(mask_dir)))
        with np.load(mask_path) as archive:
            mask = archive["mask"]
        if mask.shape != X.shape or mask.dtype != bool or np.any(mask & np.isnan(X)):
            raise ValueError(f"Mask không tương thích với dữ liệu: {mask_path}")
        masks.append({"path": str(mask_path), "sha256": _sha256(mask_path)})
    sources = {}
    for folder in ("datasets", "models", "trainers", "tools", "options", "utils", "evaluation", "configs"):
        for source in sorted((REPO_ROOT / folder).rglob("*")):
            if source.suffix in (".py", ".yaml"):
                sources[str(source.relative_to(REPO_ROOT))] = _sha256(source)
    sources["requirements.txt"] = _sha256(REPO_ROOT / "requirements.txt")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None
    cfg["data"]["prepared_path"] = str(path)
    cfg.setdefault("missing", {})["cache_dir"] = str(mask_dir)
    return {"git_commit": commit, "git_status": status, "sources": sources,
            "data": {"path": str(path), "sha256": _sha256(path), "shape": list(X.shape)},
            "masks": masks}


def main(argv=None):
    cfg, args = parse_experiment_options(argv, __doc__, DEFAULT_CONFIG, ARG_TO_CONFIG, _add_arguments)
    if any(option.split("=", 1)[0] in ("deep.model", "experiment.output_dir") for option in args.opts or []):
        raise ValueError("Dùng --models và --output-dir để chọn mô hình/thư mục chiến dịch.")
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Chiến dịch đã tồn tại, chọn --output-dir mới: {root}")
    models = list(ALL_MODEL_NAMES) if args.models == ["all"] else args.models
    if "all" in models or len(set(models)) != len(models):
        raise ValueError("--models phải không trùng; 'all' phải đứng riêng.")
    if args.seed is not None and args.seeds is not None:
        raise ValueError("Chỉ dùng một trong --seed hoặc --seeds.")
    seeds = args.seeds if args.seeds is not None else (
        [args.seed] if args.seed is not None else cfg["experiment"].get("seeds", [cfg["experiment"].get("seed", 42)])
    )
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds phải là các số nguyên khác nhau trong [0, 2**32).")
    overrides = {model: [] for model in models}
    for item in args.model_opts:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError("--model-opts phải có dạng model.key=value.")
        model, override = item.split(".", 1)
        if model not in overrides:
            raise ValueError(f"Override cho mô hình không được chọn: {model}")
        if override.split("=", 1)[0] == "model":
            raise ValueError("Dùng --models để chọn mô hình, không override model.")
        overrides[model].append(override)
    cases = missing_configs_from_args(args)
    if args.missing_kind is None and args.missing_rate is None:
        cases = [FULL_DATA_CONFIG, *cases]
    provenance = _provenance(cfg, cases)
    jobs = []
    for model in models:
        resolved = cfg.copy()
        resolved["experiment"].update(seeds=seeds, output_dir=str(root / "results" / "deep"))
        deep = resolved.setdefault("deep", {})
        deep.setdefault("deterministic", True)
        apply_overrides(deep, overrides[model])
        allowed = {value.split(".", 1)[1] for value in ARG_TO_CONFIG.values()} | {"device", "pypots"}
        if set(deep) - allowed:
            raise ValueError(f"Tuỳ chọn deep không hỗ trợ: {set(deep) - allowed}")
        if not isinstance(deep["deterministic"], bool):
            raise ValueError("deep.deterministic phải là true/false.")
        for name, default in (("epochs", 50), ("hidden_size", 64), ("batch_size", 128)):
            if type(deep.get(name, default)) is not int or deep.get(name, default) < 1:
                raise ValueError(f"deep.{name} phải là số nguyên >= 1.")
        if not np.isfinite(deep.get("lr", 0.001)) or deep.get("lr", 0.001) <= 0:
            raise ValueError("deep.lr phải hữu hạn và > 0.")
        if model in CUSTOM_MODEL_NAMES and deep.get("pypots"):
            raise ValueError(f"{model} không dùng deep.pypots; hãy override riêng từng mô hình.")
        deep["model"] = model
        training_kwargs(deep)
        config_path = root / "configs" / f"{model}.yaml"
        command = [sys.executable, "tools/run_deep.py", "--config", str(config_path)]
        for flag, value in (("--missing-kind", args.missing_kind), ("--missing-rate", args.missing_rate)):
            if value is not None:
                command.extend([flag, str(value)])
        jobs.append({"model": model, "config": resolved, "command": command, "status": "pending"})
    manifest = {"protocol": DEEP_PROTOCOL, "n_runs": len(models) * len(cases) * len(seeds),
                "cases": [{"kind": c.kind, "rate": c.rate} for c in cases], "seeds": seeds,
                "jobs": jobs, "provenance": provenance, "status": "planned"}
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    from tools.aggregate_results import main as aggregate
    if any(job["model"] not in CUSTOM_MODEL_NAMES for job in jobs):
        from models.deep_pypots import BRITSClassifier, GRUDClassifier, SAITSClassifier
        constructors = {"brits": BRITSClassifier, "saits": SAITSClassifier, "grud_pypots": GRUDClassifier}
        for job in jobs:
            if job["model"] in constructors:
                unknown = set(job["config"]["deep"].get("pypots", {})) - set(inspect.signature(constructors[job["model"]]).parameters)
                if unknown:
                    raise ValueError(f"Tuỳ chọn PyPOTS không hỗ trợ cho {job['model']}: {unknown}")
    packages = {d.metadata["Name"].lower() for d in importlib.metadata.distributions() if d.metadata["Name"]}
    manifest["environment"] = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "runtime_env": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
        )},
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": {name: importlib.metadata.version(name) for name in sorted(packages)},
    }
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    for job in jobs:
        dump_config(job["config"], str(root / "configs" / f"{job['model']}.yaml"))
    manifest.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    save_json(manifest, str(root / "manifest.json"))
    try:
        for job in jobs:
            job["status"] = "running"
            job["log"] = str(root / "logs" / f"{job['model']}.log")
            save_json(manifest, str(root / "manifest.json"))
            print(f"Running {job['model']} — log: {job['log']}", flush=True)
            start = time.perf_counter()
            with open(job["log"], "w", encoding="utf-8") as log:
                process = subprocess.run(job["command"], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            job.update(returncode=process.returncode, pipeline_seconds=time.perf_counter() - start)
            job["status"] = "complete" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                manifest["status"] = "failed"
                return process.returncode
        code = aggregate(str(root / "results"))
        manifest["status"] = "complete" if code == 0 else "failed"
        return code
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        for job in jobs:
            if job["status"] == "running":
                job["status"] = manifest["status"]
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(manifest, str(root / "manifest.json"))


if __name__ == "__main__":
    raise SystemExit(main())
