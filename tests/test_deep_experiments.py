import json
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasets.cache import save_prepared_data
from datasets.missingness import ALL_CONFIGS, generate_or_load_mask
from utils.utils import dump_config


@pytest.fixture
def campaign(tmp_path):
    data_path = tmp_path / "windows.npz"
    X = np.random.default_rng(1).normal(size=(12, 4, 2)).astype(np.float32)
    save_prepared_data(str(data_path), X, np.arange(12) % 2,
                       pd.date_range("2010-01-01", periods=12, freq="h"),
                       np.arange(6), np.arange(6, 9), np.arange(9, 12))
    mask_dir = tmp_path / "masks"
    for config in ALL_CONFIGS:
        generate_or_load_mask(X, config, 123, str(mask_dir))
    config_path = tmp_path / "config.yaml"
    dump_config({"experiment": {"seeds": [42, 1, 2]},
                 "data": {"prepared_path": str(data_path)},
                 "missing": {"cache_dir": str(mask_dir), "seed": 123},
                 "deep": {"model": "gru_fill", "hidden_size": 4, "epochs": 1,
                          "batch_size": 4, "patience": 1, "lr": 0.001, "device": "cpu"}},
                str(config_path))
    return config_path, tmp_path / "experiment"


def test_dry_run_plans_full_matrix_and_model_overrides(campaign, capsys):
    from tools.run_deep_experiments import main
    config, output = campaign
    assert main(["--config", str(config), "--output-dir", str(output), "--dry-run",
                 "--hidden-size", "8", "--model-opts", "saits.pypots.n_heads=2",
                 "brits.hidden_size=12"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["n_runs"] == 180
    assert len(plan["cases"]) == 10
    assert not output.exists()
    configs = {job["model"]: job["config"] for job in plan["jobs"]}
    assert configs["brits"]["deep"]["hidden_size"] == 12
    assert configs["gru_d"]["deep"]["hidden_size"] == 8
    assert configs["saits"]["deep"]["pypots"]["n_heads"] == 2
    assert configs["gru_fill"]["deep"].get("pypots", {}) == {}
    assert configs["gru_d"]["deep"]["deterministic"] is True


@pytest.mark.parametrize("extra", [
    ["--models", "gru_fill", "gru_fill"],
    ["--seeds", "1", "1"],
    ["--models", "gru_fill", "--model-opts", "saits.hidden_size=8"],
    ["--models", "gru_fill", "--model-opts", "gru_fill.typo=8"],
    ["--model-opts", "gru_fill.model=brits"],
    ["--opts", "deep.deterministic=not-a-bool"],
    ["--epochs", "0"],
    ["--opts", "deep.model=gru_d"],
])
def test_invalid_plan_fails_before_creating_output(campaign, extra):
    from tools.run_deep_experiments import main
    config, output = campaign
    with pytest.raises(ValueError):
        main(["--config", str(config), "--output-dir", str(output), "--dry-run", *extra])
    assert not output.exists()


def test_campaign_runs_two_models_and_saves_provenance(campaign):
    from tools.run_deep_experiments import main
    config, output = campaign
    assert main(["--config", str(config), "--output-dir", str(output),
                 "--models", "gru_fill", "gru_mask_delta", "--seeds", "7", "8",
                 "--missing-kind", "point", "--missing-rate", "0.3"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["n_runs"] == 4
    assert all(job["status"] == "complete" for job in manifest["jobs"])
    assert manifest["provenance"]["data"]["sha256"]
    assert len(manifest["provenance"]["masks"]) == 1
    assert manifest["provenance"]["sources"]["tools/run_deep_experiments.py"]
    assert manifest["environment"]["packages"]["torch"]
    for name in ("torch", "transformers"):
        if name in manifest["environment"]["packages"]:
            assert manifest["environment"]["packages"][name] == importlib.metadata.version(name)
    assert manifest["environment"]["python"]
    assert manifest["environment"]["torch_threads"] >= 1
    assert "CUBLAS_WORKSPACE_CONFIG" in manifest["environment"]["runtime_env"]
    assert (output / "results" / "summary_mean.csv").is_file()
    predictions = list((output / "results" / "deep").rglob("predictions.npz"))
    assert len(predictions) == 4
    assert len(list((output / "logs").glob("*.log"))) == 2
    for path in (output / "results" / "deep").glob("deep_*.json"):
        runs = json.loads(path.read_text())
        assert [run["seed"] for run in runs] == [7, 8]
        assert all(Path(run["checkpoint_path"]).is_file() for run in runs)
    with pytest.raises(FileExistsError):
        main(["--config", str(config), "--output-dir", str(output)])


def test_same_campaign_inputs_reproduce_predictions(campaign):
    from tools.run_deep_experiments import main
    config, output = campaign
    paths = []
    for name in ("first", "second"):
        folder = output / name
        assert main(["--config", str(config), "--output-dir", str(folder),
                     "--models", "gru_d", "--seed", "7", "--missing-kind", "point",
                     "--missing-rate", "0.3"]) == 0
        paths.append(next(folder.rglob("predictions.npz")))
    with np.load(paths[0]) as first, np.load(paths[1]) as second:
        for key in ("val_y_score", "test_y_score", "test_y_pred", "test_idx"):
            np.testing.assert_array_equal(first[key], second[key])


def test_incompatible_mask_fails_before_training(campaign):
    from tools.run_deep_experiments import main
    config, output = campaign
    mask_path = next((config.parent / "masks").glob("mask_point_30_*.npz"))
    np.savez_compressed(mask_path, mask=np.zeros((1, 2, 3), dtype=bool))
    with pytest.raises(ValueError, match="Mask"):
        main(["--config", str(config), "--output-dir", str(output), "--models", "gru_d",
              "--missing-kind", "point", "--missing-rate", "0.3"])
    assert not output.exists()


def test_unknown_pypots_option_fails_before_starting_any_model(campaign):
    pytest.importorskip("pypots")
    from tools.run_deep_experiments import main
    config, output = campaign
    with pytest.raises(ValueError, match="PyPOTS"):
        main(["--config", str(config), "--output-dir", str(output),
              "--models", "gru_d", "saits", "--model-opts", "saits.pypots.typo=123"])
    assert not output.exists()


def test_failure_is_recorded_and_stops_remaining_models(campaign, monkeypatch):
    from tools import run_deep_experiments as runner
    config, output = campaign
    calls = []
    original = runner.subprocess.run

    def fail_training(command, **kwargs):
        if "tools/run_deep.py" in command:
            calls.append(command)
            return runner.subprocess.CompletedProcess(command, 2)
        return original(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fail_training)
    assert runner.main(["--config", str(config), "--output-dir", str(output),
                        "--models", "gru_fill", "gru_d"]) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert [job["status"] for job in manifest["jobs"]] == ["failed", "pending"]
    assert len(calls) == 1
