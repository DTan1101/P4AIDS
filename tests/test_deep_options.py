import numpy as np
import pytest

from datasets.missingness import MissingnessConfig
from tools import run_deep
from trainers.deep_trainer import DeepRunResult


def test_cli_resolves_data_path_seed_and_pypots_options(tmp_path, monkeypatch):
    config = tmp_path / "deep.yaml"
    config.write_text(f"""experiment:
  seeds: [42, 1, 2]
  output_dir: {tmp_path / 'results'}
data:
  prepared_path: custom-data.npz
deep:
  model: brits
  hidden_size: 7
  lr: 0.002
  pypots:
    reconstruction_weight: 0.5
""")
    loaded, calls = [], []
    def load(path):
        loaded.append(path)
        return dict(X=np.zeros((4, 2, 2)), y=np.array([0, 1, 0, 1]),
                    train_idx=np.array([0, 1]), val_idx=np.array([2]), test_idx=np.array([3]))
    monkeypatch.setattr(run_deep, "load_prepared_data", load)
    missing = MissingnessConfig("point", 0.3)
    monkeypatch.setattr(run_deep, "resolve_masks", lambda *a: {missing: np.ones((4, 2, 2), bool)})
    def train(seeds, *args, **kwargs):
        calls.append((seeds, kwargs))
        return [DeepRunResult(missing, "brits", seeds[0], test_metrics={"macro_f1": 0.5})]
    monkeypatch.setattr(run_deep, "run_deep_multiseed", train)
    assert run_deep.main(["--config", str(config), "--seed", "9", "--hidden-size", "11",
                          "--lr", "0.003"]) == 0
    assert loaded == ["custom-data.npz"]
    assert calls[0][0] == [9]
    assert calls[0][1]["hidden_size"] == 11
    assert calls[0][1]["lr"] == 0.003
    assert calls[0][1]["reconstruction_weight"] == 0.5


@pytest.mark.parametrize("name", ["brits", "grud_pypots", "saits"])
def test_pypots_factory_uses_size_and_lr(name):
    pytest.importorskip("pypots")
    from models.deep_pypots import build_pypots_model
    model = build_pypots_model(name, 4, 2, hidden_size=8, lr=0.004, device="cpu")
    assert model.optimizer.torch_optimizer.param_groups[0]["lr"] == 0.004
    if name == "saits":
        assert model.d_model == 8
    else:
        assert model.rnn_hidden_size == 8
    with pytest.raises(TypeError):
        build_pypots_model(name, 4, 2, typo_option=123)
