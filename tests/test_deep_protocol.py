import numpy as np
import pytest
import torch

from datasets.dl_dataset import prepare_dl_arrays
from models.deep import build_custom_deep_model


def test_normalization_uses_only_observed_train_values():
    X = np.array([[[10., np.nan], [20., np.nan], [999., np.nan]],
                  [[1000., 5.], [2000., 6.], [3000., 7.]]], dtype=np.float32)
    mask = ~np.isnan(X)
    mask[0, 2, 0] = False
    value, _, _, mean, std = prepare_dl_arrays(X, mask, np.array([0]))
    np.testing.assert_allclose(mean, [15., 0.])
    np.testing.assert_allclose(std, [5., 1.])
    np.testing.assert_allclose(value[0, :, 0], [-1., 1., 1.])
    assert np.isfinite(value).all()


@pytest.mark.parametrize("name", ["gru_fill", "gru_mask_delta", "gru_d"])
def test_custom_checkpoint_reproduces_predictions(tmp_path, name):
    from trainers.deep_trainer import run_deep_custom
    from utils.utils import load_checkpoint
    rng = np.random.default_rng(5)
    X = rng.normal(size=(16, 4, 2)).astype(np.float32)
    mask = np.ones_like(X, dtype=bool)
    mask[0] = False
    y = np.arange(16) % 2
    from datasets.missingness import MissingnessConfig
    result = run_deep_custom(X, y, mask, np.arange(8), np.arange(8, 12), np.arange(12, 16),
                             MissingnessConfig("point", 0.3), model_name=name,
                             hidden_size=4, epochs=2, batch_size=4, device="cpu",
                             artifact_dir=str(tmp_path))
    saved = np.load(result.prediction_path)
    np.testing.assert_array_equal(saved["test_idx"], np.arange(12, 16))
    np.testing.assert_array_equal(saved["test_y_true"], y[12:])
    model = build_custom_deep_model(name, 2, hidden_size=4)
    checkpoint = load_checkpoint(result.checkpoint_path, model)
    meta = checkpoint["meta"]
    assert meta["protocol"] == result.protocol
    assert meta["selection_metric"] == "val_macro_f1"
    value, m, d, mean, std = prepare_dl_arrays(X, mask, np.arange(8))
    np.testing.assert_allclose(meta["mean"], mean)
    np.testing.assert_allclose(meta["std"], std)
    from datasets.dl_dataset import compute_delta, forward_fill, normalize
    value = forward_fill(normalize(np.where(mask, X, np.nan), np.array(meta["mean"]),
                                   np.array(meta["std"])), mask)
    m, d = mask.astype(np.float32), compute_delta(mask)
    model.eval()
    with torch.no_grad():
        logits = model(*(torch.from_numpy(a[12:]) for a in (value, m, d)))
    np.testing.assert_allclose(torch.softmax(logits, 1)[:, 1], saved["test_y_score"], rtol=1e-6)
    np.testing.assert_array_equal(logits.argmax(1), saved["test_y_pred"])


@pytest.mark.parametrize("name", ["brits", "saits", "grud_pypots"])
def test_pypots_checkpoint_uses_global_f1_and_reproduces_predictions(tmp_path, name):
    pytest.importorskip("pypots")
    from datasets.dl_dataset import prepare_observed_arrays
    from datasets.missingness import MissingnessConfig
    from models.deep_pypots import build_pypots_model
    from trainers.deep_trainer import run_deep_pypots
    from utils.utils import load_checkpoint
    rng = np.random.default_rng(13)
    X = rng.normal(size=(24, 4, 2)).astype(np.float32)
    mask = rng.random(X.shape) > 0.3
    X[~mask] = np.nan
    y = np.arange(24) % 2
    options = {"n_layers": 1, "n_heads": 2, "d_ffn": 8} if name == "saits" else {}
    result = run_deep_pypots(X, y, mask, np.arange(12), np.arange(12, 18), np.arange(18, 24),
                             MissingnessConfig("block", 0.3), model_name=name,
                             hidden_size=4, lr=0.002, epochs=2, patience=None, batch_size=4,
                             device="cpu", artifact_dir=str(tmp_path), **options)
    assert len(result.history["val_macro_f1"]) == 2
    assert result.val_metrics["macro_f1"] == pytest.approx(max(result.history["val_macro_f1"]))
    assert result.best_epoch == int(np.argmax(result.history["val_macro_f1"]))
    saved = np.load(result.prediction_path)
    checkpoint = load_checkpoint(result.checkpoint_path)
    meta = checkpoint["meta"]
    model = build_pypots_model(name, meta["n_steps"], meta["n_features"],
                                class_weight=torch.tensor(meta["class_weight"]),
                                device="cpu", **meta["model_kwargs"])
    load_checkpoint(result.checkpoint_path, model.model)
    observed, mean, std = prepare_observed_arrays(X, mask, np.arange(12))
    np.testing.assert_allclose(meta["mean"], mean)
    np.testing.assert_allclose(meta["std"], std)
    from datasets.dl_dataset import normalize
    observed = normalize(np.where(mask, X, np.nan), np.array(meta["mean"]), np.array(meta["std"]))
    predicted = model.predict({"X": observed[18:]})
    np.testing.assert_allclose(predicted["classification_proba"][:, 1], saved["test_y_score"], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(predicted["classification"], saved["test_y_pred"])
    np.testing.assert_array_equal(saved["test_idx"], np.arange(18, 24))


def test_pypots_early_stopping_and_errors_propagate(monkeypatch):
    pytest.importorskip("pypots")
    from models import deep_pypots
    X = np.zeros((8, 4, 2), dtype=np.float32)
    y = np.arange(8) % 2
    model = deep_pypots.build_pypots_model("brits", 4, 2, hidden_size=4, epochs=5,
                                           patience=1, batch_size=2, device="cpu")
    monkeypatch.setattr(deep_pypots, "macro_f1", lambda *a: 0.5)
    model.fit({"X": X[:4], "y": y[:4]}, val_set={"X": X[4:], "y": y[4:]})
    assert len(model.history["val_macro_f1"]) == 2
    assert model.best_epoch == 0
    def fail(*args, **kwargs):
        raise RuntimeError("training failed")
    monkeypatch.setattr(model.model, "forward", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        model.fit({"X": X[:4], "y": y[:4]}, val_set={"X": X[4:], "y": y[4:]})
