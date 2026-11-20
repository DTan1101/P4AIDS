import json

import numpy as np
import pytest

from tools import aggregate_results as aggregate


def test_deep_summary_preserves_seeds_and_uses_mean_for_cd(tmp_path, monkeypatch):
    folder = tmp_path / "deep"
    folder.mkdir()
    runs = [{"config": {"kind": "point", "rate": 0.3}, "model_name": "gru_fill",
             "seed": seed, "test_metrics": {"macro_f1": score}}
            for seed, score in [(42, 0.2), (1, 0.8)]]
    (folder / "deep_gru_fill_point_30.json").write_text(json.dumps(runs))
    rows = aggregate.load_deep(str(tmp_path))
    assert [r["seed"] for r in rows] == [42, 1]
    summary = aggregate.aggregate_seeds(rows)
    assert summary[0]["n_seeds"] == 2
    assert summary[0]["macro_f1"] == pytest.approx(0.5)
    assert summary[0]["macro_f1_std"] == pytest.approx(np.std([0.2, 0.8], ddof=1))
    assert summary[0]["pr_auc"] is None

    all_rows = rows + [dict(r, model=model, rate=rate) for r in rows
                       for model, rate in [("gru_fill", 0.5), ("brits", 0.3),
                                           ("brits", 0.5), ("saits", 0.3), ("saits", 0.5)]]
    captured = []
    monkeypatch.setattr(aggregate, "friedman_test", lambda matrix: (0.0, 1.0))
    monkeypatch.setattr(aggregate, "critical_difference_diagram",
                        lambda models, matrix, path: captured.append(matrix))
    aggregate.build_cd_diagram(all_rows, str(tmp_path / "cd.png"))
    np.testing.assert_allclose(captured[0], np.full((2, 3), 0.5))


def test_summary_keeps_protocols_separate_and_rejects_duplicate_seed():
    row = aggregate._row("deep", "gru_d", "block", 0.5, "test", {"macro_f1": 0.7})
    row.update(seed=42, protocol="deep-v2")
    with pytest.raises(ValueError, match="seed"):
        aggregate.aggregate_seeds([row, row])
    summary = aggregate.aggregate_seeds([row, dict(row, protocol="legacy")])
    assert len(summary) == 2
    assert all(r["n_seeds"] == 1 and r["macro_f1_std"] == 0.0 for r in summary)
