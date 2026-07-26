"""Metric computation and the metrics.json write/collect roundtrip.

These feed the numbers in RESULTS.md, so a silent bug here is the worst kind.
Values are checked against hand-computed results on tiny arrays.
"""

import math

import pytest

from src.evaluation.eval_utils import (
    _regression_metrics, _classification_metrics, write_run_artifacts, collect_run_metrics,
)


def test_regression_metrics_perfect():
    m = _regression_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert m["r2"] == pytest.approx(1.0)
    assert m["mse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["n"] == 4


def test_regression_metrics_known_values():
    # y_true mean = 2, SS_tot = 2, SS_res = 1 -> r2 = 0.5; mse = mae = 1/3.
    m = _regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert m["r2"] == pytest.approx(0.5)
    assert m["mse"] == pytest.approx(1 / 3)
    assert m["mae"] == pytest.approx(1 / 3)


def test_classification_metrics_perfect():
    m = _classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["acc"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["n"] == 4
    assert m["n_pos"] == 2


def test_write_and_collect_roundtrip(tmp_path):
    rec_mic = write_run_artifacts(
        tmp_path / "ecoli" / "bert", data_name="ecoli", model_name="bert",
        task="mic", split="random", seed=42, features=False,
        y_true=[1.0, 2.0, 3.0], y_pred=[1.0, 2.0, 3.0], sequences=["A", "B", "C"],
    )
    assert rec_mic["r2"] == pytest.approx(1.0)
    assert (tmp_path / "ecoli" / "bert" / "predictions.csv").exists()
    assert (tmp_path / "ecoli" / "bert" / "metrics.json").exists()

    write_run_artifacts(
        tmp_path / "whitelab" / "xgb", data_name="whitelab", model_name="xgb",
        task="hemo", split="cluster", seed=1, features=True,
        y_true=[0, 1, 0, 1], y_pred=[0, 1, 0, 1], y_score=[0.2, 0.8, 0.3, 0.7],
    )

    df = collect_run_metrics(tmp_path)
    assert len(df) == 2
    assert set(df["model"]) == {"bert", "xgb"}
    assert {"task", "split", "seed"} <= set(df.columns)
