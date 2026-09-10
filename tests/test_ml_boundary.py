"""backtests/run_ml_signal.py はパッケージ化されていないスクリプトなので、
importlibでファイルパスから直接読み込んでテストする。"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "backtests" / "run_ml_signal.py"
spec = importlib.util.spec_from_file_location("run_ml_signal", MODULE_PATH)
run_ml_signal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_ml_signal)


def test_purge_training_boundary_removes_rows_whose_label_can_reach_test():
    idx = pd.date_range("2026-01-01", periods=30, freq="h")
    train = pd.DataFrame({"x": range(30)}, index=idx)

    purged = run_ml_signal.purge_training_boundary(train, horizon=12)

    assert len(purged) == 18
    assert purged.index[-1] == idx[17]


def test_purge_training_boundary_rejects_too_short_window():
    idx = pd.date_range("2026-01-01", periods=12, freq="h")
    train = pd.DataFrame({"x": range(12)}, index=idx)

    try:
        run_ml_signal.purge_training_boundary(train, horizon=12)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_walk_forward_ml_fits_only_purged_training_rows(monkeypatch):
    fit_lengths = []

    class FakeModel:
        feature_importances_ = np.array([1.0] * len(run_ml_signal.FEATURE_COLUMNS))

        def fit(self, x, y):
            fit_lengths.append(len(x))
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(run_ml_signal, "make_model", FakeModel)
    idx = pd.date_range("2024-01-01", periods=3000, freq="h")
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    df = pd.DataFrame(
        {
            "close": prices,
            "open": prices,
            "high": prices,
            "low": prices,
            "volume": rng.uniform(1.0, 2.0, len(idx)),
        },
        index=idx,
    )

    report = run_ml_signal.walk_forward_ml(df, train_bars=2000, test_bars=500)

    assert report.folds
    assert fit_lengths
    # 各foldのtrain窓はtrain_bars本だが、末尾HORIZON本を切り落として学習する
    assert all(length == 2000 - run_ml_signal.HORIZON for length in fit_lengths)
