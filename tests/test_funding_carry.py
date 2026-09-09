import pandas as pd

from funding_carry import generate_signal


def _8h_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="8h")


def test_enters_after_warmup_when_funding_persistently_positive():
    funding = pd.Series(0.001, index=_8h_index(50))

    signal = generate_signal(funding, window=21, threshold=0.0)

    assert (signal.iloc[22:] == 1.0).all()


def test_stays_flat_when_funding_persistently_negative():
    funding = pd.Series(-0.001, index=_8h_index(50))

    signal = generate_signal(funding, window=21, threshold=0.0)

    assert (signal.iloc[22:] == 0.0).all()


def test_insufficient_history_returns_flat_signal():
    funding = pd.Series(0.001, index=_8h_index(5))  # shorter than window=21

    signal = generate_signal(funding, window=21, threshold=0.0)

    assert (signal == 0.0).all()
