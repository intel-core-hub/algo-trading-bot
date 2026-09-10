import pandas as pd

from trend_following import generate_signal


def test_daily_close_signal_is_not_visible_until_next_day():
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    # Daily closes: day1=100, day2=200, day3=50.
    values = [100.0] * 24 + [200.0] * 24 + [50.0] * 24
    close = pd.Series(values, index=idx)

    signal = generate_signal(close, fast_days=1, slow_days=2)

    # Day 2's bullish close cannot be used during day 2 itself.
    assert (signal.loc["2024-01-02"] == 0.0).all()
    # It becomes known on day 3 and is then forward-filled intraday.
    assert (signal.loc["2024-01-03"] == 1.0).all()
