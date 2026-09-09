import pandas as pd

from risk import apply_drawdown_stop, volatility_target_position


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


def test_volatility_target_position_is_flat_during_warmup():
    close = pd.Series(100.0, index=_hourly_index(8))
    signal = pd.Series(1.0, index=close.index)

    sized = volatility_target_position(signal, close, vol_window=2, max_leverage=2.0)

    # rolling std needs vol_window non-NaN pct_change values, so the first
    # bars (before the window fills) have no scale yet and must stay flat
    assert (sized.iloc[:2] == 0.0).all()


def test_volatility_target_position_scales_up_to_max_leverage_when_vol_is_zero():
    close = pd.Series(100.0, index=_hourly_index(8))  # zero realized vol throughout
    signal = pd.Series(1.0, index=close.index)

    sized = volatility_target_position(signal, close, vol_window=2, max_leverage=2.0)

    assert (sized.iloc[2:] == 2.0).all()


def test_volatility_target_position_rebalance_every_holds_between_updates():
    close = pd.Series(100.0, index=_hourly_index(8))  # zero vol -> scale is always max_leverage
    signal = pd.Series([0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], index=close.index)

    sized = volatility_target_position(
        signal, close, vol_window=2, max_leverage=2.0, rebalance_every=1
    )
    held = volatility_target_position(
        signal, close, vol_window=2, max_leverage=2.0, rebalance_every=2
    )

    # every bar: sign flips each step once the window has filled
    assert sized.iloc[2] == 2.0 and sized.iloc[3] == -2.0
    # rebalanced every 2 bars: bar 3 should hold bar 2's value instead of flipping
    assert held.iloc[2] == 2.0 and held.iloc[3] == 2.0


def test_apply_drawdown_stop_leaves_signal_untouched_when_never_breached():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=_hourly_index(5))
    signal = pd.Series(1.0, index=close.index)

    out = apply_drawdown_stop(signal, close, max_drawdown=0.5, cooldown_bars=3)

    assert out.equals(signal)


def test_apply_drawdown_stop_flattens_for_trigger_bar_plus_cooldown():
    # crash of 50% at bar 1 breaches the 20% max_drawdown and should force
    # flat for the trigger bar plus cooldown_bars bars after it
    close = pd.Series([100.0, 50.0, 50.0, 50.0, 50.0, 50.0], index=_hourly_index(6))
    signal = pd.Series(1.0, index=close.index)

    out = apply_drawdown_stop(signal, close, max_drawdown=0.2, cooldown_bars=3)

    assert out.iloc[0] == 1.0  # no drawdown yet on the first bar
    assert (out.iloc[1:5] == 0.0).all()  # trigger bar (1) + 3 cooldown bars (2-4)
