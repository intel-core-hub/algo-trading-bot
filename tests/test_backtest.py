import pandas as pd

from backtest import run_backtest


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


def test_flat_signal_has_zero_return_and_no_trades():
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=_hourly_index(4))
    signal = pd.Series(0.0, index=close.index)

    result = run_backtest(close, signal)

    assert result.total_return == 0.0
    assert result.n_trades == 0


def test_long_and_hold_return_matches_compounded_price_return_at_zero_cost():
    close = pd.Series([100.0, 110.0, 121.0], index=_hourly_index(3))
    signal = pd.Series(1.0, index=close.index)

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0)

    expected_return = 1.1 * 1.1 - 1
    assert abs(result.total_return - expected_return) < 1e-9
    # entry + forced close at the end of the window (still positioned)
    assert result.n_trades == 2


def test_final_open_position_is_charged_a_closing_cost():
    close = pd.Series([100.0] * 5, index=_hourly_index(5))
    signal = pd.Series(1.0, index=close.index)

    result = run_backtest(close, signal, fee_bps=10.0, slippage_bps=0.0)

    # no price movement at all, so any negative return is pure cost drag
    assert result.total_return < 0
    # entry cost + forced exit cost at the final bar
    assert result.n_trades == 2
    assert result.returns.iloc[-1] < 0


def test_no_extra_closing_cost_when_already_flat_at_end():
    close = pd.Series([100.0] * 5, index=_hourly_index(5))
    signal = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=close.index)

    result = run_backtest(close, signal, fee_bps=10.0, slippage_bps=0.0)

    # entry + exit only - no forced close since position is already 0 at the end
    assert result.n_trades == 2
    assert result.returns.iloc[-1] == 0.0


def test_win_rate_is_computed_per_closed_trade_not_per_positive_bar():
    # One trade: two winning holding bars (+10%, +10%) followed by an exit bar
    # that loses roughly 50%. Most of the *bars* while positioned were positive,
    # but the entry->exit trade itself is a clear loser, so win_rate must be 0%.
    close = pd.Series([100.0, 110.0, 121.0, 60.0], index=_hourly_index(4))
    signal = pd.Series([1.0, 1.0, 1.0, 0.0], index=close.index)

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0)

    assert result.win_rate == 0.0


def test_win_rate_counts_each_reversal_as_a_separate_trade():
    # trade 1: long from bar0, reversed to short at bar2 (a losing long trade:
    # 100->90 while long). trade 2: short from bar2, closed flat at bar3 on a
    # further decline (a winning short trade: 90->80).
    close = pd.Series([100.0, 100.0, 90.0, 80.0], index=_hourly_index(4))
    signal = pd.Series([1.0, 1.0, -1.0, 0.0], index=close.index)

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0)

    assert result.win_rate == 0.5
