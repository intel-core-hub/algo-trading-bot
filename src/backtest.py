"""シンプルで透明なベクトル化バックテストエンジン。

前提:
- signal は各バーの「期末に保有したいポジション」を表す(-1: ショート, 0: ノーポジ, 1: ロング)。
- 実際の約定は「そのバーの終値」で行われるとみなす(次バー始値ではない、簡略化した近似)。
- 手数料とスリッページはポジション変化があったバーにのみ発生する。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    win_rate: float


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = 24 * 365,
) -> BacktestResult:
    signal = signal.reindex(close.index).fillna(0.0).clip(-1, 1)
    price_returns = close.pct_change().fillna(0.0)

    position = signal.shift(1).fillna(0.0)
    strategy_returns = position * price_returns

    turnover = signal.diff().abs().fillna(signal.abs())
    cost = turnover * (fee_bps + slippage_bps) / 10_000
    net_returns = strategy_returns - cost

    equity_curve = (1 + net_returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1)

    ann_factor = np.sqrt(periods_per_year)
    sharpe = float(net_returns.mean() / net_returns.std() * ann_factor) if net_returns.std() > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min())

    n_trades = int((signal.diff().fillna(signal) != 0).sum())
    win_rate = float((net_returns[position != 0] > 0).mean()) if (position != 0).any() else 0.0

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
        win_rate=win_rate,
    )


def train_test_split_by_time(df: pd.DataFrame, test_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]
