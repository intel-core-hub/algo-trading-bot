"""汎用walk-forwardパラメータ最適化エンジン。

考え方:
- データをtrain窓・test窓のペアに分割し、窓をtest_barsずつ前にずらしながら繰り返す
  (アンカーなしのローリングウィンドウ)。
- 各foldでtrain窓に対してパラメータグリッドをbrute-force探索し、指定メトリクス
  (デフォルトsharpe)が最良のパラメータをtest窓にそのまま適用する。
- test窓のパラメータはtrain窓のみから決めるため、test窓の結果はout-of-sampleとみなせる。
- 全foldのtest窓リターンを連結すれば、単一のtrain/test分割よりも頑健な
  out-of-sample評価になる。
"""

import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from backtest import BacktestResult, max_drawdown_from_returns, run_backtest


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_result: BacktestResult
    test_result: BacktestResult


@dataclass
class WalkForwardReport:
    folds: list[Fold] = field(default_factory=list)
    combined_returns: pd.Series = field(default_factory=pd.Series)

    @property
    def combined_result(self) -> BacktestResult:
        equity_curve = (1 + self.combined_returns).cumprod()
        total_return = float(equity_curve.iloc[-1] - 1) if len(equity_curve) else 0.0
        ann_factor = (24 * 365) ** 0.5
        sharpe = (
            float(self.combined_returns.mean() / self.combined_returns.std() * ann_factor)
            if self.combined_returns.std() > 0
            else 0.0
        )
        max_drawdown = max_drawdown_from_returns(self.combined_returns)
        return BacktestResult(
            equity_curve=equity_curve,
            returns=self.combined_returns,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            n_trades=-1,
            win_rate=float((self.combined_returns > 0).mean()) if len(self.combined_returns) else 0.0,
        )


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid.keys())
    combos = itertools.product(*param_grid.values())
    return [dict(zip(keys, combo)) for combo in combos]


def naive_selector(
    signal_fn: Callable[..., pd.Series],
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    metric: str = "sharpe",
) -> Callable[[pd.DataFrame, dict], float]:
    """train窓全体に対して1回バックテストし、そのメトリクスをそのままスコアにする(素朴な選択)。

    ノイズの多い単一の窓のSharpeを最大化するだけなので、過学習しやすい。
    """

    def selector(train_df: pd.DataFrame, params: dict) -> float:
        signal = signal_fn(train_df["close"], **params)
        result = run_backtest(train_df["close"], signal, fee_bps, slippage_bps)
        return getattr(result, metric)

    return selector


def robust_selector(
    signal_fn: Callable[..., pd.Series],
    n_sub_windows: int = 4,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    metric: str = "sharpe",
    min_trades_per_sub_window: int = 3,
) -> Callable[[pd.DataFrame, dict], float]:
    """train窓をn_sub_windows個の連続する部分期間に分け、各部分でのメトリクスの
    (平均 - 標準偏差)をスコアにする(頑健な選択)。

    単一窓でたまたま良かっただけのパラメータ(サブ期間ごとにばらつきが大きい)を
    ペナルティで弾き、どのサブ期間でも安定して機能するパラメータを優先する。
    取引回数が少なすぎるサブ期間があるパラメータ(ほぼノートレードで指標が不安定になる
    もの)は失格とする。
    """

    def selector(train_df: pd.DataFrame, params: dict) -> float:
        sub_len = len(train_df) // n_sub_windows
        if sub_len < 10:
            sub_windows = [train_df]
        else:
            sub_windows = [
                train_df.iloc[i * sub_len : (i + 1) * sub_len] for i in range(n_sub_windows)
            ]

        sub_scores = []
        for sub_df in sub_windows:
            signal = signal_fn(sub_df["close"], **params)
            result = run_backtest(sub_df["close"], signal, fee_bps, slippage_bps)
            if result.n_trades < min_trades_per_sub_window:
                return float("-inf")
            sub_scores.append(getattr(result, metric))

        scores = pd.Series(sub_scores)
        return float(scores.mean() - scores.std(ddof=0))

    return selector


def walk_forward_optimize(
    df: pd.DataFrame,
    signal_fn: Callable[..., pd.Series],
    param_grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    metric: str = "sharpe",
    param_filter: Callable[[dict], bool] | None = None,
    selector: Callable[[pd.DataFrame, dict], float] | None = None,
) -> WalkForwardReport:
    """selector未指定時はnaive_selector(単一窓のSharpe最大化)を使う。"""
    combos = expand_grid(param_grid)
    if param_filter is not None:
        combos = [p for p in combos if param_filter(p)]
    if not combos:
        raise ValueError("param_grid produced no valid parameter combinations")

    if selector is None:
        selector = naive_selector(signal_fn, fee_bps, slippage_bps, metric)

    folds: list[Fold] = []
    combined_returns: list[pd.Series] = []

    start = 0
    n = len(df)
    while start + train_bars + test_bars <= n:
        train_df = df.iloc[start : start + train_bars]
        test_df = df.iloc[start + train_bars : start + train_bars + test_bars]

        best_params, best_score = None, None
        for params in combos:
            score = selector(train_df, params)
            if best_score is None or score > best_score:
                best_params, best_score = params, score

        train_signal = signal_fn(train_df["close"], **best_params)
        best_train_result = run_backtest(train_df["close"], train_signal, fee_bps, slippage_bps)

        test_signal = signal_fn(test_df["close"], **best_params)
        test_result = run_backtest(test_df["close"], test_signal, fee_bps, slippage_bps)

        folds.append(
            Fold(
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                best_params=best_params,
                train_result=best_train_result,
                test_result=test_result,
            )
        )
        combined_returns.append(test_result.returns)
        start += test_bars

    if not folds:
        raise ValueError("not enough bars for even a single train/test fold")

    return WalkForwardReport(folds=folds, combined_returns=pd.concat(combined_returns))
