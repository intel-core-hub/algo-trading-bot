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
    win_rate: float  # ポジションを持っているバーのうちリターンがプラスだった割合(トレード単位の勝率ではない)


def max_drawdown_from_returns(net_returns: pd.Series) -> float:
    """最大ドローダウンを、開始時点のエクイティ1.0を含めて計算する。

    equity_curve = (1+net_returns).cumprod() は最初のバーの時点で既に
    1回分のリターン(コスト込み)を織り込んでいるため、equity_curveの
    cummaxをそのまま基準にすると「1本目のバー自体が最大の下落だった」場合の
    ドローダウンが0%と誤って計算されてしまう(基準となる開始時点の1.0が
    比較対象に含まれていないため)。先頭に1.0を明示的に加えてから計算する。
    """
    if len(net_returns) == 0:
        return 0.0
    equity = (1 + net_returns).cumprod()
    values = pd.Series([1.0, *equity.to_numpy(dtype=float)])
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def _run_returns_backtest(
    raw_returns: pd.Series,
    signal: pd.Series,
    cost_bps: float,
    periods_per_year: int,
    charge_final_close: bool = True,
) -> BacktestResult:
    """signal(ポジション)と、フルロング(position=1)時の各バーのリターン系列から
    BacktestResultを組み立てる共通ロジック。run_backtest(価格ベース)と
    run_cashflow_backtest(ファンディングなどキャッシュフローベース)で共有する。

    cost_bps は turnover(シグナルの変化量)1単位あたりの片道コスト。
    0→1のエントリーで1回、1→0のイグジットでもう1回課金されるので、
    ワンラウンドトリップの実質コストは2*cost_bpsになる。
    charge_final_close=True(デフォルト)の場合、期間の最後にまだポジションが
    残っていれば、それを手仕舞うコストも最終バーに追加で計上する
    (放置すると実際には発生する決済コストを無視してリターンを過大評価してしまう)。
    """
    signal = signal.reindex(raw_returns.index).fillna(0.0).clip(-1, 1)

    position = signal.shift(1).fillna(0.0)
    strategy_returns = position * raw_returns

    turnover = signal.diff().abs().fillna(signal.abs())
    cost = turnover * cost_bps / 10_000
    net_returns = strategy_returns - cost

    n_trades = int((signal.diff().fillna(signal) != 0).sum())

    final_signal = signal.iloc[-1] if len(signal) else 0.0
    if charge_final_close and final_signal != 0:
        net_returns.iloc[-1] -= abs(final_signal) * cost_bps / 10_000
        n_trades += 1

    equity_curve = (1 + net_returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1) if len(equity_curve) else 0.0

    ann_factor = np.sqrt(periods_per_year)
    sharpe = float(net_returns.mean() / net_returns.std() * ann_factor) if net_returns.std() > 0 else 0.0

    max_drawdown = max_drawdown_from_returns(net_returns)

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


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = 24 * 365,
) -> BacktestResult:
    price_returns = close.pct_change().fillna(0.0)
    return _run_returns_backtest(price_returns, signal, fee_bps + slippage_bps, periods_per_year)


def run_cashflow_backtest(
    cashflow_rate: pd.Series,
    signal: pd.Series,
    cost_bps: float = 20.0,
    periods_per_year: int = 3 * 365,
) -> BacktestResult:
    """価格リターンではなく、資金調達率のような「レート型キャッシュフロー」を積み上げる
    戦略向け(例: ファンディングキャリー)。position=1で各バーcashflow_rateをそのまま
    受け取るとみなす。デフォルトのperiods_per_yearは資金調達が8時間毎(年1095回)の想定。
    cost_bpsは片道コスト(ラウンドトリップは2*cost_bps。_run_returns_backtest参照)。
    """
    return _run_returns_backtest(cashflow_rate, signal, cost_bps, periods_per_year)


def run_funding_carry_backtest(
    spot_close: pd.Series,
    perp_price: pd.Series,
    funding_rate: pd.Series,
    signal: pd.Series,
    cost_bps: float = 30.0,
    periods_per_year: int = 3 * 365,
) -> BacktestResult:
    """現物ロング($1) + 無期限先物ショート(証拠金$1、1倍レバレッジ)のベーシスリスクを
    考慮したキャリー戦略バックテスト。run_cashflow_backtest(funding_rateをそのまま
    リターンとみなす簡略版)と違い、現物と先物の価格が完全には連動しない
    (ベーシス変動)ことによる損益も反映する。perp_priceには無期限先物の実際の約定価格
    (fetch_ohlcv market_type="future")を渡すこと — funding rate APIのmark priceは
    清算判定用の指数で実売買価格から乖離することがあり、そのまま使うと見せかけの
    ベーシスリスクで結果が歪む。

    投入資本は$2(現物$1 + 先物証拠金$1、無レバレッジ)とみなし、資本に対するリターンを
    計算する: 0.5*(現物リターン - 先物価格リターン) + 0.5*funding_rate
    (ヘッジ残差の半分 + 証拠金$1に対して発生するfundingの半分)。
    cost_bpsは現物+先物の両レッグ分を合わせた片道コスト
    (ラウンドトリップは2*cost_bps。_run_returns_backtest参照)。
    """
    spot_return = spot_close.pct_change().fillna(0.0)
    perp_price_return = perp_price.pct_change().fillna(0.0)
    combined_return = 0.5 * (spot_return - perp_price_return) + 0.5 * funding_rate
    return _run_returns_backtest(combined_return, signal, cost_bps, periods_per_year)


def train_test_split_by_time(df: pd.DataFrame, test_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]
