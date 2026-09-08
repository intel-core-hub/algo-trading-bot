"""個々には勝てない戦略でも、組み合わせれば(相関が低ければ)リスクが下がるかを検証する。

MAクロス・モメンタム・平均回帰・トレンドフォローの4戦略を等ウェイトで平均し、
1つの「ポートフォリオ」シグナルとして評価する。個々の戦略のリターン相関も確認する
(相関が低い/マイナスなら組み合わせにリスク低減効果が期待できるが、相関が高ければ
「弱い戦略を集めても弱いまま」になりやすい)。

Usage:
    python backtests/run_portfolio.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "strategies"))

from backtest import run_backtest, train_test_split_by_time  # noqa: E402
import ma_cross  # noqa: E402
import mean_reversion  # noqa: E402
import momentum  # noqa: E402
import trend_following  # noqa: E402

DATA_FILES = {
    "BTC/USDT": ROOT / "data" / "binance_BTC-USDT_1h.csv",
    "ETH/USDT": ROOT / "data" / "binance_ETH-USDT_1h.csv",
}

STRATEGIES = {
    "ma_cross": lambda d: ma_cross.generate_signal(d["close"], fast=20, slow=50),
    "momentum": lambda d: momentum.generate_signal(d["close"], lookback=24),
    "mean_reversion": lambda d: mean_reversion.generate_signal(d["close"], window=20, entry_z=2.0, exit_z=0.5),
    "trend_following": lambda d: trend_following.generate_signal(d["close"], fast_days=50, slow_days=200),
}


def evaluate(label: str, close: pd.Series, signal: pd.Series) -> None:
    result = run_backtest(close, signal)
    print(
        f"  {label:18s} | return={result.total_return:+8.2%} | sharpe={result.sharpe:+6.2f} "
        f"| max_dd={result.max_drawdown:7.2%} | trades={result.n_trades:4d}"
    )
    return result


def main() -> None:
    for symbol, path in DATA_FILES.items():
        if not path.exists():
            print(f"skip {symbol}: {path} not found")
            continue

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

        # 個々の戦略のネットリターン相関(全期間で計算)
        returns_by_strategy = {}
        for name, gen in STRATEGIES.items():
            signal = gen(df)
            result = run_backtest(df["close"], signal)
            returns_by_strategy[name] = result.returns
        corr = pd.DataFrame(returns_by_strategy).corr()
        print(f"\n### {symbol}: 戦略間リターン相関(全期間) ###")
        print(corr.round(2).to_string())

        train, test = train_test_split_by_time(df, test_fraction=0.3)
        print(f"\n### {symbol}: train/testでの個別 vs ポートフォリオ ###")
        for split_name, split_df in [("train", train), ("test ", test)]:
            print(f"  -- {split_name} {split_df.index[0]} -> {split_df.index[-1]} ({len(split_df)} bars)")
            signals = {name: gen(split_df) for name, gen in STRATEGIES.items()}
            for name, signal in signals.items():
                evaluate(name, split_df["close"], signal)

            portfolio_signal = pd.concat(signals.values(), axis=1).mean(axis=1).clip(-1, 1)
            evaluate("portfolio(equal-wt)", split_df["close"], portfolio_signal)
            evaluate("buy_and_hold", split_df["close"], pd.Series(1.0, index=split_df.index))


if __name__ == "__main__":
    main()
