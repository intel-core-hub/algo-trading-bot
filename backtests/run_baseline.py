"""ベースライン戦略(MAクロス/モメンタム)をtrain/testに分けて検証する。

Usage:
    python backtests/run_baseline.py
data/ に事前に src/data.py でキャッシュしたOHLCVが必要。
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

DATA_PATH = ROOT / "data" / "binance_BTC-USDT_1h.csv"


def evaluate(name: str, close: pd.Series, signal: pd.Series) -> None:
    result = run_backtest(close, signal)
    print(
        f"{name:12s} | return={result.total_return:+.2%} | sharpe={result.sharpe:+.2f} "
        f"| max_dd={result.max_drawdown:.2%} | trades={result.n_trades} | win_rate={result.win_rate:.2%}"
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    train, test = train_test_split_by_time(df, test_fraction=0.3)

    strategies = {
        "ma_cross": lambda d: ma_cross.generate_signal(d["close"], fast=20, slow=50),
        "momentum": lambda d: momentum.generate_signal(d["close"], lookback=24),
        "mean_reversion": lambda d: mean_reversion.generate_signal(d["close"], window=20, entry_z=2.0, exit_z=0.5),
        "trend_following": lambda d: trend_following.generate_signal(d["close"], fast_days=50, slow_days=200),
    }

    for split_name, split_df in [("train (in-sample)", train), ("test (out-of-sample)", test)]:
        print(f"\n=== {split_name}: {split_df.index[0]} -> {split_df.index[-1]} ({len(split_df)} bars) ===")
        for name, gen in strategies.items():
            signal = gen(split_df)
            evaluate(name, split_df["close"], signal)
        # baseline: buy and hold
        evaluate("buy_and_hold", split_df["close"], pd.Series(1.0, index=split_df.index))


if __name__ == "__main__":
    main()
