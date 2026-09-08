"""1h/4h/日足で同じ戦略群を検証し、時間足を粗くするとノイズが減って結果が変わるか確認する。

これまでの検証(run_baseline.py, run_walk_forward.py, run_ml_signal.py)はすべて1h足の
みで行っており、ノイズの多い時間足で戦略もML分類器も一貫したエッジを示せなかった。
本スクリプトは同じ戦略ロジック・同じtrain/test分割手法(70/30、train_test_split_by_time)
を4h足・日足にもそのまま適用し、時間足を粗くするだけで結果が変わるかを切り分ける。

Usage:
    python backtests/run_timeframe_sweep.py
data/ に事前に `python src/data.py <symbol> <timeframe> <total_bars>` でキャッシュした
1h/4h/1d OHLCVが必要。
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

DATASETS = [
    ("BTC/USDT", "1h", ROOT / "data" / "binance_BTC-USDT_1h.csv"),
    ("BTC/USDT", "4h", ROOT / "data" / "binance_BTC-USDT_4h.csv"),
    ("BTC/USDT", "1d", ROOT / "data" / "binance_BTC-USDT_1d.csv"),
    ("ETH/USDT", "1h", ROOT / "data" / "binance_ETH-USDT_1h.csv"),
    ("ETH/USDT", "4h", ROOT / "data" / "binance_ETH-USDT_4h.csv"),
    ("ETH/USDT", "1d", ROOT / "data" / "binance_ETH-USDT_1d.csv"),
]

STRATEGIES = {
    "ma_cross": lambda d: ma_cross.generate_signal(d["close"], fast=20, slow=50),
    "momentum": lambda d: momentum.generate_signal(d["close"], lookback=24),
    "mean_reversion": lambda d: mean_reversion.generate_signal(d["close"], window=20, entry_z=2.0, exit_z=0.5),
    "trend_following": lambda d: trend_following.generate_signal(d["close"], fast_days=50, slow_days=200),
}


def evaluate(label: str, close: pd.Series, signal: pd.Series, periods_per_year: int) -> None:
    result = run_backtest(close, signal, periods_per_year=periods_per_year)
    print(
        f"{label:34s} | return={result.total_return:+8.2%} | sharpe={result.sharpe:+6.2f} "
        f"| max_dd={result.max_drawdown:7.2%} | trades={result.n_trades:4d}"
    )


PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def main() -> None:
    for symbol, timeframe, path in DATASETS:
        if not path.exists():
            print(f"skip {symbol} {timeframe}: {path} not found")
            continue

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        train, test = train_test_split_by_time(df, test_fraction=0.3)
        ppy = PERIODS_PER_YEAR[timeframe]

        print(f"\n### {symbol} {timeframe}: {df.index[0]} -> {df.index[-1]} ({len(df)} bars) ###")
        for split_name, split_df in [("train", train), ("test ", test)]:
            print(f"  -- {split_name} {split_df.index[0]} -> {split_df.index[-1]} ({len(split_df)} bars)")
            for name, gen in STRATEGIES.items():
                signal = gen(split_df)
                evaluate(f"  {name}", split_df["close"], signal, ppy)
            evaluate("  buy_and_hold", split_df["close"], pd.Series(1.0, index=split_df.index), ppy)


if __name__ == "__main__":
    main()
