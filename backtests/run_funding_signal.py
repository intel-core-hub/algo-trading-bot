"""ファンディングレート逆張り戦略(価格以外のデータ)をtrain/test分割で検証する。

パラメータ(window=90, entry_z=1.5, exit_z=0.5)は事前にひとつだけ決め打ちし、
グリッドサーチはしない(walk-forwardの検証で「train窓のベストを選ぶ」操作自体が
過学習することを既に確認済みのため、価格以外データの検証では素朴なオーバーフィットを
避ける)。

Usage:
    python backtests/run_funding_signal.py
data/ に事前に
`python src/data.py <symbol> 1h 20000` と `python src/data.py funding <symbol>:USDT 3000`
でOHLCV・ファンディングレートをキャッシュしておく必要がある。
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "strategies"))

from backtest import run_backtest, train_test_split_by_time  # noqa: E402
import funding_contrarian  # noqa: E402

SYMBOLS = {
    "BTC/USDT": (
        ROOT / "data" / "binance_BTC-USDT_1h.csv",
        ROOT / "data" / "binance_BTC-USDT-USDT_funding.csv",
    ),
    "ETH/USDT": (
        ROOT / "data" / "binance_ETH-USDT_1h.csv",
        ROOT / "data" / "binance_ETH-USDT-USDT_funding.csv",
    ),
}


def evaluate(label: str, close: pd.Series, signal: pd.Series) -> None:
    result = run_backtest(close, signal)
    print(
        f"  {label:16s} | return={result.total_return:+8.2%} | sharpe={result.sharpe:+6.2f} "
        f"| max_dd={result.max_drawdown:7.2%} | trades={result.n_trades:4d}"
    )


def main() -> None:
    for symbol, (ohlcv_path, funding_path) in SYMBOLS.items():
        if not ohlcv_path.exists() or not funding_path.exists():
            print(f"skip {symbol}: missing data (run src/data.py first)")
            continue

        ohlcv = pd.read_csv(ohlcv_path, index_col="timestamp", parse_dates=True)
        funding = pd.read_csv(funding_path, index_col="timestamp", parse_dates=True)["funding_rate"]

        # ファンディングは8時間毎なので、直近値を1h足にforward-fillして揃える。
        funding_hourly = funding.reindex(ohlcv.index, method="ffill")
        df = ohlcv.copy()
        df["funding_rate"] = funding_hourly
        df = df.dropna(subset=["funding_rate"])

        train, test = train_test_split_by_time(df, test_fraction=0.3)
        print(f"\n### {symbol}: {df.index[0]} -> {df.index[-1]} ({len(df)} bars, funding-aligned) ###")

        for split_name, split_df in [("train", train), ("test ", test)]:
            print(f"  -- {split_name} {split_df.index[0]} -> {split_df.index[-1]} ({len(split_df)} bars)")
            signal = funding_contrarian.generate_signal(split_df["funding_rate"])
            evaluate("funding_contrarian", split_df["close"], signal)
            evaluate("buy_and_hold", split_df["close"], pd.Series(1.0, index=split_df.index))


if __name__ == "__main__":
    main()
