"""リスク管理オーバーレイ(ボラティリティ・ターゲティング/ドローダウン・ストップ)の効果を確認する。

これまでの検証で「勝てる方向性シグナル」は見つかっていないが、自動発注に進むには
シグナルの優劣によらずリスク管理層が要る。ここではbuy&holdとtrend_followingに
src/risk.pyのオーバーレイを重ね、リターンそのものは改善しなくても、ボラティリティや
最大ドローダウンといったリスク指標が実際に抑えられるかを確認する。

Usage:
    python backtests/run_risk_managed.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "strategies"))

from backtest import run_backtest, train_test_split_by_time  # noqa: E402
from risk import apply_drawdown_stop, volatility_target_position  # noqa: E402
import trend_following  # noqa: E402

DATA_FILES = {
    "BTC/USDT": ROOT / "data" / "binance_BTC-USDT_1h.csv",
    "ETH/USDT": ROOT / "data" / "binance_ETH-USDT_1h.csv",
}


def evaluate(label: str, close: pd.Series, signal: pd.Series) -> None:
    result = run_backtest(close, signal)
    realized_vol = result.returns.std() * (24 * 365) ** 0.5
    print(
        f"  {label:34s} | return={result.total_return:+8.2%} | sharpe={result.sharpe:+6.2f} "
        f"| ann_vol={realized_vol:6.2%} | max_dd={result.max_drawdown:7.2%} | trades={result.n_trades:4d}"
    )


def main() -> None:
    for symbol, path in DATA_FILES.items():
        if not path.exists():
            print(f"skip {symbol}: {path} not found")
            continue

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        train, test = train_test_split_by_time(df, test_fraction=0.3)

        print(f"\n### {symbol} ###")
        for split_name, split_df in [("train", train), ("test ", test)]:
            print(f"  -- {split_name} {split_df.index[0]} -> {split_df.index[-1]} ({len(split_df)} bars)")
            close = split_df["close"]

            buy_hold = pd.Series(1.0, index=close.index)
            trend = trend_following.generate_signal(close, fast_days=50, slow_days=200)

            for base_name, base_signal in [("buy_and_hold", buy_hold), ("trend_following", trend)]:
                evaluate(base_name, close, base_signal)

                vol_targeted = volatility_target_position(
                    base_signal, close, target_annual_vol=0.3, rebalance_every=24
                )
                evaluate(f"{base_name} + vol_target(30%, daily rebal)", close, vol_targeted)

                dd_stopped = apply_drawdown_stop(vol_targeted, close, max_drawdown=0.20, cooldown_bars=24)
                evaluate(f"{base_name} + vol_target + dd_stop(20%)", close, dd_stopped)


if __name__ == "__main__":
    main()
