"""ファンディング・キャリー(現物ロング+無期限先物ショートのデルタニュートラル)を検証する。

価格方向は予測せず、資金調達率そのものをリターン源泉とする(run_cashflow_backtest)。
2パターンを比較する:
- always_on_carry: 常にポジションを建てっぱなし(コストは最初の建玉時のみ)
- conditional_carry: 直近7日平均の資金調達率がプラスの間だけ建てる(funding_carry.py)

cost_bps=30は現物レッグ+先物レッグ両方の手数料・スリッページを合わせた概算
(単純な単一レッグ戦略のfee_bps+slippage_bps=15の倍程度を想定)。

Usage:
    python backtests/run_funding_carry.py
data/ に事前に `python src/data.py funding <symbol>:USDT 3000` でキャッシュした
資金調達率が必要。
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "strategies"))

from backtest import run_cashflow_backtest, train_test_split_by_time  # noqa: E402
import funding_carry  # noqa: E402

FUNDING_FILES = {
    "BTC/USDT": ROOT / "data" / "binance_BTC-USDT-USDT_funding.csv",
    "ETH/USDT": ROOT / "data" / "binance_ETH-USDT-USDT_funding.csv",
}

COST_BPS = 30.0  # 現物+先物 両レッグ分の概算コスト


def evaluate(label: str, cashflow: pd.Series, signal: pd.Series) -> None:
    result = run_cashflow_backtest(cashflow, signal, cost_bps=COST_BPS)
    print(
        f"  {label:30s} | return={result.total_return:+8.2%} | sharpe={result.sharpe:+6.2f} "
        f"| max_dd={result.max_drawdown:7.2%} | trades={result.n_trades:4d}"
    )


def main() -> None:
    for symbol, path in FUNDING_FILES.items():
        if not path.exists():
            print(f"skip {symbol}: {path} not found (run src/data.py first)")
            continue

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        train, test = train_test_split_by_time(df, test_fraction=0.3)
        print(f"\n### {symbol} funding: {df.index[0]} -> {df.index[-1]} ({len(df)} intervals, 8h) ###")

        for split_name, split_df in [("train", train), ("test ", test)]:
            funding = split_df["funding_rate"]
            avg_annualized = funding.mean() * 3 * 365
            print(
                f"  -- {split_name} {split_df.index[0]} -> {split_df.index[-1]} "
                f"({len(split_df)} intervals, avg funding annualized={avg_annualized:+.2%})"
            )

            always_on = pd.Series(1.0, index=funding.index)
            evaluate("always_on_carry(no filter)", funding, always_on)

            conditional = funding_carry.generate_signal(funding, window=21, threshold=0.0)
            evaluate("conditional_carry(7d avg>0)", funding, conditional)


if __name__ == "__main__":
    main()
