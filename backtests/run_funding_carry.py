"""ファンディング・キャリー(現物ロング+無期限先物ショートのデルタニュートラル)を検証する。

2つのバックテストを並べて比較する:
- naive: run_cashflow_backtest — 資金調達率をそのままリターンとみなす簡略版
         (現物と先物の価格は完全に連動する = ベーシスリスクゼロという前提)
- basis-aware: run_funding_carry_backtest — 現物と無期限先物の実際の価格差変動
         (ベーシスリスク)を反映し、投入資本($1現物+$1証拠金=$2)に対するリターン
         として計算する、より現実に近い版

注意: 先物側の価格には無期限先物の「実際の約定価格」(fetch_ohlcv market_type="future")
を使う。当初はfunding rate APIに付随する"mark price"を使っていたが、これは複数取引所の
指数を織り込んだ清算判定用の参照価格であり、実際に約定できる価格と大きく乖離すること
がある(検証の結果、mark price基準だとBTCのベーシス標準偏差が1.58%/8hにもなり、これを
そのままバックテストに使うと有効なアービトラージとは無関係な「見せかけのベーシスリスク」
による巨大なボラティリティ・ドラッグが発生し、結果が大きく歪んだ。実約定価格ベースだと
標準偏差0.04%/8hまで下がり、現実の裁定コストに近い水準になる)。

BTC/ETH/SOL/BNBの4銘柄で頑健性を確認し、証拠金/清算リスクの目安として先物価格の
急変動幅も出力する。

Usage:
    python backtests/run_funding_carry.py
data/ に事前に
`python src/data.py funding <symbol>:USDT 6000`、
`python src/data.py <symbol> 8h 6000`、
`python src/data.py perp <symbol>:USDT 8h 6000`
でキャッシュしておく必要がある。
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "strategies"))

from backtest import (  # noqa: E402
    run_cashflow_backtest,
    run_funding_carry_backtest,
    train_test_split_by_time,
)
import funding_carry  # noqa: E402

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
COST_BPS = 30.0  # 現物+先物 両レッグ分の概算コスト


def load(symbol: str) -> pd.DataFrame | None:
    safe = symbol.replace("/", "-")
    funding_path = ROOT / "data" / f"binance_{safe}-USDT_funding.csv"
    spot_path = ROOT / "data" / f"binance_{safe}_8h.csv"
    perp_path = ROOT / "data" / f"binance_{safe}-USDT_perp_8h.csv"
    if not funding_path.exists() or not spot_path.exists() or not perp_path.exists():
        return None

    funding = pd.read_csv(funding_path, index_col="timestamp", parse_dates=True)
    spot = pd.read_csv(spot_path, index_col="timestamp", parse_dates=True)
    perp = pd.read_csv(perp_path, index_col="timestamp", parse_dates=True)

    df = pd.DataFrame(index=spot.index)
    df["spot_close"] = spot["close"]
    df["perp_close"] = perp["close"].reindex(df.index, method="ffill")
    df["funding_rate"] = funding["funding_rate"].reindex(df.index, method="ffill")
    return df.dropna(subset=["funding_rate", "perp_close", "spot_close"])


def evaluate(label: str, result) -> None:
    print(
        f"  {label:34s} | return={result.total_return:+8.2%} | sharpe={result.sharpe:+6.2f} "
        f"| max_dd={result.max_drawdown:7.2%} | trades={result.n_trades:4d}"
    )


def main() -> None:
    for symbol in SYMBOLS:
        df = load(symbol)
        if df is None:
            print(f"skip {symbol}: data not found (run src/data.py first)")
            continue

        # 証拠金/清算リスクの目安: 先物価格の急変動幅(全期間)
        interval_move = df["perp_close"].pct_change().abs()
        rolling_7d_move = df["perp_close"].pct_change(21)  # 21本x8h=7日
        print(f"\n### {symbol} ###")
        print(
            f"  [margin risk] max single-interval(8h) move={interval_move.max():.2%} | "
            f"max 7d rolling rally={rolling_7d_move.max():+.2%} | max 7d rolling crash={rolling_7d_move.min():+.2%}"
        )

        train, test = train_test_split_by_time(df, test_fraction=0.3)
        for split_name, split_df in [("train", train), ("test ", test)]:
            avg_annualized = split_df["funding_rate"].mean() * 3 * 365
            print(
                f"  -- {split_name} {split_df.index[0]} -> {split_df.index[-1]} "
                f"({len(split_df)} intervals, avg funding annualized={avg_annualized:+.2%})"
            )

            always_on = pd.Series(1.0, index=split_df.index)
            conditional = funding_carry.generate_signal(split_df["funding_rate"], window=21, threshold=0.0)

            for sig_name, signal in [("always_on", always_on), ("conditional(7d avg>0)", conditional)]:
                naive = run_cashflow_backtest(split_df["funding_rate"], signal, cost_bps=COST_BPS)
                evaluate(f"{sig_name} naive(funding only)", naive)

                basis_aware = run_funding_carry_backtest(
                    split_df["spot_close"], split_df["perp_close"], split_df["funding_rate"], signal, cost_bps=COST_BPS
                )
                evaluate(f"{sig_name} basis-aware(spot+perp)", basis_aware)


if __name__ == "__main__":
    main()
