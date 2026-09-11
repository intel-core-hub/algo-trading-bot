"""bitFlyer Crypto CFD(FX_BTC_JPY)のfunding rateだけで、キャリー戦略の経済性を
一次判定する(価格データ不要の簡易版)。

背景(PLAN_algo-trading-bot.md参照): Binance Futures Testnetは日本居住者アカウントが
構造的に先物メニューへ到達できないため、代替の実行先候補としてbitFlyer Crypto CFDを
調査した。`GET /v1/getfundingratehistory`はPublic API(認証不要)でfunding rate履歴を
取得できるため、実口座・APIキーが無くてもこの経済性の一次検証だけは今すぐ行える。

このスクリプトが検証するのはfunding rateの水準・トレンドのみで、現物とCFDの価格差
(ベーシスリスク)は含まない(bitFlyer Crypto CFDのOHLC/約定履歴を取得できる公開
エンドポイントが見当たらなかったため)。Binance側の検証(backtests/run_funding_carry.py)
と違い、ベーシスリスクを考慮しない楽観側の試算であることに注意。

もう一つ重要な非対称性: bitFlyer Crypto CFDには funding rate とは別建てで
「レバレッジポイント」という日次コストがあり、買建玉・売建玉を問わずポジション評価額の
0.04%/日(年率換算約14.6%)がかかる(公式手数料ページで確認、2026-09-11時点)。
funding rateは買い/売り間のゼロサムの授受だが、レバレッジポイントは方向によらず
bitFlyerへ支払う純粋なコストなので、キャリー戦略のCFDショート脚には両方が重なる。

Usage:
    python backtests/check_bitflyer_funding_viability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data import fetch_bitflyer_funding_rate_history  # noqa: E402

INTERVALS_PER_YEAR = 365.25 * 24 / 8  # 8時間ごと
LEVERAGE_POINT_DAILY = 0.0004  # 建玉金額の0.04%/日(方向によらず、公式手数料ページより)
LEVERAGE_POINT_ANNUAL = LEVERAGE_POINT_DAILY * 365.25
TRAILING_WINDOW = 21  # 21本 x 8h = 7日(live/check_and_trade_carry.pyと合わせる)


def _annualized(mean_per_interval: float) -> float:
    return mean_per_interval * INTERVALS_PER_YEAR


def _sharpe_like(rates: pd.Series) -> float:
    mean, std = rates.mean(), rates.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return (mean / std) * np.sqrt(INTERVALS_PER_YEAR)


def _report_period(name: str, rates: pd.Series) -> None:
    gross_annual = _annualized(rates.mean())
    net_annual = gross_annual - LEVERAGE_POINT_ANNUAL
    print(
        f"{name:12s} n={len(rates):5d}  gross_annualized={gross_annual:8.2%}  "
        f"net_of_leverage_point={net_annual:8.2%}  sharpe_like={_sharpe_like(rates):6.2f}  "
        f"pct_positive={(rates > 0).mean():.1%}"
    )


def main() -> None:
    df = fetch_bitflyer_funding_rate_history(total_records=10000)
    print(f"fetched {len(df)} records: {df.index.min()} -> {df.index.max()}")
    print(f"leverage-point cost (short CFD leg, direction-independent): {LEVERAGE_POINT_ANNUAL:.2%}/year\n")

    print("--- full period ---")
    _report_period("full", df["funding_rate"])
    print()

    split_idx = int(len(df) * 0.7)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    print("--- train/test (chronological 70/30, same convention as the Binance analysis) ---")
    _report_period("train", train["funding_rate"])
    _report_period("test", test["funding_rate"])
    print()

    print("--- by quarter ---")
    for period, g in df.groupby(df.index.to_period("Q")):
        _report_period(str(period), g["funding_rate"])
    print()

    trailing = df["funding_rate"].rolling(TRAILING_WINDOW).mean().iloc[-1]
    trailing_annual = _annualized(trailing)
    print("--- current live signal (trailing 7d avg, same window as decide_carry_action) ---")
    print(f"trailing_annualized={trailing_annual:.2%}  net_of_leverage_point={trailing_annual - LEVERAGE_POINT_ANNUAL:.2%}")


if __name__ == "__main__":
    main()
