"""公開OHLCVデータの取得ユーティリティ(APIキー不要、取引所の公開エンドポイントのみ使用)。"""

import json
import urllib.request
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "binance",
    total_bars: int = 5000,
    market_type: str = "spot",
) -> pd.DataFrame:
    """取引所の1回あたりの上限(通常1000本)を超えて、ページングしながら履歴を取得する。

    market_type="future"で無期限先物の実売買価格(mark priceではなく約定ベース)OHLCVを
    取得できる(symbolは"BTC/USDT:USDT"のような無期限先物表記にする)。
    """
    options = {"defaultType": "future"} if market_type == "future" else {}
    exchange = getattr(ccxt, exchange_id)({"options": options})
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - total_bars * timeframe_ms

    all_rows: list[list] = []
    while len(all_rows) < total_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + timeframe_ms
        if len(batch) < 1000:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp").tail(total_bars)


def fetch_and_cache(
    symbol: str,
    timeframe: str,
    exchange_id: str = "binance",
    total_bars: int = 5000,
    market_type: str = "spot",
) -> Path:
    df = fetch_ohlcv(symbol, timeframe, exchange_id, total_bars=total_bars, market_type=market_type)
    DATA_DIR.mkdir(exist_ok=True)
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    suffix = "_perp" if market_type == "future" else ""
    out_path = DATA_DIR / f"{exchange_id}_{safe_symbol}{suffix}_{timeframe}.csv"
    df.to_csv(out_path)
    return out_path


def fetch_funding_rate(
    symbol: str = "BTC/USDT:USDT",
    exchange_id: str = "binance",
    total_records: int = 3000,
) -> pd.DataFrame:
    """無期限先物の資金調達率(funding rate)履歴を取得する(8時間ごとが一般的)。

    ロング過多なら資金調達率がプラスに振れてロングがショートに支払う(逆張り指標として
    使われることが多い: 極端にプラス=ロング過熱=下落しやすい、という仮説)。
    """
    exchange = getattr(ccxt, exchange_id)({"options": {"defaultType": "future"}})
    interval_ms = 8 * 3600 * 1000
    since = exchange.milliseconds() - total_records * interval_ms

    all_rows: list[dict] = []
    while len(all_rows) < total_records:
        batch = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1]["timestamp"] + 1
        if len(batch) < 1000:
            break

    rows = []
    for r in all_rows:
        mark_price = r.get("info", {}).get("markPrice")
        rows.append(
            {
                "timestamp": r["timestamp"],
                "funding_rate": r["fundingRate"],
                "mark_price": float(mark_price) if mark_price else None,
            }
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp").tail(total_records)


def fetch_and_cache_funding_rate(
    symbol: str = "BTC/USDT:USDT",
    exchange_id: str = "binance",
    total_records: int = 3000,
) -> Path:
    df = fetch_funding_rate(symbol, exchange_id, total_records)
    DATA_DIR.mkdir(exist_ok=True)
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    out_path = DATA_DIR / f"{exchange_id}_{safe_symbol}_funding.csv"
    df.to_csv(out_path)
    return out_path


def fetch_bitflyer_funding_rate_history(
    product_code: str = "FX_BTC_JPY",
    total_records: int = 5000,
) -> pd.DataFrame:
    """bitFlyer Crypto CFDのfunding rate履歴を取得する(8時間ごと、認証不要のPublic API)。

    符号の意味はBinanceの無期限先物と同じ: プラス=CFD価格が現物より高く、買い建玉の
    保有者から売り建玉の保有者へ支払われる。このプロジェクトのキャリー戦略(現物ロング+
    CFDショート)は売り建玉側なので、プラスのfunding rateを受け取る側になる。

    ccxtはこのエンドポイントを公式サポートしていないため、標準ライブラリのurllibで
    直接叩く(`GET /v1/getfundingratehistory`、`to`パラメータを直近の取得済み最古の
    calculation_dateに設定してページングする)。
    """
    url = "https://api.bitflyer.com/v1/getfundingratehistory"
    all_rows: list[dict] = []
    seen: set[str] = set()
    cursor: str | None = None
    while len(all_rows) < total_records:
        query = f"?product_code={product_code}&count=500"
        if cursor:
            query += f"&to={cursor}"
        with urllib.request.urlopen(url + query) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        new = [r for r in batch if r["calculation_date"] not in seen]
        if not new:
            break
        seen.update(r["calculation_date"] for r in new)
        all_rows.extend(new)
        cursor = batch[-1]["calculation_date"]

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["calculation_date"])
    df["funding_rate"] = df["rate"].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df.set_index("timestamp")[["funding_rate"]].tail(total_records)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "funding":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT:USDT"
        total_records = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
        path = fetch_and_cache_funding_rate(symbol, total_records=total_records)
        print(f"saved: {path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "perp":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT:USDT"
        timeframe = sys.argv[3] if len(sys.argv) > 3 else "8h"
        total_bars = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
        path = fetch_and_cache(symbol, timeframe, total_bars=total_bars, market_type="future")
        print(f"saved: {path}")
    else:
        symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
        timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
        total_bars = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        path = fetch_and_cache(symbol, timeframe, total_bars=total_bars)
        print(f"saved: {path}")



