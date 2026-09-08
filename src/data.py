"""公開OHLCVデータの取得ユーティリティ(APIキー不要、取引所の公開エンドポイントのみ使用)。"""

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "binance",
    total_bars: int = 5000,
) -> pd.DataFrame:
    """取引所の1回あたりの上限(通常1000本)を超えて、ページングしながら履歴を取得する。"""
    exchange = getattr(ccxt, exchange_id)()
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
) -> Path:
    df = fetch_ohlcv(symbol, timeframe, exchange_id, total_bars=total_bars)
    DATA_DIR.mkdir(exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    out_path = DATA_DIR / f"{exchange_id}_{safe_symbol}_{timeframe}.csv"
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

    df = pd.DataFrame([{"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]} for r in all_rows])
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "funding":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT:USDT"
        total_records = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
        path = fetch_and_cache_funding_rate(symbol, total_records=total_records)
        print(f"saved: {path}")
    else:
        symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
        timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
        total_bars = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        path = fetch_and_cache(symbol, timeframe, total_bars=total_bars)
        print(f"saved: {path}")



