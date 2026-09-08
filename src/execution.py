"""ファンディング・キャリー戦略のペーパートレード/テストネット発注ロジック。

スコープ(README参照): ここで作るのはテストネット/ペーパートレード止まりの自動発注
コードまで。実際の取引所APIキー発行・保管、実資金での発注実行はユーザー自身の判断・
操作で行う。このモジュールはBinanceのテストネット環境専用で、本番(実資金)環境への
接続は意図的にサポートしない。

未検証: このモジュールはAPIキーを持たない環境で書かれており、実際のテストネット
発注は一度も実行・確認できていない。利用前に必ずdry_run=Trueで挙動を確認し、
少額のテストネット資金で試すこと。

必要な環境変数(.envに設定、Gitには含めない):
- BINANCE_TESTNET_SPOT_API_KEY / BINANCE_TESTNET_SPOT_API_SECRET
  (https://testnet.binance.vision で発行)
- BINANCE_TESTNET_FUTURES_API_KEY / BINANCE_TESTNET_FUTURES_API_SECRET
  (https://testnet.binancefuture.com で発行、spotとは別アカウント・別キー)
"""

import os
from dataclasses import dataclass

import ccxt
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CarryDecision:
    action: str  # "enter" | "hold" | "exit" | "stay_flat"
    reason: str


def decide_carry_action(
    trailing_funding_avg: float,
    currently_positioned: bool,
    entry_threshold: float = 0.0,
) -> CarryDecision:
    """backtests/run_funding_carry.pyの検証では「常に建てっぱなし」が最良だったが、
    実運用でそれを無条件にやるのはリスクが高いため、直近の資金調達率トレンドが
    entry_thresholdを下回ったら手仕舞う、という軽いセーフガードだけ加える。
    (バックテストで判明した通り、頻繁な出し入れはコスト負けするので、この判定は
    1日1回程度の頻度で呼ぶ想定 — 毎時呼ぶような使い方はしないこと)
    """
    if trailing_funding_avg >= entry_threshold:
        if currently_positioned:
            return CarryDecision("hold", f"funding avg {trailing_funding_avg:.4%} still >= threshold, staying in")
        return CarryDecision("enter", f"funding avg {trailing_funding_avg:.4%} >= threshold, entering carry")
    if currently_positioned:
        return CarryDecision("exit", f"funding avg {trailing_funding_avg:.4%} < threshold, exiting carry")
    return CarryDecision("stay_flat", f"funding avg {trailing_funding_avg:.4%} < threshold, staying flat")


def _get_testnet_exchange(exchange_id: str, api_key_env: str, api_secret_env: str) -> ccxt.Exchange:
    api_key = os.environ.get(api_key_env)
    api_secret = os.environ.get(api_secret_env)
    if not api_key or not api_secret:
        raise RuntimeError(
            f"{api_key_env} / {api_secret_env} が.envに設定されていません。"
            "テストネットのAPIキーは利用者自身が発行してください(実資金のキーは絶対に使わないこと)。"
        )
    exchange = getattr(ccxt, exchange_id)({"apiKey": api_key, "secret": api_secret})
    exchange.set_sandbox_mode(True)
    return exchange


def get_testnet_spot_exchange() -> ccxt.Exchange:
    return _get_testnet_exchange("binance", "BINANCE_TESTNET_SPOT_API_KEY", "BINANCE_TESTNET_SPOT_API_SECRET")


def get_testnet_futures_exchange() -> ccxt.Exchange:
    exchange = _get_testnet_exchange(
        "binance", "BINANCE_TESTNET_FUTURES_API_KEY", "BINANCE_TESTNET_FUTURES_API_SECRET"
    )
    exchange.options["defaultType"] = "future"
    return exchange


def place_carry_orders(symbol: str, notional_usd: float, dry_run: bool = True) -> None:
    """現物ロング + 無期限先物ショートのペアオーダーを建てる(テストネット限定)。

    dry_run=True(デフォルト)なら実際には何も送信せず、意図する注文をログ出力するのみ。
    """
    if dry_run:
        print(f"[DRY RUN] would BUY spot {symbol} notional=${notional_usd:.2f}")
        print(f"[DRY RUN] would SELL(short) perp {symbol} notional=${notional_usd:.2f}")
        return

    spot = get_testnet_spot_exchange()
    futures = get_testnet_futures_exchange()

    spot_price = spot.fetch_ticker(symbol)["last"]
    spot_amount = notional_usd / spot_price
    spot.create_market_buy_order(symbol, spot_amount)

    perp_symbol = f"{symbol}:USDT"
    perp_price = futures.fetch_ticker(perp_symbol)["last"]
    perp_amount = notional_usd / perp_price
    futures.create_order(perp_symbol, "market", "sell", perp_amount)


def close_carry_orders(symbol: str, notional_usd: float, dry_run: bool = True) -> None:
    """建てたキャリーポジションを手仕舞う(現物売却 + 先物ショートの買い戻し)。"""
    if dry_run:
        print(f"[DRY RUN] would SELL spot {symbol} notional=${notional_usd:.2f}")
        print(f"[DRY RUN] would BUY(cover) perp {symbol} notional=${notional_usd:.2f}")
        return

    spot = get_testnet_spot_exchange()
    futures = get_testnet_futures_exchange()

    spot_price = spot.fetch_ticker(symbol)["last"]
    spot_amount = notional_usd / spot_price
    spot.create_market_sell_order(symbol, spot_amount)

    perp_symbol = f"{symbol}:USDT"
    perp_price = futures.fetch_ticker(perp_symbol)["last"]
    perp_amount = notional_usd / perp_price
    futures.create_order(perp_symbol, "market", "buy", perp_amount, params={"reduceOnly": True})
