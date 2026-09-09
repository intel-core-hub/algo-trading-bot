"""ファンディング・キャリー戦略のペーパートレード/テストネット発注ロジック。

スコープ(README参照): ここで作るのはテストネット/ペーパートレード止まりの自動発注
コードまで。実際の取引所APIキー発行・保管、実資金での発注実行はユーザー自身の判断・
操作で行う。このモジュールはBinanceのテストネット環境専用で、本番(実資金)環境への
接続は意図的にサポートしない。

未検証: このモジュールはAPIキーを持たない環境で書かれており、実際のテストネット
発注は一度も実行・確認できていない。利用前に必ずdry_run=Trueで挙動を確認し、
少額のテストネット資金で試すこと。

2レッグ発注は原子的ではない(取引所APIの制約上、現物と先物を単一トランザクションで
同時に約定させることはできない)。そのため以下の順序と復旧方針を採用する:
- エントリー: 現物ロングを先に建て、成功したら先物ショートを建てる。先物が失敗したら
  直ちに現物を反対売買して巻き戻す(失敗時に残るのが「現物ロングのみ」という、
  レバレッジも清算リスクも無い一番安全な状態になるようにする)。
- イグジット: 先物の買い戻しを先に行い、成功したら現物を売却する。現物売却が失敗しても
  残るのは「現物ロングのみ」で、エントリー失敗時と同じ安全な状態になる。
- 巻き戻し自体が失敗した場合(例: 現物ロング成立後に先物注文もその巻き戻しも失敗)は
  needs_manual_intervention=Trueを立てて例外は投げず、呼び出し側が状態ファイルに
  記録し人手での確認を促せるようにする。

必要な環境変数(.envに設定、Gitには含めない):
- BINANCE_TESTNET_SPOT_API_KEY / BINANCE_TESTNET_SPOT_API_SECRET
  (https://testnet.binance.vision で発行)
- BINANCE_TESTNET_FUTURES_API_KEY / BINANCE_TESTNET_FUTURES_API_SECRET
  (https://testnet.binancefuture.com で発行、spotとは別アカウント・別キー)
"""

import os
from dataclasses import dataclass, field

import ccxt
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CarryDecision:
    action: str  # "enter" | "hold" | "exit" | "stay_flat"
    reason: str


@dataclass
class CarryExecutionResult:
    dry_run: bool
    spot_filled: bool = False
    perp_filled: bool = False
    unwound: bool = False
    needs_manual_intervention: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def fully_positioned(self) -> bool:
        """両レッグとも約定した(=キャリーポジションが意図通り建った/閉じた)かどうか。"""
        return self.spot_filled and self.perp_filled


def decide_carry_action(
    trailing_funding_avg: float,
    currently_positioned: bool,
    entry_threshold: float = 0.0,
) -> CarryDecision:
    """backtests/run_funding_carry.pyの検証結果に基づく判断。

    重要: 検証でOOSでもプラスだったのは「常に建てっぱなし」のalways_on_carryであり、
    直近7日平均が閾値を割ったら毎回手仕舞う短周期のconditional_carryはOOSでコスト負け
    していた(BTC test -3.79%、ETH test -4.94%)。そのためここでのセーフガードは
    意図的に緩くしてある: 通常はentry_threshold=0.0のまま一度建てたら持ち続け、
    「明確なレジーム転換」と呼べるレベルまで資金調達が悪化した場合のみ手仕舞う
    判断にすること(呼び出し側でtrailing_funding_avgに渡す集計期間を短くしすぎない
    こと — 例えば7日平均を毎日評価するような使い方は、バックテストで負けが
    確認されている運用と同じになるので避ける)。
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


def fetch_actual_position(symbol: str) -> dict:
    """状態ファイル(JSON)を信用する前に、実際の取引所残高/ポジションを取得して
    照合するための関数。spot残高(baseアセット)とfutures建玉数量を返す。
    テストネットAPIキーが必要。呼び出し側でこの結果と状態ファイルを突き合わせ、
    食い違いがあれば自動売買を止めて人手の確認を促すこと。
    """
    base_asset = symbol.split("/")[0]
    spot = get_testnet_spot_exchange()
    futures = get_testnet_futures_exchange()

    spot_balance = spot.fetch_balance()
    spot_amount = float(spot_balance.get("total", {}).get(base_asset, 0.0))

    perp_symbol = f"{symbol}:USDT"
    positions = futures.fetch_positions([perp_symbol])
    perp_amount = 0.0
    for p in positions:
        if p.get("symbol") == perp_symbol:
            perp_amount = float(p.get("contracts") or 0.0) * (-1 if p.get("side") == "short" else 1)

    return {"spot_amount": spot_amount, "perp_amount": perp_amount}


def reconcile_state(symbol: str, state_positioned: bool, tolerance: float = 1e-6) -> tuple[bool, str]:
    """状態ファイルの`positioned`と実際の取引所残高/建玉を突き合わせる。
    戻り値: (matches, message)。matches=Falseなら自動売買を進めず人手で確認すること。
    """
    actual = fetch_actual_position(symbol)
    has_spot = abs(actual["spot_amount"]) > tolerance
    has_short_perp = actual["perp_amount"] < -tolerance
    actually_positioned = has_spot and has_short_perp

    if actually_positioned == state_positioned:
        return True, f"state matches exchange (positioned={state_positioned}, {actual})"
    return False, (
        f"MISMATCH: state says positioned={state_positioned} but exchange shows "
        f"spot={actual['spot_amount']}, perp={actual['perp_amount']} "
        f"(actually_positioned={actually_positioned}). Manual review required before trading."
    )


def place_carry_orders(symbol: str, notional_usd: float, dry_run: bool = True) -> CarryExecutionResult:
    """現物ロング + 無期限先物ショートのペアオーダーを建てる(テストネット限定)。

    dry_run=True(デフォルト)なら実際には何も送信せず、意図する注文をログ出力するのみ。
    """
    if dry_run:
        print(f"[DRY RUN] would BUY spot {symbol} notional=${notional_usd:.2f}")
        print(f"[DRY RUN] would SELL(short) perp {symbol} notional=${notional_usd:.2f}")
        return CarryExecutionResult(dry_run=True, spot_filled=True, perp_filled=True)

    result = CarryExecutionResult(dry_run=False)
    spot = get_testnet_spot_exchange()

    spot_price = spot.fetch_ticker(symbol)["last"]
    spot_amount = notional_usd / spot_price
    try:
        spot.create_market_buy_order(symbol, spot_amount)
        result.spot_filled = True
    except Exception as exc:  # noqa: BLE001 - 取引所APIの例外型は多岐にわたるため
        result.errors.append(f"spot buy failed: {exc}")
        return result  # 現物すら建っていないので巻き戻し不要

    try:
        futures = get_testnet_futures_exchange()
        perp_symbol = f"{symbol}:USDT"
        perp_price = futures.fetch_ticker(perp_symbol)["last"]
        perp_amount = notional_usd / perp_price
        futures.create_order(perp_symbol, "market", "sell", perp_amount)
        result.perp_filled = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"perp short failed: {exc}")
        try:
            spot.create_market_sell_order(symbol, spot_amount)
            result.unwound = True
            result.spot_filled = False
        except Exception as unwind_exc:  # noqa: BLE001
            result.errors.append(f"unwind of spot leg also failed: {unwind_exc}")
            result.needs_manual_intervention = True  # 現物ロングだけが裸で残っている

    return result


def close_carry_orders(symbol: str, notional_usd: float, dry_run: bool = True) -> CarryExecutionResult:
    """建てたキャリーポジションを手仕舞う。先物の買い戻しを先に行い、成功してから
    現物を売却する(現物売却が失敗しても残るのは低リスクな現物ロングのみにするため)。
    """
    if dry_run:
        print(f"[DRY RUN] would BUY(cover) perp {symbol} notional=${notional_usd:.2f}")
        print(f"[DRY RUN] would SELL spot {symbol} notional=${notional_usd:.2f}")
        return CarryExecutionResult(dry_run=True, spot_filled=True, perp_filled=True)

    result = CarryExecutionResult(dry_run=False)
    futures = get_testnet_futures_exchange()
    perp_symbol = f"{symbol}:USDT"

    try:
        perp_price = futures.fetch_ticker(perp_symbol)["last"]
        perp_amount = notional_usd / perp_price
        futures.create_order(perp_symbol, "market", "buy", perp_amount, params={"reduceOnly": True})
        result.perp_filled = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"perp cover failed: {exc}")
        result.needs_manual_intervention = True  # 先物ショートがまだ残っている(レバレッジ・清算リスクあり)
        return result

    try:
        spot = get_testnet_spot_exchange()
        spot_price = spot.fetch_ticker(symbol)["last"]
        spot_amount = notional_usd / spot_price
        spot.create_market_sell_order(symbol, spot_amount)
        result.spot_filled = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"spot sell failed: {exc}")
        # 先物は既に手仕舞い済みで残るのは現物ロングのみ(低リスク)。次回実行時に再試行される。

    return result
