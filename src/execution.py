"""ファンディング・キャリー戦略のペーパートレード/テストネット発注ロジック。

スコープ(README参照): ここで作るのはテストネット/ペーパートレード止まりの自動発注
コードまで。実際の取引所APIキー発行・保管、実資金での発注実行はユーザー自身の判断・
操作で行う。このモジュールはBinanceのテストネット環境専用で、本番(実資金)環境への
接続は意図的にサポートしない。

未検証: このモジュールはAPIキーを持たない環境で書かれており、実際のテストネット
発注は一度も実行・確認できていない。利用前に必ずdry_run=Trueで挙動を確認し、
少額のテストネット資金で試すこと。

2レッグ発注は原子的ではない(取引所APIの制約上、現物と先物を単一トランザクションで
同時に約定させることはできない)ため、以下の方針で「あいまいな失敗」に対処する。

- エントリー前に必ず先物レッグのレバレッジを1倍に明示設定する(口座のデフォルト
  レバレッジが高倍率になっていると、意図しないレバレッジ付きポジションになり
  清算リスクが跳ね上がるため)。設定自体が失敗したら、まだ何の注文も送っていない
  ので何もせず中断する。
- エントリー: 現物ロングを先に建て、実際に約定した数量(注文時点の見積もりではなく
  fetch_orderで確認した約定数量)を使って先物ショートのサイズを決める(見積もり価格と
  実際の約定価格・数量がズレるとヘッジ比率が崩れるため)。先物注文がAPI例外で失敗した
  場合、実際には約定していた可能性がある(ネットワーク断でレスポンスだけ届かない等)ので、
  現物を盲目的に売り戻す前に取引所の実ポジションを確認する。確認の結果ショートが
  存在していれば、現物を売ってしまうと「ヘッジの無い裸のショート」という一番危険な
  状態を自ら作ってしまうため、自動での巻き戻しはせずneeds_manual_intervention=Trueで
  止める。ショートが存在しなければ安全に現物を売り戻す。
- イグジット: 先物の買い戻しを先に行い、成功したら現物を売却する(現物売却が失敗しても
  残るのは低リスクな現物ロングのみ)。手仕舞い数量は「その時点の価格から再計算した
  想定数量」でも「呼び出し側が記録している数量」でもなく、発注の直前にその場で
  取引所から取り直した実際の保有数量を使う(状態ファイルの記録と実際の残高の間には
  タイミング次第でズレが生じうるため、最終的な発注サイズは常に取引所を正とする)。
  さらに実ポジションが「現物とショートが釣り合ったcarry」に分類できない場合
  (現物のみ/ショートのみ/大きく不均衡)は発注せず停止する。

必要な環境変数(.envに設定、Gitには含めない):
- BINANCE_TESTNET_SPOT_API_KEY / BINANCE_TESTNET_SPOT_API_SECRET
  (https://testnet.binance.vision で発行)
- BINANCE_TESTNET_FUTURES_API_KEY / BINANCE_TESTNET_FUTURES_API_SECRET
  (https://testnet.binancefuture.com で発行、spotとは別アカウント・別キー)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

import ccxt
from dotenv import load_dotenv

load_dotenv()


class CarryAccountState(str, Enum):
    """取引所側で実際に観測されるキャリーポジションの状態。"""

    FLAT = "flat"
    SPOT_ONLY = "spot_only"
    PERP_ONLY = "perp_only"  # 現物なしでショートだけが残っている、一番危険な状態
    CARRY_BALANCED = "carry_balanced"  # 現物とショートの数量が釣り合っている
    CARRY_IMBALANCED = "carry_imbalanced"  # 両方あるが数量が釣り合っていない


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
    spot_amount: float = 0.0  # 実際に約定したbase資産数量
    perp_amount: float = 0.0  # 実際に約定したbase資産数量(絶対値)
    spot_order_id: str | None = None
    perp_order_id: str | None = None

    @property
    def fully_positioned(self) -> bool:
        """両レッグとも意図通り約定した(=キャリーポジションが正しく建った/閉じた)か。

        needs_manual_intervention=Trueの場合は、たとえ結果的に数量が一致していても
        「あいまいな失敗を経由した」という事実自体が要確認事項なので、必ずFalseを返す
        (呼び出し側がif result.fully_positioned: ... elif needs_manual_intervention: ...
        という順で分岐しても、要確認フラグを見落とさないようにするため)。
        """
        if self.needs_manual_intervention:
            return False
        if not (self.spot_filled and self.perp_filled):
            return False
        if self.dry_run:
            return True
        if self.spot_amount <= 0 or self.perp_amount <= 0:
            return False
        return _amounts_close(self.spot_amount, self.perp_amount)


def _amounts_close(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-8) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def decide_carry_action(
    trailing_funding_avg: float,
    currently_positioned: bool,
    entry_threshold: float = 0.0,
) -> CarryDecision:
    """backtests/run_funding_carry.pyの検証結果に基づく判断。

    重要: 検証でOOSでもプラスだったのは「常に建てっぱなし」のalways_on_carryであり、
    直近7日平均を毎回評価して出し入れするconditional_carryはOOSでコスト負けしていた
    (BTC test -3.79%、ETH test -4.94%)。そのためこの関数自体は毎回シンプルな閾値判定
    しか行わないが、呼び出し側(live/check_and_trade_carry.py)でクールダウンを設け、
    頻繁な出し入れにならないようにしてある。
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


def _market_contract_size(exchange: ccxt.Exchange, symbol: str) -> float:
    """デリバティブの1コントラックスあたりbase資産数量を返す(現物や擬似取引所では1.0)。

    Binanceの無期限先物は銘柄によって1コントラクト=1 base単位でない場合がある
    (contractSizeが1以外)。これを無視してbase数量をそのままコントラクト数として
    送ると発注サイズが実際の意図と大きくズレる。
    """
    try:
        exchange.load_markets()
        market = exchange.market(symbol)
    except (AttributeError, KeyError, TypeError):
        return 1.0
    contract_size = float(market.get("contractSize") or 1.0)
    if contract_size <= 0:
        raise RuntimeError(f"invalid contractSize for {symbol}: {contract_size}")
    return contract_size


def _base_to_contract_amount(exchange: ccxt.Exchange, symbol: str, base_amount: float) -> float:
    contracts = base_amount / _market_contract_size(exchange, symbol)
    if hasattr(exchange, "amount_to_precision"):
        contracts = float(exchange.amount_to_precision(symbol, contracts))
    if contracts <= 0:
        raise RuntimeError(f"order amount rounded to zero for {symbol} (base_amount={base_amount})")
    return contracts


def _order_snapshot(exchange: ccxt.Exchange, symbol: str, order: dict | None) -> dict:
    """create_orderの戻り値が不完全な場合(市場注文でよくある)、fetch_orderで
    確定した約定情報を取り直す。"""
    if not isinstance(order, dict):
        raise RuntimeError(f"exchange returned no order details for {symbol}; cannot verify fill")
    order_id = order.get("id")
    status = order.get("status")
    needs_refresh = order.get("filled") is None or status not in ("closed", "canceled")
    if order_id and needs_refresh and hasattr(exchange, "fetch_order"):
        try:
            return exchange.fetch_order(order_id, symbol)
        except Exception:  # noqa: BLE001 - 取得できなければ元のレスポンスにフォールバック
            pass
    return order


def _filled_amount(exchange: ccxt.Exchange, symbol: str, order: dict | None, *, contract: bool) -> tuple[float, str | None]:
    snapshot = _order_snapshot(exchange, symbol, order)
    filled = snapshot.get("filled")
    if filled is None and snapshot.get("status") == "closed":
        filled = snapshot.get("amount")
    filled = float(filled or 0.0)
    if filled <= 0:
        raise RuntimeError(f"{symbol} order has no verified fill: {snapshot}")
    if contract:
        filled *= _market_contract_size(exchange, symbol)
    order_id = snapshot.get("id") or (order.get("id") if isinstance(order, dict) else None)
    return filled, str(order_id) if order_id is not None else None


def _fetch_perp_signed_base_amount(futures: ccxt.Exchange, perp_symbol: str) -> float:
    """先物の符号付きbase数量(ロング=正、ショート=負)を返す。ポジション無しは0。"""
    positions = futures.fetch_positions([perp_symbol])
    contract_size = _market_contract_size(futures, perp_symbol)
    for p in positions:
        if p.get("symbol") != perp_symbol:
            continue
        contracts = float(p.get("contracts") or 0.0)
        sign = -1.0 if p.get("side") == "short" else 1.0
        return contracts * contract_size * sign
    return 0.0


def fetch_actual_position(symbol: str) -> dict[str, float]:
    """状態ファイル(JSON)を信用する前に、実際の取引所残高/ポジションを取得して
    照合するための関数。spot残高(baseアセット)とfutures建玉数量(符号付き)を返す。
    テストネットAPIキーが必要。

    このbot专用のサブアカウント/テストネット口座を使うことを強く推奨する。もし口座に
    無関係な現物在庫が既にあると、それをこのbotのポジションと誤認せず、
    reconcile_stateはあえてブロックする(意図的な安全側の挙動)。
    """
    base_asset = symbol.split("/")[0]
    spot = get_testnet_spot_exchange()
    futures = get_testnet_futures_exchange()

    spot_balance = spot.fetch_balance()
    spot_amount = float(spot_balance.get("total", {}).get(base_asset, 0.0))

    perp_symbol = f"{symbol}:USDT"
    perp_amount = _fetch_perp_signed_base_amount(futures, perp_symbol)
    return {"spot_amount": spot_amount, "perp_amount": perp_amount}


def classify_actual_position(actual: dict[str, float], tolerance: float = 1e-8) -> CarryAccountState:
    spot_amount = float(actual.get("spot_amount", 0.0))
    perp_amount = float(actual.get("perp_amount", 0.0))
    has_spot = spot_amount > tolerance
    has_perp = abs(perp_amount) > tolerance

    if not has_spot and not has_perp:
        return CarryAccountState.FLAT
    if has_spot and not has_perp:
        return CarryAccountState.SPOT_ONLY
    if not has_spot and has_perp:
        return CarryAccountState.PERP_ONLY
    if perp_amount < -tolerance and _amounts_close(spot_amount, abs(perp_amount)):
        return CarryAccountState.CARRY_BALANCED
    return CarryAccountState.CARRY_IMBALANCED


def reconcile_state(
    symbol: str,
    expected_status: str | None = None,
    *,
    state_positioned: bool | None = None,
    expected_spot_amount: float | None = None,
    expected_perp_amount: float | None = None,
    tolerance: float = 1e-8,
) -> tuple[bool, str]:
    """状態ファイルの内容と実際の取引所残高/建玉を突き合わせる(fail-closed)。

    expected_statusは"flat"/"carry"/"spot_only"のいずれか。state_positionedは
    後方互換のために残してある古い呼び出し方(bool)。expected_spot_amount /
    expected_perp_amount を渡すと、状態(flat/carry/spot_only)の一致だけでなく
    記録されている数量そのものまで突き合わせる。
    """
    if expected_status is None:
        if state_positioned is None:
            raise TypeError("expected_status or state_positioned is required")
        expected_status = "carry" if state_positioned else "flat"

    actual = fetch_actual_position(symbol)
    observed = classify_actual_position(actual, tolerance=tolerance)
    expected_map = {
        "flat": CarryAccountState.FLAT,
        "carry": CarryAccountState.CARRY_BALANCED,
        "spot_only": CarryAccountState.SPOT_ONLY,
    }
    if expected_status not in expected_map:
        return False, f"MISMATCH: unsupported saved status={expected_status!r}; manual review required"

    expected_observed = expected_map[expected_status]
    reasons: list[str] = []
    if observed != expected_observed:
        reasons.append(f"saved status={expected_status} expects {expected_observed.value}, observed {observed.value}")

    if expected_status in ("carry", "spot_only") and expected_spot_amount is not None:
        if not _amounts_close(actual["spot_amount"], expected_spot_amount):
            reasons.append(
                f"spot qty mismatch expected={expected_spot_amount:.12g}, actual={actual['spot_amount']:.12g}"
            )
    if expected_status == "carry" and expected_perp_amount is not None:
        if actual["perp_amount"] >= -tolerance or not _amounts_close(abs(actual["perp_amount"]), expected_perp_amount):
            reasons.append(
                f"perp qty mismatch expected short={expected_perp_amount:.12g}, actual={actual['perp_amount']:.12g}"
            )

    if reasons:
        return False, f"MISMATCH: {'; '.join(reasons)}. actual={actual}. Manual review required before trading."
    return True, f"state matches exchange (saved={expected_status}, observed={observed.value}, actual={actual})"


def _ensure_1x_leverage(futures: ccxt.Exchange, perp_symbol: str) -> None:
    """先物レッグを必ず1倍(無レバレッジ)で建てる。

    取引所口座のデフォルトレバレッジ設定(テストネットでも往々にして5〜20倍等が
    デフォルトになっている)に任せると、「現物+先物のデルタニュートラル・キャリー」
    のつもりが、意図せずレバレッジ付きの先物ポジションになり清算リスクが跳ね上がる。
    毎回のエントリー前に明示的に1倍へ設定する。

    取引所オブジェクトが`set_leverage`自体を持たない場合、以前は何もせず黙って
    スキップしていた(=レバレッジが未確認のまま発注に進んでしまう、安全機構が
    無言で無効化される経路)。これも他の失敗と同様に明示的なエラーとして扱い、
    呼び出し元(place_carry_orders)の既存のtry/exceptで「まだ何も注文していない
    ので中断する」という同じ安全なパスに合流させる。
    """
    if not hasattr(futures, "set_leverage"):
        raise RuntimeError(f"exchange does not support set_leverage; cannot guarantee 1x leverage for {perp_symbol}")
    futures.set_leverage(1, perp_symbol)


def place_carry_orders(symbol: str, notional_usd: float, dry_run: bool = True) -> CarryExecutionResult:
    """現物ロング + 無期限先物ショートを、実際の現物約定数量でヘッジして建てる。"""
    if notional_usd <= 0:
        raise ValueError("notional_usd must be positive")
    if dry_run:
        print(f"[DRY RUN] would BUY spot {symbol} notional=${notional_usd:.2f}")
        print(f"[DRY RUN] would SELL(short) perp {symbol} using the actual spot filled base quantity (1x leverage)")
        return CarryExecutionResult(dry_run=True, spot_filled=True, perp_filled=True)

    result = CarryExecutionResult(dry_run=False)
    futures = get_testnet_futures_exchange()
    perp_symbol = f"{symbol}:USDT"
    try:
        _ensure_1x_leverage(futures, perp_symbol)
    except Exception as exc:  # noqa: BLE001
        # まだ何の注文も送っていないので、失敗してもポジション状態には影響しない。
        result.errors.append(f"failed to set 1x leverage before entry, aborting: {exc}")
        return result

    spot = get_testnet_spot_exchange()
    spot_price = float(spot.fetch_ticker(symbol)["last"])
    requested_spot_amount = notional_usd / spot_price

    try:
        spot_order = spot.create_market_buy_order(symbol, requested_spot_amount)
        result.spot_amount, result.spot_order_id = _filled_amount(spot, symbol, spot_order, contract=False)
        result.spot_filled = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"spot buy failed or fill could not be verified: {exc}")
        return result  # 現物すら建っていないので巻き戻し不要

    try:
        perp_order_amount = _base_to_contract_amount(futures, perp_symbol, result.spot_amount)
        perp_order = futures.create_order(perp_symbol, "market", "sell", perp_order_amount)
        result.perp_amount, result.perp_order_id = _filled_amount(futures, perp_symbol, perp_order, contract=True)
        result.perp_filled = True
        if not _amounts_close(result.spot_amount, result.perp_amount):
            result.errors.append(
                f"hedge fill mismatch: spot={result.spot_amount:.12g}, perp={result.perp_amount:.12g}"
            )
            result.needs_manual_intervention = True
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"perp short failed or fill could not be verified: {exc}")
        # API例外は「実は約定していた」可能性を排除できないので、盲目的に現物を
        # 売り戻す前に取引所の実ポジションを確認する。
        try:
            signed_perp = _fetch_perp_signed_base_amount(futures, perp_symbol)
        except Exception as verify_exc:  # noqa: BLE001
            result.errors.append(f"could not verify perp exposure after failure: {verify_exc}")
            result.needs_manual_intervention = True
            return result
        if signed_perp < -1e-8:
            # ショートが実在する: 現物を売ると裸のショートが残ってしまうので自動では触らない
            result.perp_filled = True
            result.perp_amount = abs(signed_perp)
            result.errors.append("perp exposure exists after order failure; refusing automatic spot unwind")
            result.needs_manual_intervention = True
            return result
        try:
            unwind_order = spot.create_market_sell_order(symbol, result.spot_amount)
            _filled_amount(spot, symbol, unwind_order, contract=False)
            result.unwound = True
            result.spot_filled = False
            result.spot_amount = 0.0
        except Exception as unwind_exc:  # noqa: BLE001
            result.errors.append(f"unwind of spot leg also failed: {unwind_exc}")
            result.needs_manual_intervention = True  # 現物ロングだけが裸で残っている
    return result


def close_carry_orders(symbol: str, notional_usd: float | None = None, dry_run: bool = True) -> CarryExecutionResult:
    """キャリーポジションを手仕舞う。先物の買い戻しを先に行い、成功してから現物を
    売却する(現物売却が失敗しても残るのは低リスクな現物ロングのみにするため)。

    手仕舞い数量は常にその場で取引所から取得した実際の保有数量を使う(現在価格からの
    再計算はしない)。呼び出し側の状態ファイルに記録されている数量は使わない
    — reconcile_stateで事前に一致確認していても、確認から発注までの間にズレる
    可能性をゼロにするため、実際に閉じる量は必ずここで取引所から取り直す。
    さらに、実際のポジションが「釣り合ったcarry」に分類できない場合(現物のみ/
    先物のみ/大きく不均衡)は発注せず停止する。notional_usdはdry-run表示専用。
    """
    if dry_run:
        print(f"[DRY RUN] would BUY(cover) perp {symbol} and SELL spot using the actual held base quantities")
        return CarryExecutionResult(dry_run=True, spot_filled=True, perp_filled=True)

    result = CarryExecutionResult(dry_run=False)
    actual = fetch_actual_position(symbol)
    if classify_actual_position(actual) != CarryAccountState.CARRY_BALANCED:
        result.errors.append(f"refusing close: exchange position is not a balanced carry (actual={actual})")
        result.needs_manual_intervention = True
        return result

    target_perp_base = abs(actual["perp_amount"])
    target_spot_base = actual["spot_amount"]

    futures = get_testnet_futures_exchange()
    perp_symbol = f"{symbol}:USDT"
    try:
        perp_contracts = _base_to_contract_amount(futures, perp_symbol, target_perp_base)
        order = futures.create_order(perp_symbol, "market", "buy", perp_contracts, params={"reduceOnly": True})
        result.perp_amount, result.perp_order_id = _filled_amount(futures, perp_symbol, order, contract=True)
        result.perp_filled = _amounts_close(result.perp_amount, target_perp_base)
        if not result.perp_filled:
            result.errors.append(
                f"perp cover only filled {result.perp_amount:.12g} of {target_perp_base:.12g} base units"
            )
            result.needs_manual_intervention = True
            return result
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"perp cover failed or fill could not be verified: {exc}")
        result.needs_manual_intervention = True  # 先物ショートがまだ残っている(レバレッジ・清算リスクあり)
        return result

    try:
        spot = get_testnet_spot_exchange()
        order = spot.create_market_sell_order(symbol, target_spot_base)
        result.spot_amount, result.spot_order_id = _filled_amount(spot, symbol, order, contract=False)
        result.spot_filled = _amounts_close(result.spot_amount, target_spot_base)
        if not result.spot_filled:
            result.errors.append(
                f"spot close only filled {result.spot_amount:.12g} of {target_spot_base:.12g} base units"
            )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"spot sell failed: {exc}")
        # 先物は既に手仕舞い済みで残るのは現物ロングのみ(低リスク)。次回再試行される。
    return result


def close_spot_only(symbol: str, dry_run: bool = True) -> CarryExecutionResult:
    """spot_only状態(先物は閉じたが現物売却が残っている等)の復旧専用処理。

    close_carry_orders同様、手仕舞い数量は常にその場で取引所から取り直す。
    """
    if dry_run:
        print(f"[DRY RUN] would SELL residual spot {symbol} using the actual held base quantity")
        return CarryExecutionResult(dry_run=True, spot_filled=True, perp_filled=True)

    result = CarryExecutionResult(dry_run=False, perp_filled=True)
    actual = fetch_actual_position(symbol)
    if abs(actual["perp_amount"]) > 1e-8:
        result.errors.append(f"refusing spot-only recovery while perp exposure exists: {actual}")
        result.needs_manual_intervention = True
        return result
    target = actual["spot_amount"]
    if target <= 0:
        result.spot_filled = True  # 既に売却済み(何もしない)
        return result
    try:
        spot = get_testnet_spot_exchange()
        order = spot.create_market_sell_order(symbol, target)
        result.spot_amount, result.spot_order_id = _filled_amount(spot, symbol, order, contract=False)
        result.spot_filled = _amounts_close(result.spot_amount, target)
        if not result.spot_filled:
            result.errors.append(f"residual spot close only filled {result.spot_amount:.12g} of {target:.12g}")
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"residual spot sell failed: {exc}")
    return result
