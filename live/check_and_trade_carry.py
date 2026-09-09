"""ファンディング・キャリーの現在のシグナルを確認し、(デフォルトはdry-runで)発注する。

liveパスはfail-closed: 発注前に必ず取引所の実状態と照合し、状態ファイルはatomicに
書き込み、同じ銘柄を同時に複数プロセスが操作しないようロックを取る。

backtests/run_funding_carry.pyの検証結果に基づく設計上の注意点:
- OOSでプラスだったのは「常に建てっぱなし」のalways_on_carryであり、直近7日平均を
  日次で評価して出し入れするconditional_carryはOOSでコスト負けしていた
  (BTC test -3.79%、ETH test -4.94%)。そのため、このスクリプトを毎日実行して
  日々判定を切り替えるような使い方は、バックテストで負けが確認されている運用を
  再現してしまう。**週1回程度の実行を想定**し、かつ直近でenter/exitした後
  min_hold_daysが経過するまでは反対方向のアクションを取らない、というクールダウンを
  入れている。

Usage:
    python live/check_and_trade_carry.py --symbol BTC/USDT --notional 100
    python live/check_and_trade_carry.py --symbol BTC/USDT --notional 100 --live   # テストネットで実発注

--live を付けない限り、実際の注文は一切送信されない(dry-run)。
--live を付けた場合でも、.envにテストネットAPIキーが無ければ実行時エラーになる
(本番/実資金のキーはこのスクリプトでは意図的にサポートしない)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data import fetch_funding_rate  # noqa: E402
from execution import (  # noqa: E402
    close_carry_orders,
    close_spot_only,
    decide_carry_action,
    place_carry_orders,
    reconcile_state,
)

STATE_DIR = ROOT / "live" / "state"
WINDOW = 21  # 21本 x 8h = 7日(strategies/funding_carry.pyと合わせる)

DEFAULT_STATE = {
    "status": "flat",  # "flat" | "carry" | "spot_only"
    "needs_manual_intervention": False,
    "last_action_at": None,
    "last_error": None,
    "spot_amount": None,
    "perp_amount": None,
    "spot_order_id": None,
    "perp_order_id": None,
}


def _state_path(symbol: str) -> Path:
    return STATE_DIR / f"{symbol.replace('/', '-')}.json"


def load_state(symbol: str) -> dict:
    path = _state_path(symbol)
    if path.exists():
        return {**DEFAULT_STATE, **json.loads(path.read_text())}
    return dict(DEFAULT_STATE)


def save_state(symbol: str, state: dict) -> None:
    """プロセスがクラッシュしても壊れたJSONが残らないよう、一時ファイルに書いてから
    atomicにリネームする。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(symbol)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
    tmp.write_text(payload)
    os.replace(tmp, path)


@contextmanager
def state_lock(symbol: str):
    """同じ銘柄を複数プロセスが同時に操作しないための簡易ロック。

    fcntl/msvcrtなどOS依存のファイルロックAPIは使わず、`open(..., "x")`
    (作成時に既存なら失敗する排他的作成)だけで実装する(Windows/Unix両対応、
    追加の依存ライブラリも不要)。プロセスが異常終了するとロックファイルが
    残ったままになりうるので、その場合は中身(PID等)を確認した上で手動で
    削除すること。
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _state_path(symbol).with_suffix(".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another process may already be managing {symbol} "
            f"(lock file exists: {lock_path}; delete it manually if you're sure no other run is active)"
        ) from exc
    try:
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def within_cooldown(state: dict, min_hold_days: float) -> bool:
    if not state.get("last_action_at"):
        return False
    last = datetime.fromisoformat(state["last_action_at"])
    return datetime.now(timezone.utc) - last < timedelta(days=min_hold_days)


def _clear_position_state(state: dict) -> None:
    state.update(
        {
            "status": "flat",
            "spot_amount": None,
            "perp_amount": None,
            "spot_order_id": None,
            "perp_order_id": None,
            "last_error": None,
            "needs_manual_intervention": False,
        }
    )


def _run(args: argparse.Namespace) -> None:
    dry_run = not args.live
    state = load_state(args.symbol)

    if state["needs_manual_intervention"]:
        print(
            f"BLOCKED: {args.symbol} state requires manual intervention "
            f"(last_error={state.get('last_error')!r}). Check the exchange and state file before retrying."
        )
        return

    if not dry_run:
        matches, message = reconcile_state(
            args.symbol,
            expected_status=state["status"],
            expected_spot_amount=state.get("spot_amount"),
            expected_perp_amount=state.get("perp_amount"),
        )
        print(f"reconcile: {message}")
        if not matches:
            state["needs_manual_intervention"] = True
            state["last_error"] = message
            save_state(args.symbol, state)
            print("BLOCKED: state/exchange mismatch. Stopping before placing any order.")
            return

    # spot_onlyは「先物は閉じたが現物売却が残った」ような復旧待ちの状態。
    # 新しいエントリー判断をする前にまずこれを片付けないと、現物ポジションが
    # 二重に積み上がってしまう。
    if state["status"] == "spot_only":
        print("recovery: residual spot-only position detected; retrying spot close")
        result = close_spot_only(args.symbol, spot_amount=state.get("spot_amount"), dry_run=dry_run)
        if result.spot_filled:
            _clear_position_state(state)
            state["last_action_at"] = datetime.now(timezone.utc).isoformat()
            if not dry_run:
                save_state(args.symbol, state)
            print("recovery complete: state=flat")
        else:
            state["last_error"] = "; ".join(result.errors)
            state["needs_manual_intervention"] = result.needs_manual_intervention
            if not dry_run:
                save_state(args.symbol, state)
            print(f"recovery failed: {result.errors}")
        return

    perp_symbol = f"{args.symbol}:USDT"
    funding = fetch_funding_rate(perp_symbol, total_records=WINDOW)
    if len(funding) < WINDOW:
        raise RuntimeError(f"insufficient funding history: expected {WINDOW} records, got {len(funding)}")
    trailing_avg = float(funding["funding_rate"].iloc[-WINDOW:].mean())

    currently_positioned = state["status"] == "carry"
    decision = decide_carry_action(trailing_avg, currently_positioned=currently_positioned)

    print(f"{args.symbol}: trailing {WINDOW * 8}h funding avg = {trailing_avg:+.4%}/interval")
    print(f"status={state['status']} raw_decision={decision.action} ({decision.reason})")
    if decision.action in ("enter", "exit") and within_cooldown(state, args.min_hold_days):
        print(f"cooldown active (min_hold_days={args.min_hold_days}): overriding to hold/stay_flat")
        decision.action = "hold" if currently_positioned else "stay_flat"

    print(f"final decision: {decision.action}")

    if decision.action == "enter":
        result = place_carry_orders(args.symbol, args.notional, dry_run=dry_run)
        if result.fully_positioned:
            state.update(
                status="carry",
                spot_amount=result.spot_amount or None,
                perp_amount=result.perp_amount or None,
                spot_order_id=result.spot_order_id,
                perp_order_id=result.perp_order_id,
                last_action_at=datetime.now(timezone.utc).isoformat(),
                last_error=None,
            )
        elif result.needs_manual_intervention:
            state.update(
                status="spot_only" if result.spot_filled and not result.perp_filled else state["status"],
                spot_amount=result.spot_amount or state.get("spot_amount"),
                perp_amount=result.perp_amount or state.get("perp_amount"),
                spot_order_id=result.spot_order_id,
                perp_order_id=result.perp_order_id,
                needs_manual_intervention=True,
                last_error="; ".join(result.errors),
            )
        if result.errors:
            print(f"errors: {result.errors}")

    elif decision.action == "exit":
        result = close_carry_orders(
            args.symbol,
            args.notional,
            dry_run=dry_run,
            spot_amount=state.get("spot_amount"),
            perp_amount=state.get("perp_amount"),
        )
        if result.fully_positioned:
            _clear_position_state(state)
            state["last_action_at"] = datetime.now(timezone.utc).isoformat()
        elif result.perp_filled and not result.spot_filled:
            state["status"] = "spot_only"  # 先物は閉じた、現物売却は次回再試行
            state["perp_amount"] = None
            state["perp_order_id"] = result.perp_order_id
            state["last_error"] = "; ".join(result.errors)
        elif result.needs_manual_intervention:
            state["needs_manual_intervention"] = True
            state["last_error"] = "; ".join(result.errors)
        if result.errors:
            print(f"errors: {result.errors}")
    else:
        print("no order needed")

    if not dry_run:
        save_state(args.symbol, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--notional", type=float, default=100.0)
    parser.add_argument("--live", action="store_true", help="テストネットで実際に発注する(デフォルトはdry-run)")
    parser.add_argument(
        "--min-hold-days",
        type=float,
        default=14.0,
        help="直近のenter/exitからこの日数が経つまでは反対方向のアクションを取らない(頻繁な出し入れによるコスト負けを防ぐ)",
    )
    args = parser.parse_args()
    if args.notional <= 0:
        parser.error("--notional must be positive")
    if args.min_hold_days < 0:
        parser.error("--min-hold-days must be non-negative")

    if args.live:
        with state_lock(args.symbol):
            _run(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()
