"""ファンディング・キャリーの現在のシグナルを確認し、(デフォルトはdry-runで)発注する。

backtests/run_funding_carry.pyの検証結果に基づく設計上の注意点:
- OOSでプラスだったのは「常に建てっぱなし」のalways_on_carryであり、直近7日平均を
  日次で評価して出し入れするconditional_carryはOOSでコスト負けしていた
  (BTC test -3.79%、ETH test -4.94%)。そのため、このスクリプトを毎日実行して
  日々判定を切り替えるような使い方は、バックテストで負けが確認されている運用を
  再現してしまう。**週1回程度の実行を想定**し、かつ直近でenter/exitした後
  min_hold_daysが経過するまでは反対方向のアクションを取らない、というクールダウンを
  入れている。
- 実行状態(建玉があるか)はJSON(live/state/)で管理しているが、プロセス停止・手動注文・
  部分約定・取引所側のリセットなどでJSONと実際の口座が食い違いうる。--liveでの実行時は
  発注前に必ず取引所の実残高/建玉と状態ファイルを突き合わせ(execution.reconcile_state)、
  食い違いがあれば自動売買を止める。
- 2レッグ発注は原子的ではないため、片方だけ約定する状態が起こりうる
  (詳細はsrc/execution.pyのモジュールdocstring参照)。needs_manual_intervention=Trueが
  一度立ったら、このスクリプトは人手で状態ファイルを確認・修正するまで自動売買を止める。

Usage:
    python live/check_and_trade_carry.py --symbol BTC/USDT --notional 100
    python live/check_and_trade_carry.py --symbol BTC/USDT --notional 100 --live   # テストネットで実発注

--live を付けない限り、実際の注文は一切送信されない(dry-run)。
--live を付けた場合でも、.envにテストネットAPIキーが無ければ実行時エラーになる
(本番/実資金のキーはこのスクリプトでは意図的にサポートしない)。
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data import fetch_funding_rate  # noqa: E402
from execution import close_carry_orders, decide_carry_action, place_carry_orders, reconcile_state  # noqa: E402

STATE_DIR = ROOT / "live" / "state"
WINDOW = 21  # 21本 x 8h = 7日(strategies/funding_carry.pyと合わせる)

DEFAULT_STATE = {
    "status": "flat",  # "flat" | "carry" | "spot_only"
    "needs_manual_intervention": False,
    "last_action_at": None,
    "last_error": None,
}


def load_state(symbol: str) -> dict:
    path = STATE_DIR / f"{symbol.replace('/', '-')}.json"
    if path.exists():
        return {**DEFAULT_STATE, **json.loads(path.read_text())}
    return dict(DEFAULT_STATE)


def save_state(symbol: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{symbol.replace('/', '-')}.json"
    path.write_text(json.dumps(state, indent=2))


def within_cooldown(state: dict, min_hold_days: float) -> bool:
    if not state.get("last_action_at"):
        return False
    last = datetime.fromisoformat(state["last_action_at"])
    return datetime.now(timezone.utc) - last < timedelta(days=min_hold_days)


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
    dry_run = not args.live

    state = load_state(args.symbol)

    if state["needs_manual_intervention"]:
        print(
            f"BLOCKED: {args.symbol} state requires manual intervention "
            f"(last_error={state.get('last_error')!r}). "
            f"Check the exchange manually, fix live/state/{args.symbol.replace('/', '-')}.json, then retry."
        )
        return

    if not dry_run:
        matches, message = reconcile_state(args.symbol, state_positioned=(state["status"] == "carry"))
        print(f"reconcile: {message}")
        if not matches:
            state["needs_manual_intervention"] = True
            state["last_error"] = message
            save_state(args.symbol, state)
            print("BLOCKED: state/exchange mismatch. Stopping before placing any order.")
            return

    perp_symbol = f"{args.symbol}:USDT"
    funding = fetch_funding_rate(perp_symbol, total_records=WINDOW)
    trailing_avg = funding["funding_rate"].mean()

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
            state["status"] = "carry"
            state["last_action_at"] = datetime.now(timezone.utc).isoformat()
        elif result.needs_manual_intervention:
            state["status"] = "spot_only"
            state["needs_manual_intervention"] = True
            state["last_error"] = "; ".join(result.errors)
        # 何も約定しなかった/巻き戻して元通り(flat)になった場合はstateを変更しない
        if result.errors:
            print(f"errors: {result.errors}")
    elif decision.action == "exit":
        result = close_carry_orders(args.symbol, args.notional, dry_run=dry_run)
        if result.fully_positioned:
            state["status"] = "flat"
            state["last_action_at"] = datetime.now(timezone.utc).isoformat()
        elif result.perp_filled and not result.spot_filled:
            state["status"] = "spot_only"  # 先物は閉じたが現物売却が失敗、低リスクなので次回再試行
            state["last_error"] = "; ".join(result.errors)
        elif result.needs_manual_intervention:
            # 先物の買い戻し自体が失敗し、レバレッジのかかったショートがまだ残っている
            state["needs_manual_intervention"] = True
            state["last_error"] = "; ".join(result.errors)
        if result.errors:
            print(f"errors: {result.errors}")
    else:
        print("no order needed")

    if not dry_run:
        save_state(args.symbol, state)


if __name__ == "__main__":
    main()
