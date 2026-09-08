"""ファンディング・キャリーの現在のシグナルを確認し、(デフォルトはdry-runで)発注する。

backtests/run_funding_carry.pyの検証結果に基づき、BTC/ETHで直近7日平均の資金調達率が
プラスなら建てる/維持、マイナスに転じたら手仕舞う、という判断を行う。1日1回程度の
頻度で実行する想定(頻繁な出し入れはバックテストでコスト負けすると判明済み)。

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data import fetch_funding_rate  # noqa: E402
from execution import close_carry_orders, decide_carry_action, place_carry_orders  # noqa: E402

STATE_DIR = ROOT / "live" / "state"
WINDOW = 21  # 21本 x 8h = 7日(strategies/funding_carry.pyと合わせる)


def load_state(symbol: str) -> dict:
    path = STATE_DIR / f"{symbol.replace('/', '-')}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"positioned": False}


def save_state(symbol: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{symbol.replace('/', '-')}.json"
    path.write_text(json.dumps(state))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--notional", type=float, default=100.0)
    parser.add_argument("--live", action="store_true", help="テストネットで実際に発注する(デフォルトはdry-run)")
    args = parser.parse_args()

    perp_symbol = f"{args.symbol}:USDT"
    funding = fetch_funding_rate(perp_symbol, total_records=WINDOW)
    trailing_avg = funding["funding_rate"].mean()

    state = load_state(args.symbol)
    decision = decide_carry_action(trailing_avg, currently_positioned=state["positioned"])

    print(f"{args.symbol}: trailing {WINDOW * 8}h funding avg = {trailing_avg:+.4%}/interval")
    print(f"decision: {decision.action} ({decision.reason})")

    dry_run = not args.live
    if decision.action == "enter":
        place_carry_orders(args.symbol, args.notional, dry_run=dry_run)
        state["positioned"] = True
    elif decision.action == "exit":
        close_carry_orders(args.symbol, args.notional, dry_run=dry_run)
        state["positioned"] = False
    else:
        print("no order needed")

    if not dry_run:
        save_state(args.symbol, state)


if __name__ == "__main__":
    main()
