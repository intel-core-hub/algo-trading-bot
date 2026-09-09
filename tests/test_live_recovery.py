"""live/check_and_trade_carry.py はパッケージ化されていないスクリプトなので、
importlibでファイルパスから直接読み込み、main()をargv経由で実行してテストする。"""

import argparse
import importlib.util
import sys
from pathlib import Path

import execution

MODULE_PATH = Path(__file__).resolve().parents[1] / "live" / "check_and_trade_carry.py"
spec = importlib.util.spec_from_file_location("check_and_trade_carry", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_spot_only_recovery_bypasses_funding_decision(monkeypatch, tmp_path):
    """spot_onlyの復旧は、資金調達率を取得したりcooldown判定をしたりする前に、
    他の何よりも優先して行われなければならない(そうしないと新規エントリーが
    既存の残留ポジションの上に積み上がってしまう)。"""
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    mod.save_state("BTC/USDT", {**mod.DEFAULT_STATE, "status": "spot_only"})
    monkeypatch.setattr(mod, "reconcile_state", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(
        mod,
        "close_spot_only",
        lambda *a, **k: execution.CarryExecutionResult(dry_run=False, spot_filled=True, perp_filled=True),
    )
    monkeypatch.setattr(
        mod, "fetch_funding_rate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch funding"))
    )

    args = argparse.Namespace(symbol="BTC/USDT", notional=100.0, live=True, min_hold_days=14.0)
    mod._run(args)

    state = mod.load_state("BTC/USDT")
    assert state["status"] == "flat"
    assert not state["needs_manual_intervention"]


def test_needs_manual_intervention_blocks_before_any_exchange_call(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    mod.save_state(
        "BTC/USDT",
        {**mod.DEFAULT_STATE, "status": "carry", "needs_manual_intervention": True, "last_error": "boom"},
    )
    monkeypatch.setattr(
        mod, "reconcile_state", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reconcile"))
    )
    monkeypatch.setattr(
        mod, "fetch_funding_rate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch funding"))
    )

    args = argparse.Namespace(symbol="BTC/USDT", notional=100.0, live=True, min_hold_days=14.0)
    mod._run(args)  # should return early without raising

    state = mod.load_state("BTC/USDT")
    assert state["needs_manual_intervention"]  # unchanged, still blocked


def test_cooldown_overrides_exit_decision_to_hold(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    mod.save_state(
        "BTC/USDT",
        {**mod.DEFAULT_STATE, "status": "carry", "last_action_at": datetime.now(timezone.utc).isoformat()},
    )
    monkeypatch.setattr(mod, "fetch_funding_rate", lambda *a, **k: __import__("pandas").DataFrame({"funding_rate": [-0.01] * 21}))

    called = {"place": False, "close": False}
    monkeypatch.setattr(mod, "place_carry_orders", lambda *a, **k: called.__setitem__("place", True))
    monkeypatch.setattr(mod, "close_carry_orders", lambda *a, **k: called.__setitem__("close", True))

    args = argparse.Namespace(symbol="BTC/USDT", notional=100.0, live=False, min_hold_days=14.0)
    mod._run(args)

    # negative funding would normally trigger "exit", but the position was
    # just entered, so the cooldown must suppress it.
    assert not called["place"]
    assert not called["close"]
