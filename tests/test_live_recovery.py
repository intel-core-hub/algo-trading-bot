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


def test_state_lock_aborts_before_touching_exchange_when_already_locked(monkeypatch, tmp_path):
    """Windows実機で、ロックファイルが既に存在する(=別プロセスが処理中)状態から
    --liveで起動した場合に、reconcile_state等のAPIキーを要する処理へ進む前に
    確実に中断されることを確認する(手元でstale lockを模した再現テスト)。"""
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock_path = tmp_path / "BTC-USDT.lock"
    lock_path.write_text("99999 stale")
    monkeypatch.setattr(
        mod, "reconcile_state", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach reconcile_state"))
    )

    try:
        with mod.state_lock("BTC/USDT"):
            raise AssertionError("must not enter the locked block")
    except RuntimeError as exc:
        assert "BTC/USDT" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for an already-locked symbol")

    # the pre-existing (stale) lock file must be left alone, not consumed by us
    assert lock_path.read_text() == "99999 stale"


def test_state_lock_releases_lock_even_when_wrapped_code_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    lock_path = tmp_path / "BTC-USDT.lock"

    try:
        with mod.state_lock("BTC/USDT"):
            assert lock_path.exists()
            raise RuntimeError("boom")
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected the original exception to propagate")

    assert not lock_path.exists()


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
