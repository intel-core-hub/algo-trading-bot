import execution


class FakeSpotExchange:
    def __init__(self, fail_buy: bool = False, fail_sell: bool = False):
        self.fail_buy = fail_buy
        self.fail_sell = fail_sell
        self.calls = []

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_market_buy_order(self, symbol, amount):
        self.calls.append(("buy", amount))
        if self.fail_buy:
            raise RuntimeError("spot buy failed")

    def create_market_sell_order(self, symbol, amount):
        self.calls.append(("sell", amount))
        if self.fail_sell:
            raise RuntimeError("spot sell failed")

    def fetch_balance(self):
        return {"total": {}}


class FakeFuturesExchange:
    def __init__(self, fail_sell: bool = False, fail_buy: bool = False):
        self.fail_sell = fail_sell
        self.fail_buy = fail_buy
        self.calls = []

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_order(self, symbol, order_type, side, amount, params=None):
        self.calls.append((side, amount, params))
        if side == "sell" and self.fail_sell:
            raise RuntimeError("perp short failed")
        if side == "buy" and self.fail_buy:
            raise RuntimeError("perp cover failed")

    def fetch_positions(self, symbols):
        return []


# --- decide_carry_action ---


def test_decide_enter_when_funding_positive_and_flat():
    decision = execution.decide_carry_action(0.001, currently_positioned=False)
    assert decision.action == "enter"


def test_decide_hold_when_funding_positive_and_positioned():
    decision = execution.decide_carry_action(0.001, currently_positioned=True)
    assert decision.action == "hold"


def test_decide_exit_when_funding_negative_and_positioned():
    decision = execution.decide_carry_action(-0.001, currently_positioned=True)
    assert decision.action == "exit"


def test_decide_stay_flat_when_funding_negative_and_flat():
    decision = execution.decide_carry_action(-0.001, currently_positioned=False)
    assert decision.action == "stay_flat"


# --- place_carry_orders: dry run never touches the network ---


def test_place_carry_orders_dry_run_reports_success_without_calling_exchange(monkeypatch):
    def boom():
        raise AssertionError("dry_run must not construct a real exchange client")

    monkeypatch.setattr(execution, "get_testnet_spot_exchange", boom)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", boom)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=True)

    assert result.dry_run is True
    assert result.fully_positioned


# --- place_carry_orders: leg-failure handling ---


def test_place_carry_orders_success_both_legs(monkeypatch):
    spot = FakeSpotExchange()
    futures = FakeFuturesExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.fully_positioned
    assert not result.needs_manual_intervention
    assert not result.errors


def test_place_carry_orders_perp_fails_unwinds_spot(monkeypatch):
    spot = FakeSpotExchange()  # unwind sell will succeed
    futures = FakeFuturesExchange(fail_sell=True)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert not result.fully_positioned
    assert result.unwound
    assert not result.spot_filled
    assert not result.needs_manual_intervention
    assert ("sell", 1.0) in spot.calls


def test_place_carry_orders_perp_fails_and_unwind_also_fails_needs_manual_intervention(monkeypatch):
    spot = FakeSpotExchange(fail_sell=True)  # unwind attempt will also fail
    futures = FakeFuturesExchange(fail_sell=True)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.needs_manual_intervention
    assert not result.fully_positioned
    assert result.spot_filled  # naked spot long left behind - the least-bad leftover state


def test_place_carry_orders_spot_leg_fails_never_touches_futures(monkeypatch):
    spot = FakeSpotExchange(fail_buy=True)
    futures = FakeFuturesExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert not result.fully_positioned
    assert not result.needs_manual_intervention
    assert futures.calls == []  # never attempted the second leg


# --- close_carry_orders: leg-failure handling ---


def test_close_carry_orders_success_both_legs(monkeypatch):
    spot = FakeSpotExchange()
    futures = FakeFuturesExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.fully_positioned
    assert not result.needs_manual_intervention


def test_close_carry_orders_perp_cover_fails_leaves_hedge_intact(monkeypatch):
    spot = FakeSpotExchange()
    futures = FakeFuturesExchange(fail_buy=True)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.needs_manual_intervention
    assert not result.perp_filled
    assert spot.calls == []  # never attempted to sell spot while still short perp


def test_close_carry_orders_spot_sell_fails_after_perp_covered_is_low_risk(monkeypatch):
    spot = FakeSpotExchange(fail_sell=True)
    futures = FakeFuturesExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.perp_filled
    assert not result.spot_filled
    assert not result.needs_manual_intervention  # leftover is just a long spot position


# --- reconcile_state ---


def test_reconcile_state_matches_when_exchange_agrees_with_saved_state(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": -1.0})

    matches, _ = execution.reconcile_state("BTC/USDT", state_positioned=True)

    assert matches


def test_reconcile_state_detects_mismatch(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 0.0, "perp_amount": 0.0})

    matches, message = execution.reconcile_state("BTC/USDT", state_positioned=True)

    assert not matches
    assert "MISMATCH" in message
