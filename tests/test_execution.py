import execution


class FakeSpotExchange:
    def __init__(self, buy_filled=1.0, sell_filled=None, fail_buy=False, fail_sell=False):
        self.buy_filled = buy_filled
        self.sell_filled = sell_filled
        self.fail_buy = fail_buy
        self.fail_sell = fail_sell
        self.calls = []

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_market_buy_order(self, symbol, amount):
        self.calls.append(("buy", amount))
        if self.fail_buy:
            raise RuntimeError("spot buy failed")
        return {"id": "spot-buy", "status": "closed", "amount": amount, "filled": self.buy_filled, "remaining": 0.0}

    def create_market_sell_order(self, symbol, amount):
        self.calls.append(("sell", amount))
        if self.fail_sell:
            raise RuntimeError("spot sell failed")
        filled = amount if self.sell_filled is None else self.sell_filled
        return {"id": "spot-sell", "status": "closed", "amount": amount, "filled": filled, "remaining": 0.0}

    def fetch_balance(self):
        return {"total": {}}


class FakeFuturesExchange:
    def __init__(
        self,
        sell_filled=None,
        buy_filled=None,
        fail_sell=False,
        fail_buy=False,
        signed_position=0.0,
        fail_leverage=False,
    ):
        self.sell_filled = sell_filled
        self.buy_filled = buy_filled
        self.fail_sell = fail_sell
        self.fail_buy = fail_buy
        self.signed_position = signed_position
        self.fail_leverage = fail_leverage
        self.calls = []
        self.leverage_calls = []

    def load_markets(self):
        return None

    def market(self, symbol):
        return {"contractSize": 1.0}

    def amount_to_precision(self, symbol, amount):
        return str(amount)

    def set_leverage(self, leverage, symbol):
        self.leverage_calls.append((leverage, symbol))
        if self.fail_leverage:
            raise RuntimeError("leverage endpoint unavailable")

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def create_order(self, symbol, order_type, side, amount, params=None):
        self.calls.append((side, amount, params))
        if side == "sell" and self.fail_sell:
            raise RuntimeError("perp short failed")
        if side == "buy" and self.fail_buy:
            raise RuntimeError("perp cover failed")
        filled = amount
        if side == "sell" and self.sell_filled is not None:
            filled = self.sell_filled
        if side == "buy" and self.buy_filled is not None:
            filled = self.buy_filled
        return {"id": f"perp-{side}", "status": "closed", "amount": amount, "filled": filled, "remaining": 0.0}

    def fetch_positions(self, symbols):
        if self.signed_position == 0:
            return []
        return [
            {
                "symbol": symbols[0],
                "contracts": abs(self.signed_position),
                "side": "short" if self.signed_position < 0 else "long",
            }
        ]


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


def test_place_hedges_actual_spot_fill_not_ticker_notional(monkeypatch):
    # requested amount from notional/price is 1.0, but only 0.8 actually filled;
    # the perp leg must be sized off the real fill, not the original request.
    spot = FakeSpotExchange(buy_filled=0.8)
    futures = FakeFuturesExchange(sell_filled=0.8)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.fully_positioned
    assert result.spot_amount == 0.8
    assert result.perp_amount == 0.8
    assert futures.calls[0][0:2] == ("sell", 0.8)


def test_place_carry_orders_sets_1x_leverage_before_any_order(monkeypatch):
    spot = FakeSpotExchange(buy_filled=0.8)
    futures = FakeFuturesExchange(sell_filled=0.8)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.fully_positioned
    assert futures.leverage_calls == [(1, "BTC/USDT:USDT")]


def test_place_carry_orders_aborts_if_leverage_cannot_be_set(monkeypatch):
    # leverage must be confirmed before any order is placed - if it can't be
    # set, abort cleanly rather than risk trading at the account's default
    # (possibly much higher) leverage.
    spot = FakeSpotExchange()
    futures = FakeFuturesExchange(fail_leverage=True)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert not result.fully_positioned
    assert not result.needs_manual_intervention  # nothing was touched, safe to just retry
    assert spot.calls == []
    assert futures.calls == []


def test_place_carry_orders_aborts_if_exchange_has_no_set_leverage_method(monkeypatch):
    # An exchange object with no set_leverage() at all must be treated the same
    # as any other failure to confirm 1x leverage: abort before touching either
    # leg, rather than silently skipping the leverage guarantee and proceeding
    # at whatever leverage the account happens to default to.
    class FuturesExchangeWithoutLeverageControl:
        def __init__(self):
            self.calls = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def create_order(self, symbol, order_type, side, amount, params=None):
            self.calls.append((side, amount, params))
            return {"id": "perp-sell", "status": "closed", "amount": amount, "filled": amount, "remaining": 0.0}

    spot = FakeSpotExchange()
    futures = FuturesExchangeWithoutLeverageControl()
    assert not hasattr(futures, "set_leverage")
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert not result.fully_positioned
    assert not result.needs_manual_intervention  # nothing was touched, safe to just retry
    assert any("set_leverage" in e for e in result.errors)
    assert spot.calls == []
    assert futures.calls == []


def test_perp_failure_without_exchange_exposure_unwinds_exact_spot_fill(monkeypatch):
    spot = FakeSpotExchange(buy_filled=0.8)
    futures = FakeFuturesExchange(fail_sell=True, signed_position=0.0)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.unwound
    assert ("sell", 0.8) in spot.calls
    assert not result.needs_manual_intervention
    assert not result.fully_positioned


def test_perp_failure_with_exchange_exposure_refuses_automatic_spot_unwind(monkeypatch):
    # the perp order raised, but the exchange shows a short already exists (e.g. a
    # dropped response after the order was actually accepted) - selling spot here
    # would leave a naked, unhedged short, so it must refuse and flag for a human.
    spot = FakeSpotExchange(buy_filled=0.8)
    futures = FakeFuturesExchange(fail_sell=True, signed_position=-0.8)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.needs_manual_intervention
    assert ("sell", 0.8) not in spot.calls


def test_ambiguous_failure_never_reports_fully_positioned_even_if_amounts_match(monkeypatch):
    # Regression test: spot_amount and the discovered perp short happen to match
    # exactly, but the perp order still raised. needs_manual_intervention=True must
    # force fully_positioned=False so a caller checking `if fully_positioned: ...
    # elif needs_manual_intervention: ...` cannot silently skip the flag.
    spot = FakeSpotExchange(buy_filled=0.8)
    futures = FakeFuturesExchange(fail_sell=True, signed_position=-0.8)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.spot_amount == result.perp_amount == 0.8
    assert result.needs_manual_intervention
    assert result.fully_positioned is False


def test_place_carry_orders_spot_leg_fails_never_touches_futures(monkeypatch):
    spot = FakeSpotExchange(fail_buy=True)
    futures = FakeFuturesExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert not result.fully_positioned
    assert not result.needs_manual_intervention
    assert futures.calls == []  # never attempted the second leg


def test_place_carry_orders_hedge_mismatch_flags_manual_intervention(monkeypatch):
    spot = FakeSpotExchange(buy_filled=1.0)
    futures = FakeFuturesExchange(sell_filled=0.5)  # only half the intended hedge filled
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.place_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.needs_manual_intervention
    assert not result.fully_positioned


# --- close_carry_orders: leg-failure handling ---


def test_close_uses_actual_exchange_quantities_not_current_price(monkeypatch):
    spot = FakeSpotExchange()
    futures = FakeFuturesExchange(buy_filled=0.8)
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 0.8, "perp_amount": -0.8})
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.fully_positioned
    assert futures.calls[0] == ("buy", 0.8, {"reduceOnly": True})
    assert ("sell", 0.8) in spot.calls


def test_close_carry_orders_perp_cover_fails_leaves_hedge_intact(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": -1.0})
    spot = FakeSpotExchange()
    futures = FakeFuturesExchange(fail_buy=True)
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.needs_manual_intervention
    assert not result.perp_filled
    assert spot.calls == []  # never attempted to sell spot while still short perp


def test_close_carry_orders_spot_sell_fails_after_perp_covered_is_low_risk(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": -1.0})
    spot = FakeSpotExchange(fail_sell=True)
    futures = FakeFuturesExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)
    monkeypatch.setattr(execution, "get_testnet_futures_exchange", lambda: futures)

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.perp_filled
    assert not result.spot_filled
    assert not result.needs_manual_intervention  # leftover is just a long spot position


def test_close_carry_orders_refuses_when_no_short_to_cover(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": 0.0})

    result = execution.close_carry_orders("BTC/USDT", 100.0, dry_run=False)

    assert result.needs_manual_intervention
    assert not result.fully_positioned


# --- close_spot_only ---


def test_close_spot_only_sells_residual_spot(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 0.8, "perp_amount": 0.0})
    spot = FakeSpotExchange()
    monkeypatch.setattr(execution, "get_testnet_spot_exchange", lambda: spot)

    result = execution.close_spot_only("BTC/USDT", dry_run=False)

    assert result.spot_filled
    assert ("sell", 0.8) in spot.calls


def test_close_spot_only_refuses_while_perp_exposure_exists(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 0.8, "perp_amount": -0.2})

    result = execution.close_spot_only("BTC/USDT", dry_run=False)

    assert result.needs_manual_intervention
    assert not result.spot_filled


# --- reconcile_state / classify_actual_position ---


def test_reconcile_state_matches_when_exchange_agrees_with_saved_state(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": -1.0})

    matches, _ = execution.reconcile_state("BTC/USDT", state_positioned=True)

    assert matches


def test_reconcile_state_detects_mismatch(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 0.0, "perp_amount": 0.0})

    matches, message = execution.reconcile_state("BTC/USDT", state_positioned=True)

    assert not matches
    assert "MISMATCH" in message


def test_flat_saved_state_does_not_accept_spot_only_exchange_state(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": 0.0})

    matches, message = execution.reconcile_state("BTC/USDT", expected_status="flat")

    assert not matches
    assert "spot_only" in message


def test_carry_reconcile_requires_balanced_quantities(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": -0.2})

    matches, message = execution.reconcile_state("BTC/USDT", expected_status="carry")

    assert not matches
    assert "carry_imbalanced" in message


def test_carry_reconcile_checks_recorded_quantities(monkeypatch):
    monkeypatch.setattr(execution, "fetch_actual_position", lambda symbol: {"spot_amount": 1.0, "perp_amount": -1.0})

    matches, _ = execution.reconcile_state(
        "BTC/USDT", expected_status="carry", expected_spot_amount=1.0, expected_perp_amount=1.0
    )

    assert matches


def test_classify_perp_only_never_looks_flat():
    observed = execution.classify_actual_position({"spot_amount": 0.0, "perp_amount": -1.0})
    assert observed is execution.CarryAccountState.PERP_ONLY
