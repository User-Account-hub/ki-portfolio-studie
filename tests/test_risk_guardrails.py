import pytest

from src.config import RiskConfig
from src.order_schema import ProposedOrder
from src.risk_guardrails import (
    OpenPosition,
    PortfolioContext,
    check_circuit_breaker,
    check_correlated_segment_exposure,
    check_daily_loss_stop,
    check_max_trades_per_symbol,
    check_micro_cap_exposure,
    check_min_cash_quota,
    check_no_margin,
    check_position_size,
    check_segment_weight,
    check_structured_products_cap,
    check_top3_concentration,
    check_trade_notional,
    evaluate_order,
    evaluate_short_positions_for_stop_loss,
)


def make_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_position_size_pct_of_portfolio=0.10,
        max_trade_notional_pct_of_nav=0.05,
        daily_loss_stop_pct=-0.03,
        max_trades_per_symbol_per_day=1,
        allow_short=True,
        allow_structured_products=True,
        structured_products_max_notional_pct_of_nav=0.20,
        short_stop_loss_pct=-0.20,
        allow_margin=False,
        max_segment_weight_pct_of_nav=0.30,
        correlated_crypto_mining_segments=["Krypto-Mining", "Digital Assets & Krypto-Oekosystem"],
        max_correlated_crypto_mining_pct_of_nav=0.30,
        max_micro_cap_pct_of_nav=0.15,
        max_top3_concentration_pct_of_nav=0.35,
        min_cash_pct_of_nav=0.05,
        circuit_breaker_drawdown_pct=-0.25,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def make_order(**overrides) -> ProposedOrder:
    defaults = dict(
        symbol="AAPL",
        instrument_type="equity",
        side="buy",
        quantity=10,
        rationale="test",
    )
    defaults.update(overrides)
    return ProposedOrder(**defaults)


def make_ctx(**overrides) -> PortfolioContext:
    defaults = dict(nav=100_000.0, cash=100_000.0, positions=[], trades_today_by_symbol={}, start_of_run_nav=100_000.0)
    defaults.update(overrides)
    return PortfolioContext(**defaults)


# --- max trades per symbol/day ---------------------------------------------


def test_max_trades_per_symbol_allows_first_trade():
    result = check_max_trades_per_symbol(make_order(), make_ctx(), max_per_day=1)
    assert result.approved


def test_max_trades_per_symbol_rejects_second_trade_same_day():
    ctx = make_ctx(trades_today_by_symbol={"AAPL": 1})
    result = check_max_trades_per_symbol(make_order(), ctx, max_per_day=1)
    assert not result.approved
    assert "AAPL" in result.reasons[0]


# --- no margin ---------------------------------------------------------------


def test_no_margin_rejects_buy_exceeding_cash():
    order = make_order(quantity=1000, side="buy")  # 1000 * 150 >> cash
    ctx = make_ctx(cash=500.0)
    result = check_no_margin(order, ctx, price=150.0, allow_margin=False)
    assert not result.approved


def test_no_margin_allows_buy_within_cash():
    order = make_order(quantity=1, side="buy")
    ctx = make_ctx(cash=500.0)
    result = check_no_margin(order, ctx, price=150.0, allow_margin=False)
    assert result.approved


def test_margin_allowed_when_config_permits():
    order = make_order(quantity=1000, side="buy")
    ctx = make_ctx(cash=500.0)
    result = check_no_margin(order, ctx, price=150.0, allow_margin=True)
    assert result.approved


# --- trade notional cap -------------------------------------------------------


def test_trade_notional_within_limit():
    order = make_order(quantity=10)  # notional 1500 at price 150
    ctx = make_ctx(start_of_run_nav=100_000.0)
    result = check_trade_notional(order, ctx, price=150.0, max_pct_of_nav=0.05)
    assert result.approved


def test_trade_notional_exceeds_limit():
    order = make_order(quantity=1000)  # notional 150_000, NAV-Limit 5% von 100k = 5000
    ctx = make_ctx(start_of_run_nav=100_000.0)
    result = check_trade_notional(order, ctx, price=150.0, max_pct_of_nav=0.05)
    assert not result.approved


def test_trade_notional_uses_fixed_start_of_run_nav_not_shrinking_cash():
    """Kap. 6.8: das Limit ist eine feste Groesse fuer den ganzen Lauf.
    Bereits stark geschrumpftes Cash (z.B. nach mehreren Buys in diesem
    Lauf) darf das Limit NICHT verschaerfen - massgeblich bleibt
    start_of_run_nav, nicht das aktuelle Cash."""
    order = make_order(quantity=10)  # notional 1500 at price 150
    ctx = make_ctx(cash=1_000.0, nav=100_000.0, start_of_run_nav=100_000.0)
    # Mit der alten cash-basierten Logik waere das Limit 5% von 1_000 = 50
    # gewesen - die Order (1500) waere faelschlich abgelehnt worden.
    result = check_trade_notional(order, ctx, price=150.0, max_pct_of_nav=0.05)
    assert result.approved


def test_trade_notional_falls_back_to_nav_without_start_of_run_nav():
    order = make_order(quantity=10)  # notional 1500 at price 150
    ctx = make_ctx(nav=100_000.0, start_of_run_nav=None)
    result = check_trade_notional(order, ctx, price=150.0, max_pct_of_nav=0.05)
    assert result.approved


# --- position size cap --------------------------------------------------------


def test_position_size_within_limit():
    order = make_order(quantity=10)  # 1500 notional, limit 10% of 100k nav = 10000
    ctx = make_ctx(nav=100_000.0)
    result = check_position_size(order, ctx, price=150.0, max_pct_of_portfolio=0.10)
    assert result.approved


def test_position_size_exceeds_limit_with_existing_position():
    existing = OpenPosition(symbol="AAPL", instrument_type="equity", side="long", quantity=60, avg_entry_price=150.0)
    order = make_order(quantity=10)  # existing 60 + 10 = 70 * 150 = 10500 > 10% of 100k
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_position_size(order, ctx, price=150.0, max_pct_of_portfolio=0.10)
    assert not result.approved


def test_position_size_check_ignored_for_sell():
    existing = OpenPosition(symbol="AAPL", instrument_type="equity", side="long", quantity=1000, avg_entry_price=150.0)
    order = make_order(quantity=10, side="sell")
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_position_size(order, ctx, price=150.0, max_pct_of_portfolio=0.10)
    assert result.approved


# --- structured products cap --------------------------------------------------


def test_structured_products_cap_within_limit():
    order = make_order(
        symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future",
        underlying_symbol="NVDA",
        side="buy",
        notional=5000,
    )
    ctx = make_ctx(nav=100_000.0)
    result = check_structured_products_cap(order, ctx, price=10.0, current_prices={}, max_pct_of_nav=0.20)
    assert result.approved


def test_structured_products_cap_exceeded():
    existing = OpenPosition(
        symbol="WARRANT-TSLA-PUT-1", instrument_type="warrant", side="long", quantity=1000, avg_entry_price=15.0
    )
    order = make_order(
        symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future",
        underlying_symbol="NVDA",
        side="buy",
        notional=10_000,
    )
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_structured_products_cap(
        order, ctx, price=10.0, current_prices={"WARRANT-TSLA-PUT-1": 15.0}, max_pct_of_nav=0.20
    )
    # existing 15_000 + new 10_000 = 25_000 > 20% of 100_000 = 20_000
    assert not result.approved


def test_structured_products_cap_not_applied_to_equity_orders():
    order = make_order(instrument_type="equity", side="buy", quantity=10)
    ctx = make_ctx(nav=100_000.0)
    result = check_structured_products_cap(order, ctx, price=150.0, current_prices={}, max_pct_of_nav=0.20)
    assert result.approved


# --- Kap. 6.8: Segmentgewicht --------------------------------------------------


def test_segment_weight_within_limit():
    existing = OpenPosition(
        symbol="NVDA", instrument_type="equity", side="long", quantity=100, avg_entry_price=200.0,
        segment="KI-Halbleiter & Compute", cap_tier="Mega-Cap",
    )
    order = make_order(symbol="AMD", quantity=50)  # 50*100=5_000; 20_000+5_000=25_000 <= 30% von 100k
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_segment_weight(
        order, ctx, price=100.0, current_prices={}, order_segment="KI-Halbleiter & Compute", max_pct_of_nav=0.30
    )
    assert result.approved


def test_segment_weight_exceeds_limit():
    existing = OpenPosition(
        symbol="NVDA", instrument_type="equity", side="long", quantity=100, avg_entry_price=200.0,
        segment="KI-Halbleiter & Compute", cap_tier="Mega-Cap",
    )
    order = make_order(symbol="AMD", quantity=150)  # 15_000; 20_000+15_000=35_000 > 30% von 100k
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_segment_weight(
        order, ctx, price=100.0, current_prices={}, order_segment="KI-Halbleiter & Compute", max_pct_of_nav=0.30
    )
    assert not result.approved


def test_segment_weight_ignores_different_segment():
    existing = OpenPosition(
        symbol="NVDA", instrument_type="equity", side="long", quantity=100, avg_entry_price=200.0,
        segment="KI-Halbleiter & Compute", cap_tier="Mega-Cap",
    )
    order = make_order(symbol="IONQ", quantity=150)
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_segment_weight(
        order, ctx, price=100.0, current_prices={}, order_segment="Quantum Computing", max_pct_of_nav=0.30
    )
    assert result.approved


def test_segment_weight_noop_without_segment():
    order = make_order(quantity=1000)
    ctx = make_ctx(nav=100_000.0)
    result = check_segment_weight(order, ctx, price=100.0, current_prices={}, order_segment=None, max_pct_of_nav=0.30)
    assert result.approved


# --- Kap. 6.8: korrelierte Krypto-Mining-/Digital-Assets-Exposure --------------

CORRELATED_SEGMENTS = {"Krypto-Mining", "Digital Assets & Krypto-Oekosystem"}


def test_correlated_exposure_within_limit():
    existing = OpenPosition(
        symbol="MARA", instrument_type="equity", side="long", quantity=100, avg_entry_price=200.0,
        segment="Krypto-Mining", cap_tier="Mid-Cap",
    )
    order = make_order(symbol="COIN", quantity=100)  # 10_000; 20_000+10_000=30_000 == 30% von 100k
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_correlated_segment_exposure(
        order, ctx, price=100.0, current_prices={},
        order_segment="Digital Assets & Krypto-Oekosystem", correlated_segments=CORRELATED_SEGMENTS,
        max_pct_of_nav=0.30,
    )
    assert result.approved


def test_correlated_exposure_exceeds_limit():
    existing = OpenPosition(
        symbol="MARA", instrument_type="equity", side="long", quantity=100, avg_entry_price=200.0,
        segment="Krypto-Mining", cap_tier="Mid-Cap",
    )
    order = make_order(symbol="COIN", quantity=101)  # 10_100; 30_100 > 30_000
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_correlated_segment_exposure(
        order, ctx, price=100.0, current_prices={},
        order_segment="Digital Assets & Krypto-Oekosystem", correlated_segments=CORRELATED_SEGMENTS,
        max_pct_of_nav=0.30,
    )
    assert not result.approved


def test_correlated_exposure_ignores_unrelated_segment():
    order = make_order(symbol="CCJ", quantity=1000)
    ctx = make_ctx(nav=100_000.0)
    result = check_correlated_segment_exposure(
        order, ctx, price=100.0, current_prices={},
        order_segment="Nuklear & Uran", correlated_segments=CORRELATED_SEGMENTS, max_pct_of_nav=0.30,
    )
    assert result.approved


# --- Kap. 6.8: Micro-Cap-Sublimit -----------------------------------------------


def test_micro_cap_within_limit():
    existing = OpenPosition(
        symbol="QBTS", instrument_type="equity", side="long", quantity=1000, avg_entry_price=10.0,
        segment="Quantum Computing", cap_tier="Micro-Cap",
    )
    order = make_order(symbol="RGTI", quantity=400)  # 4_000; 10_000+4_000=14_000 <= 15% von 100k
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_micro_cap_exposure(
        order, ctx, price=10.0, current_prices={}, order_cap_tier="Micro/Small-Cap", max_pct_of_nav=0.15
    )
    assert result.approved


def test_micro_cap_exceeds_limit_with_mixed_tier():
    """"Micro/Small-Cap" (Mischform, z.B. RGTI in Anhang A) zählt ebenfalls als Micro-Cap."""
    existing = OpenPosition(
        symbol="QBTS", instrument_type="equity", side="long", quantity=1000, avg_entry_price=10.0,
        segment="Quantum Computing", cap_tier="Micro-Cap",
    )
    order = make_order(symbol="RGTI", quantity=600)  # 6_000; 16_000 > 15_000
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_micro_cap_exposure(
        order, ctx, price=10.0, current_prices={}, order_cap_tier="Micro/Small-Cap", max_pct_of_nav=0.15
    )
    assert not result.approved


def test_micro_cap_ignores_non_micro_order():
    order = make_order(symbol="NVDA", quantity=1000)
    ctx = make_ctx(nav=100_000.0)
    result = check_micro_cap_exposure(
        order, ctx, price=100.0, current_prices={}, order_cap_tier="Mega-Cap", max_pct_of_nav=0.15
    )
    assert result.approved


# --- Kap. 6.8: Top-3-Konzentration ----------------------------------------------


def test_top3_concentration_within_limit():
    existing = [
        OpenPosition(symbol="A", instrument_type="equity", side="long", quantity=200, avg_entry_price=100.0),
        OpenPosition(symbol="B", instrument_type="equity", side="long", quantity=50, avg_entry_price=100.0),
    ]
    order = make_order(symbol="C", quantity=50)  # 5_000; top3 = 20_000+5_000+5_000=30_000 <= 35% von 100k
    ctx = make_ctx(nav=100_000.0, positions=existing)
    result = check_top3_concentration(order, ctx, price=100.0, current_prices={}, max_pct_of_nav=0.35)
    assert result.approved


def test_top3_concentration_exceeds_limit():
    existing = [
        OpenPosition(symbol="A", instrument_type="equity", side="long", quantity=200, avg_entry_price=100.0),
        OpenPosition(symbol="B", instrument_type="equity", side="long", quantity=100, avg_entry_price=100.0),
    ]
    order = make_order(symbol="C", quantity=100)  # 10_000; top3 = 20_000+10_000+10_000=40_000 > 35_000
    ctx = make_ctx(nav=100_000.0, positions=existing)
    result = check_top3_concentration(order, ctx, price=100.0, current_prices={}, max_pct_of_nav=0.35)
    assert not result.approved


def test_top3_concentration_ignored_for_sell():
    existing = [OpenPosition(symbol="A", instrument_type="equity", side="long", quantity=1000, avg_entry_price=100.0)]
    order = make_order(symbol="A", side="sell", quantity=10)
    ctx = make_ctx(nav=100_000.0, positions=existing)
    result = check_top3_concentration(order, ctx, price=100.0, current_prices={}, max_pct_of_nav=0.35)
    assert result.approved


# --- Kap. 6.8: Mindest-Cash-Quote ------------------------------------------------


def test_min_cash_quota_within_limit():
    order = make_order(quantity=40)  # 40*100=4_000
    ctx = make_ctx(nav=100_000.0, cash=10_000.0)
    result = check_min_cash_quota(order, ctx, price=100.0, min_pct_of_nav=0.05)
    assert result.approved


def test_min_cash_quota_violated():
    order = make_order(quantity=55)  # 5_500; 10_000-5_500=4_500 < 5% von 100k = 5_000
    ctx = make_ctx(nav=100_000.0, cash=10_000.0)
    result = check_min_cash_quota(order, ctx, price=100.0, min_pct_of_nav=0.05)
    assert not result.approved


def test_min_cash_quota_ignored_for_short():
    """Short erhöht laut Cashflow-Modell das Cash (Verkaufserlös) - nicht betroffen."""
    order = make_order(side="short", quantity=1000)
    ctx = make_ctx(nav=100_000.0, cash=1_000.0)
    result = check_min_cash_quota(order, ctx, price=100.0, min_pct_of_nav=0.05)
    assert result.approved


# --- Kap. 6.8: Portfolio-Circuit-Breaker -----------------------------------------


def test_circuit_breaker_blocks_leveraged_position_after_drawdown():
    order = make_order(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA", side="buy", notional=1000
    )
    ctx = make_ctx(nav=74_000.0, peak_nav=100_000.0)  # -26% Drawdown
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25)
    assert not result.approved


def test_circuit_breaker_allows_leveraged_position_before_threshold():
    order = make_order(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA", side="buy", notional=1000
    )
    ctx = make_ctx(nav=76_000.0, peak_nav=100_000.0)  # -24% Drawdown
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25)
    assert result.approved


def test_circuit_breaker_ignores_equity_orders():
    order = make_order(instrument_type="equity", side="buy", quantity=10)
    ctx = make_ctx(nav=50_000.0, peak_nav=100_000.0)  # -50% Drawdown, betrifft aber nur Hebelpositionen
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25)
    assert result.approved


def test_circuit_breaker_noop_without_peak_nav():
    order = make_order(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA", side="buy", notional=1000
    )
    ctx = make_ctx(nav=1_000.0, peak_nav=None)
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25)
    assert result.approved


# --- daily loss stop -----------------------------------------------------------


def test_daily_loss_stop_not_triggered():
    ctx = make_ctx(nav=99_000.0, start_of_run_nav=100_000.0)  # -1%
    result = check_daily_loss_stop(ctx, daily_loss_stop_pct=-0.03)
    assert result.approved


def test_daily_loss_stop_triggered():
    ctx = make_ctx(nav=96_000.0, start_of_run_nav=100_000.0)  # -4%
    result = check_daily_loss_stop(ctx, daily_loss_stop_pct=-0.03)
    assert not result.approved


def test_daily_loss_stop_noop_without_baseline():
    ctx = make_ctx(nav=50_000.0, start_of_run_nav=None)
    result = check_daily_loss_stop(ctx, daily_loss_stop_pct=-0.03)
    assert result.approved


# --- evaluate_order (integration of all checks) --------------------------------


def test_evaluate_order_rejects_when_short_disallowed():
    config = make_config(allow_short=False)
    order = make_order(side="short", quantity=10)
    ctx = make_ctx()
    result = evaluate_order(order, ctx, price=150.0, current_prices={}, config=config)
    assert not result.approved
    assert "Short" in result.reasons[0]


def test_evaluate_order_rejects_when_daily_loss_stop_active():
    config = make_config()
    order = make_order(side="buy", quantity=1)
    ctx = make_ctx(nav=96_000.0, start_of_run_nav=100_000.0)
    result = evaluate_order(order, ctx, price=150.0, current_prices={}, config=config)
    assert not result.approved


def test_evaluate_order_approves_clean_order():
    config = make_config()
    order = make_order(side="buy", quantity=1)
    ctx = make_ctx()
    result = evaluate_order(order, ctx, price=150.0, current_prices={}, config=config)
    assert result.approved


def test_evaluate_order_rejects_when_segment_weight_exceeded():
    """Integrationstest: evaluate_order reicht order_segment tatsächlich an
    check_segment_weight durch. Andere Notional-/Konzentrationslimiten werden
    grosszügig überschrieben, um gezielt nur die Segment-Regel zu prüfen."""
    config = make_config(
        max_trade_notional_pct_of_nav=1.0,
        max_position_size_pct_of_portfolio=1.0,
        max_top3_concentration_pct_of_nav=1.0,
    )
    existing = OpenPosition(
        symbol="NVDA", instrument_type="equity", side="long", quantity=200, avg_entry_price=150.0,
        segment="KI-Halbleiter & Compute", cap_tier="Mega-Cap",
    )
    order = make_order(symbol="AMD", quantity=100)  # 15_000; 30_000+15_000=45_000 > 30% von 100k
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = evaluate_order(
        order, ctx, price=150.0, current_prices={}, config=config, order_segment="KI-Halbleiter & Compute"
    )
    assert not result.approved
    assert any("Segment" in r for r in result.reasons)


def test_evaluate_order_sell_bypasses_daily_loss_stop():
    """Reducing exposure must remain possible even when the loss-stop is active."""
    config = make_config()
    existing = OpenPosition(symbol="AAPL", instrument_type="equity", side="long", quantity=10, avg_entry_price=150.0)
    order = make_order(side="sell", quantity=5)
    ctx = make_ctx(nav=96_000.0, start_of_run_nav=100_000.0, positions=[existing])
    result = evaluate_order(order, ctx, price=150.0, current_prices={}, config=config)
    assert result.approved


# --- short stop-loss sweep -------------------------------------------------------


def test_short_stop_loss_triggers_forced_close():
    positions = [OpenPosition(symbol="TSLA", instrument_type="equity", side="short", quantity=5, avg_entry_price=200.0)]
    current_prices = {"TSLA": 245.0}  # +22.5% gegen die Short-Position
    forced = evaluate_short_positions_for_stop_loss(positions, current_prices, short_stop_loss_pct=-0.20)
    assert len(forced) == 1
    assert forced[0].symbol == "TSLA"
    assert forced[0].loss_pct == pytest.approx(0.225)
    assert "TSLA" in forced[0].documentation


def test_short_stop_loss_not_triggered_below_threshold():
    positions = [OpenPosition(symbol="TSLA", instrument_type="equity", side="short", quantity=5, avg_entry_price=200.0)]
    current_prices = {"TSLA": 210.0}  # +5%, unter Schwelle
    forced = evaluate_short_positions_for_stop_loss(positions, current_prices, short_stop_loss_pct=-0.20)
    assert forced == []


def test_short_stop_loss_ignores_long_positions():
    positions = [OpenPosition(symbol="AAPL", instrument_type="equity", side="long", quantity=5, avg_entry_price=100.0)]
    current_prices = {"AAPL": 500.0}
    forced = evaluate_short_positions_for_stop_loss(positions, current_prices, short_stop_loss_pct=-0.20)
    assert forced == []


# --- LOW-3: investment-universe allowlist (defense in depth) --------------------


def test_evaluate_order_rejects_symbol_outside_universe():
    config = make_config()
    order = make_order(symbol="GME", side="buy", quantity=1)
    ctx = make_ctx()
    result = evaluate_order(
        order, ctx, price=150.0, current_prices={}, config=config,
        universe_symbols={"AAPL", "NVDA"},
    )
    assert not result.approved
    assert "Universum" in result.reasons[0]


def test_evaluate_order_allows_symbol_in_universe():
    config = make_config()
    order = make_order(symbol="AAPL", side="buy", quantity=1)
    ctx = make_ctx()
    result = evaluate_order(
        order, ctx, price=150.0, current_prices={}, config=config,
        universe_symbols={"AAPL", "NVDA"},
    )
    assert result.approved


def test_evaluate_order_skips_universe_check_when_not_provided():
    """Backward compatible: no universe set -> gate is skipped."""
    config = make_config()
    order = make_order(symbol="ANYTHING", side="buy", quantity=1)
    ctx = make_ctx()
    result = evaluate_order(order, ctx, price=150.0, current_prices={}, config=config)
    assert result.approved


# --- MED-2: leverage controls cover flagged leveraged ETFs (NVDL/TSDD) ----------


def test_structured_products_cap_applies_to_flagged_leveraged_etf():
    """A leveraged ETF (instrument_type 'etf', leveraged=True) must count toward
    the leverage cap even though it is not a tagged structured product."""
    existing = OpenPosition(
        symbol="NVDL", instrument_type="etf", side="long", quantity=1000,
        avg_entry_price=15.0, leveraged=True,
    )
    order = make_order(symbol="TSDD", instrument_type="etf", side="buy", notional=10_000)
    ctx = make_ctx(nav=100_000.0, positions=[existing])
    result = check_structured_products_cap(
        order, ctx, price=10.0, current_prices={"NVDL": 15.0}, max_pct_of_nav=0.20,
        order_leveraged=True,
    )
    # existing 15_000 + new 10_000 = 25_000 > 20% of 100_000
    assert not result.approved


def test_structured_products_cap_ignores_plain_etf():
    order = make_order(symbol="SPY", instrument_type="etf", side="buy", notional=50_000)
    ctx = make_ctx(nav=100_000.0)
    result = check_structured_products_cap(
        order, ctx, price=500.0, current_prices={}, max_pct_of_nav=0.20, order_leveraged=False
    )
    assert result.approved


def test_circuit_breaker_blocks_flagged_leveraged_etf_after_drawdown():
    order = make_order(symbol="NVDL", instrument_type="etf", side="buy", notional=1000)
    ctx = make_ctx(nav=74_000.0, peak_nav=100_000.0)  # -26% drawdown
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25, order_leveraged=True)
    assert not result.approved


def test_circuit_breaker_ignores_plain_etf_after_drawdown():
    order = make_order(symbol="SPY", instrument_type="etf", side="buy", quantity=10)
    ctx = make_ctx(nav=50_000.0, peak_nav=100_000.0)
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25, order_leveraged=False)
    assert result.approved


def test_evaluate_order_routes_leveraged_flag_to_circuit_breaker():
    """Integration: evaluate_order passes order_leveraged through so a flagged
    leveraged ETF is blocked by the circuit breaker after a drawdown."""
    config = make_config()
    order = make_order(symbol="NVDL", instrument_type="etf", side="buy", quantity=1)
    ctx = make_ctx(nav=74_000.0, start_of_run_nav=74_000.0, peak_nav=100_000.0)
    result = evaluate_order(
        order, ctx, price=15.0, current_prices={}, config=config, order_leveraged=True
    )
    assert not result.approved
    assert any("Circuit-Breaker" in r for r in result.reasons)
