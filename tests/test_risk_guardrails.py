import pytest

from src.config import RiskConfig
from src.order_schema import ProposedOrder
from src.risk_guardrails import (
    OpenPosition,
    PortfolioContext,
    check_daily_loss_stop,
    check_max_trades_per_symbol,
    check_no_margin,
    check_position_size,
    check_structured_products_cap,
    check_trade_notional,
    evaluate_order,
    evaluate_short_positions_for_stop_loss,
)


def make_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_position_size_pct_of_portfolio=0.10,
        max_trade_notional_pct_of_cash=0.05,
        daily_loss_stop_pct=-0.03,
        max_trades_per_symbol_per_day=1,
        allow_short=True,
        allow_structured_products=True,
        structured_products_max_notional_pct_of_nav=0.20,
        short_stop_loss_pct=-0.20,
        allow_margin=False,
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
    ctx = make_ctx(cash=100_000.0)
    result = check_trade_notional(order, ctx, price=150.0, max_pct_of_cash=0.05)
    assert result.approved


def test_trade_notional_exceeds_limit():
    order = make_order(quantity=1000)  # notional 150_000, cash limit 5% of 100k = 5000
    ctx = make_ctx(cash=100_000.0)
    result = check_trade_notional(order, ctx, price=150.0, max_pct_of_cash=0.05)
    assert not result.approved


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
