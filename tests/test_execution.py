"""Tests for src/execution.py's fixed transaction-cost deduction (2026-09-17,
risk_config.yaml's transaction_cost_pct_of_notional) - verifies the cash
impact on both the normal order path and the forced short-stop-loss path,
and that trades.transaction_cost is recorded. Uses the real db/schema.sql
against an in-memory SQLite DB (same pattern as test_db.py) and the
deterministic simulated-fill path for structured products, so no fake
Alpaca broker client is needed.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import db, execution
from src.config import RiskConfig
from src.market_phase import MarketPhase, MarketPhaseClassification
from src.order_schema import ProposedOrder
from src.position_sizing import VolatilityScaling
from src.risk_guardrails import ForcedStopLossAction

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def make_portfolio(conn: sqlite3.Connection, name: str = "test", initial_cash: float = 100_000.0) -> sqlite3.Row:
    conn.execute(
        "INSERT INTO portfolios (name, currency, initial_cash_balance, cash_balance) VALUES (?, ?, ?, ?)",
        (name, "USD", initial_cash, initial_cash),
    )
    conn.commit()
    return db.get_portfolio(conn, name)


def make_risk_config(**overrides) -> RiskConfig:
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
        correlated_crypto_mining_segments=[],
        max_correlated_crypto_mining_pct_of_nav=0.30,
        max_micro_cap_pct_of_nav=0.15,
        max_top3_concentration_pct_of_nav=0.35,
        min_cash_pct_of_nav=0.05,
        circuit_breaker_drawdown_pct=-0.25,
        transaction_cost_pct_of_notional=0.001,
        volatility_scaling_min_factor=0.5,
        volatility_scaling_max_factor=1.5,
        drawdown_tier1_pct=-0.10,
        drawdown_tier1_position_size_factor=0.75,
        drawdown_tier2_pct=-0.15,
        drawdown_tier2_position_size_factor=0.50,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


# --- execute_proposed_orders: buy --------------------------------------------


def test_execute_proposed_orders_deducts_transaction_cost_on_buy():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.001)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future",
        underlying_symbol="NVDA",
        side="buy",
        notional=4_000.0,
        rationale="test",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,  # unbenutzt fuer strukturierte Produkte (simulierter Fill)
    )
    assert results[0].approved

    updated = db.get_portfolio(conn, "test")
    expected_cost = 4_000.0 * 0.001
    assert updated["cash_balance"] == pytest.approx(100_000.0 - 4_000.0 - expected_cost)

    trade = conn.execute("SELECT * FROM trades WHERE portfolio_id = ?", (portfolio["id"],)).fetchone()
    assert trade["transaction_cost"] == pytest.approx(expected_cost)
    assert trade["notional"] == pytest.approx(4_000.0)  # Kosten fliessen NICHT ins notional ein


def test_execute_proposed_orders_zero_cost_config_deducts_nothing():
    """Der Cash-Effekt muss exakt 0 sein, nicht nur 'klein', wenn die
    Kosten-Annahme auf 0 gesetzt ist - Regressionsschutz gegen einen
    Rundungs-/Vorzeichenfehler in der neuen Abzugslogik."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future",
        underlying_symbol="NVDA",
        side="buy",
        notional=4_000.0,
        rationale="test",
    )

    execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    updated = db.get_portfolio(conn, "test")
    assert updated["cash_balance"] == pytest.approx(100_000.0 - 4_000.0)


# --- execute_proposed_orders: sell (cost must reduce proceeds, not add to them) --


def test_execute_proposed_orders_deducts_transaction_cost_on_sell():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.001)

    # Bestehende Long-Position direkt anlegen - der Kauf ist nicht Teil dieses Tests.
    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future", underlying_symbol="NVDA", side="long",
        delta_quantity=40.0, fill_price=100.0,
    )

    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="sell", quantity=40.0, rationale="test",
    )
    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )
    assert results[0].approved

    updated = db.get_portfolio(conn, "test")
    notional = 40.0 * 100.0
    expected_cost = notional * 0.001
    # Erloes wird durch die Kosten GESCHMÄLERT, nicht erhoeht - Cash darf nicht
    # ueber (Ausgangs-Cash + voller Notional) hinausgehen.
    assert updated["cash_balance"] == pytest.approx(100_000.0 + notional - expected_cost)


# --- execute_proposed_orders: volatilitätsadjustierte Positionsgrössen-Skalierung ---


def test_execute_proposed_orders_scales_notional_before_guardrail_check():
    """Faktor 1.5 muss VOR der Guardrail-Pruefung greifen: das tatsaechlich
    gebuchte Notional ist das skalierte, nicht das von Claude vorgeschlagene."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    # max_trade_notional_pct_of_nav angehoben (0.10 statt Default 0.05), damit
    # dieser Test ausschliesslich die Skalierung selbst prueft - das
    # Zusammenspiel mit einem durch die Skalierung ueberschrittenen Limit
    # deckt der eigene Test unten ab.
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0, max_trade_notional_pct_of_nav=0.10)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test",
    )
    scaling = VolatilityScaling(
        symbol="MINI-NVDA-LONG-1", annualized_volatility=0.10, universe_avg_volatility=0.15, scaling_factor=1.5,
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        volatility_scaling={"MINI-NVDA-LONG-1": scaling},
    )

    assert results[0].approved
    assert results[0].volatility_scaling == scaling
    assert results[0].order.notional == pytest.approx(6_000.0)  # 4000 * 1.5

    trade = conn.execute("SELECT * FROM trades WHERE portfolio_id = ?", (portfolio["id"],)).fetchone()
    assert trade["notional"] == pytest.approx(6_000.0)
    updated = db.get_portfolio(conn, "test")
    assert updated["cash_balance"] == pytest.approx(100_000.0 - 6_000.0)


def test_execute_proposed_orders_scaling_cannot_bypass_trade_notional_limit():
    """Die Skalierung ist eine Verfeinerung INNERHALB der Kap.-6.8-Limiten,
    kein Aushebeln: eine durch die Skalierung ueber das Limit gehobene Order
    muss weiterhin abgelehnt werden, obwohl die urspruengliche (unskalierte)
    Groesse innerhalb des Limits gelegen haette."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    # Limit = 5% von 100'000 = 5'000 - das unskalierte Notional (4'000) liegt
    # darunter, das skalierte (4'000 * 1.5 = 6'000) darueber.
    risk_config = make_risk_config(max_trade_notional_pct_of_nav=0.05)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test",
    )
    scaling = VolatilityScaling(
        symbol="MINI-NVDA-LONG-1", annualized_volatility=0.10, universe_avg_volatility=0.15, scaling_factor=1.5,
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        volatility_scaling={"MINI-NVDA-LONG-1": scaling},
    )

    assert not results[0].approved
    assert results[0].volatility_scaling == scaling
    assert any("Trade-Notional" in r for r in results[0].reasons)


def test_execute_proposed_orders_sell_orders_are_not_scaled():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future", underlying_symbol="NVDA", side="long",
        delta_quantity=40.0, fill_price=100.0,
    )
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="sell", quantity=40.0, rationale="test",
    )
    scaling = VolatilityScaling(
        symbol="MINI-NVDA-LONG-1", annualized_volatility=0.10, universe_avg_volatility=0.15, scaling_factor=1.5,
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        volatility_scaling={"MINI-NVDA-LONG-1": scaling},
    )

    assert results[0].approved
    assert results[0].volatility_scaling is None
    assert results[0].order.quantity == pytest.approx(40.0)  # unveraendert


def test_execute_proposed_orders_scales_notional_by_conviction():
    """Konviktion "high" muss die Positionsgroesse leicht erhoehen (Faktor
    1.15, siehe position_sizing.CONVICTION_SCALING_FACTORS), auch ganz ohne
    Volatilitaetsdaten fuer das Symbol."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", conviction="high",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    assert results[0].approved
    assert results[0].conviction_scaling_factor == pytest.approx(1.15)
    assert results[0].order.notional == pytest.approx(4_600.0)  # 4000 * 1.15


def test_execute_proposed_orders_low_conviction_scales_down():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", conviction="low",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    assert results[0].approved
    assert results[0].conviction_scaling_factor == pytest.approx(0.8)
    assert results[0].order.notional == pytest.approx(3_200.0)  # 4000 * 0.8


def test_execute_proposed_orders_no_conviction_is_neutral():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test",  # kein conviction-Feld
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    assert results[0].approved
    assert results[0].conviction_scaling_factor == pytest.approx(1.0)
    assert results[0].order.notional == pytest.approx(4_000.0)


def test_execute_proposed_orders_combines_volatility_and_conviction_multiplicatively():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0, max_trade_notional_pct_of_nav=0.20)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", conviction="high",
    )
    scaling = VolatilityScaling(
        symbol="MINI-NVDA-LONG-1", annualized_volatility=0.10, universe_avg_volatility=0.15, scaling_factor=1.5,
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        volatility_scaling={"MINI-NVDA-LONG-1": scaling},
    )

    assert results[0].approved
    assert results[0].volatility_scaling == scaling
    assert results[0].conviction_scaling_factor == pytest.approx(1.15)
    # 4000 * 1.5 (Vola) * 1.15 (Konviktion) = 6900
    assert results[0].order.notional == pytest.approx(6_900.0)


def test_execute_proposed_orders_conviction_scaling_cannot_bypass_guardrail():
    """Wie bei der Volatilitaets-Skalierung: die Konviktions-Skalierung ist
    eine Verfeinerung INNERHALB der Kap.-6.8-Limiten, kein Aushebeln - eine
    durch "high" ueber das Limit gehobene Order muss weiterhin abgelehnt
    werden."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    # Limit = 5% von 100'000 = 4'600 - das unskalierte Notional (4'000) liegt
    # knapp darunter, das konviktions-skalierte (4'000 * 1.15 = 4'600) genau
    # an der Grenze; 4'050 * 1.15 = 4'657.50 liegt klar darueber.
    risk_config = make_risk_config(max_trade_notional_pct_of_nav=0.046)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_050.0, rationale="test", conviction="high",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    assert not results[0].approved
    assert results[0].conviction_scaling_factor == pytest.approx(1.15)
    assert any("Trade-Notional" in r for r in results[0].reasons)


def test_execute_proposed_orders_sell_orders_are_not_conviction_scaled():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future", underlying_symbol="NVDA", side="long",
        delta_quantity=40.0, fill_price=100.0,
    )
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="sell", quantity=40.0, rationale="test", conviction="high",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    assert results[0].approved
    assert results[0].conviction_scaling_factor is None
    assert results[0].order.quantity == pytest.approx(40.0)


def test_execute_proposed_orders_no_scaling_data_leaves_order_unchanged():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        volatility_scaling={},  # kein Eintrag fuer dieses Symbol
    )

    assert results[0].approved
    assert results[0].volatility_scaling is None
    assert results[0].order.notional == pytest.approx(4_000.0)


# --- execute_proposed_orders: Markt-Phasen-Abgleich (2026-09-20) -------------


def make_market_phase(symbol: str, phase: MarketPhase) -> MarketPhaseClassification:
    return MarketPhaseClassification(
        symbol=symbol, phase=phase, price=100.0, sma20=100.0, sma50=100.0, volatility_20d_annualized=0.2
    )


def test_execute_proposed_orders_documents_market_phase_contradiction():
    """Kein Veto: eine widersprechende Zyklus-Position/Marktphase-Kombination
    (hier: 'mania' bei tatsaechlich regelbasiert erkanntem Bear-Markt) darf
    die Order weder blockieren noch veraendern - nur dokumentieren."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", cycle_position="mania",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        market_phases={"MINI-NVDA-LONG-1": make_market_phase("MINI-NVDA-LONG-1", MarketPhase.BEAR)},
    )

    assert results[0].approved  # kein Veto
    contradiction = results[0].market_phase_contradiction
    assert contradiction is not None
    assert contradiction.claude_cycle_position.value == "mania"
    assert contradiction.rule_based_phase == MarketPhase.BEAR


def test_execute_proposed_orders_no_contradiction_when_phases_align():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", cycle_position="mania",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        market_phases={"MINI-NVDA-LONG-1": make_market_phase("MINI-NVDA-LONG-1", MarketPhase.BULL)},
    )

    assert results[0].approved
    assert results[0].market_phase_contradiction is None


def test_execute_proposed_orders_no_market_phase_check_without_cycle_position():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test",  # kein cycle_position
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        market_phases={"MINI-NVDA-LONG-1": make_market_phase("MINI-NVDA-LONG-1", MarketPhase.BEAR)},
    )

    assert results[0].market_phase_contradiction is None


def test_execute_proposed_orders_no_market_phase_check_without_phase_data():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config(transaction_cost_pct_of_notional=0.0)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", cycle_position="crash",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        market_phases=None,  # keine Klassifikation verfuegbar
    )

    assert results[0].market_phase_contradiction is None


def test_execute_proposed_orders_documents_contradiction_even_when_order_rejected():
    """Auch eine von den Guardrails abgelehnte Order dokumentiert einen
    gefundenen Markt-Phasen-Widerspruch - die beiden Mechanismen sind
    unabhaengig voneinander."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    # Sehr niedriges Limit, damit die Order sicher abgelehnt wird.
    risk_config = make_risk_config(max_trade_notional_pct_of_nav=0.001)
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test", cycle_position="crash",
    )

    results = execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
        market_phases={"MINI-NVDA-LONG-1": make_market_phase("MINI-NVDA-LONG-1", MarketPhase.BULL)},
    )

    assert not results[0].approved
    contradiction = results[0].market_phase_contradiction
    assert contradiction is not None
    assert contradiction.claude_cycle_position.value == "crash"


# --- execute_forced_stop_loss_actions ----------------------------------------


class ImmediateFillClient:
    """Minimal stand-in for alpaca.trading.client.TradingClient - the cover
    is already 'filled' in submit_order()'s own response, so
    _poll_until_settled's early-return path is taken and get_order_by_id/
    cancel_order_by_id are never called (same case as
    test_submit_equity_order_already_filled_on_submit_needs_no_poll in
    test_broker_alpaca.py)."""

    def __init__(self, filled_qty: float, filled_avg_price: float):
        self._response = SimpleNamespace(
            id="o1",
            status=SimpleNamespace(value="filled"),
            filled_qty=str(filled_qty),
            filled_avg_price=str(filled_avg_price),
        )

    def submit_order(self, request):
        return self._response


def test_execute_forced_stop_loss_actions_deducts_transaction_cost():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)

    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="TSLA",
        instrument_type="equity", underlying_symbol=None, side="short",
        delta_quantity=10.0, fill_price=100.0,
    )
    action = ForcedStopLossAction(
        symbol="TSLA", instrument_type="equity", quantity=10.0,
        entry_price=100.0, current_price=130.0, loss_pct=0.30, documentation="test",
    )
    broker_client = ImmediateFillClient(filled_qty=10.0, filled_avg_price=130.0)

    execution.execute_forced_stop_loss_actions(
        conn, portfolio, [action], broker_client=broker_client, transaction_cost_pct=0.001,
    )

    updated = db.get_portfolio(conn, "test")
    notional = 10.0 * 130.0
    expected_cost = notional * 0.001
    assert updated["cash_balance"] == pytest.approx(100_000.0 - notional - expected_cost)

    trade = conn.execute("SELECT * FROM trades WHERE portfolio_id = ?", (portfolio["id"],)).fetchone()
    assert trade["transaction_cost"] == pytest.approx(expected_cost)


def test_execute_forced_stop_loss_actions_does_not_crash_on_structured_product():
    """Regressionstest: ein Short-Stop-Loss auf ein strukturiertes Produkt
    (leverage_certificate/mini_future/warrant) durfte bislang nicht ausgeloest
    werden, ohne die Pipeline abstuerzen zu lassen - execute_forced_stop_loss_
    actions baute intern einen (ungenutzten) ProposedOrder ohne
    underlying_symbol, das strukturierte Produkte laut Pydantic-Validierung
    zwingend brauchen. Seit PR #3 stehen NVDL/TSDD unter den Hebel-Guardrails
    und koennten geshortet werden - real erreichbar, nicht nur theoretisch,
    auch wenn dieser konkrete Crash-Pfad echte strukturierte Produkte
    (mini_future/leverage_certificate/warrant) statt der als 'etf' getaggten
    NVDL/TSDD betrifft. Fix: der tote ProposedOrder-Codepfad wurde entfernt."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)

    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="MINI-TSLA-SHORT-1",
        instrument_type="mini_future", underlying_symbol="TSLA", side="short",
        delta_quantity=10.0, fill_price=100.0,
    )
    action = ForcedStopLossAction(
        symbol="MINI-TSLA-SHORT-1", instrument_type="mini_future", quantity=10.0,
        entry_price=100.0, current_price=130.0, loss_pct=0.30, documentation="test",
    )

    # Darf keine ValidationError (oder sonstige Exception) werfen:
    trade_ids = execution.execute_forced_stop_loss_actions(
        conn, portfolio, [action], broker_client=None, transaction_cost_pct=0.001,
    )
    assert len(trade_ids) == 1

    updated = db.get_portfolio(conn, "test")
    notional = 10.0 * 130.0
    expected_cost = notional * 0.001
    assert updated["cash_balance"] == pytest.approx(100_000.0 - notional - expected_cost)

    position = conn.execute(
        "SELECT * FROM positions WHERE portfolio_id = ? AND symbol = ?", (portfolio["id"], "MINI-TSLA-SHORT-1")
    ).fetchone()
    assert position["status"] == "closed"
    assert position["closure_reason"] == "short_stop_loss_forced"


# --- boundary_conditions (Kap. 7, 2026-09-17) --------------------------------


def test_execute_proposed_orders_stores_boundary_conditions_on_buy():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config()
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future",
        underlying_symbol="NVDA",
        side="buy",
        notional=4_000.0,
        rationale="test",
        boundary_conditions=[
            {"description": "Fällt unter $150", "check_type": "price_below", "threshold_price": 150.0},
            {"description": "Fed pausiert Zinssenkungen", "check_type": "qualitative"},
        ],
    )

    execution.execute_proposed_orders(
        conn, portfolio, [order],
        model="test", prompt="p", raw_response="r",
        risk_config=risk_config,
        current_prices={"NVDA": 100.0},
        start_of_run_nav=100_000.0,
        broker_client=None,
    )

    rows = conn.execute(
        "SELECT * FROM boundary_conditions WHERE portfolio_id = ? ORDER BY id", (portfolio["id"],)
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["check_type"] == "price_below"
    assert rows[0]["threshold_price"] == 150.0
    assert rows[0]["status"] == "open"
    assert rows[1]["check_type"] == "qualitative"
    assert rows[1]["threshold_price"] is None
    # Beide an dieselbe (neu eroeffnete) Position gebunden.
    assert rows[0]["position_id"] == rows[1]["position_id"]


def test_execute_proposed_orders_without_boundary_conditions_stores_nothing():
    """Optional/best-effort: eine Order ohne genannte Randbedingung darf
    keine Zeilen erzeugen und keinen Fehler verursachen."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config()
    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1", instrument_type="mini_future", underlying_symbol="NVDA",
        side="buy", notional=4_000.0, rationale="test",
    )

    execution.execute_proposed_orders(
        conn, portfolio, [order], model="test", prompt="p", raw_response="r",
        risk_config=risk_config, current_prices={"NVDA": 100.0}, start_of_run_nav=100_000.0,
        broker_client=None,
    )

    rows = conn.execute(
        "SELECT * FROM boundary_conditions WHERE portfolio_id = ?", (portfolio["id"],)
    ).fetchall()
    assert rows == []


# --- correlation (rein dokumentarisch, kein Veto) ----------------------------


def _correlation_matrix(pairs: dict[tuple[str, str], float], symbols: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(1.0, index=symbols, columns=symbols)
    for (a, b), value in pairs.items():
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value
    return matrix


def test_execute_proposed_orders_records_correlation_warning_without_veto():
    """Kernanforderung: eine hohe Korrelation zu einer bestehenden Position
    wird dokumentiert, blockiert die BUY-Order aber NICHT (kein Guardrail-
    Veto, anders als die harten Kap.-6.8-Limiten)."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config()

    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="AMD", instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10.0, fill_price=100.0,
    )
    correlation_matrix = _correlation_matrix({("NVDA", "AMD"): 0.92}, symbols=["NVDA", "AMD"])

    order = ProposedOrder(symbol="NVDA", instrument_type="equity", side="buy", notional=4_000.0, rationale="test")
    results = execution.execute_proposed_orders(
        conn, portfolio, [order], model="test", prompt="p", raw_response="r",
        risk_config=risk_config, current_prices={"NVDA": 100.0, "AMD": 100.0}, start_of_run_nav=100_000.0,
        broker_client=ImmediateFillClient(filled_qty=40.0, filled_avg_price=100.0),
        correlation_matrix=correlation_matrix,
    )

    assert results[0].approved is True  # kein Veto trotz hoher Korrelation
    assert len(results[0].correlation_warnings) == 1
    warning = results[0].correlation_warnings[0]
    assert warning.candidate_symbol == "NVDA"
    assert warning.existing_symbol == "AMD"
    assert warning.correlation == pytest.approx(0.92)


def test_execute_proposed_orders_no_warning_below_threshold():
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config()

    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="AMD", instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10.0, fill_price=100.0,
    )
    correlation_matrix = _correlation_matrix({("NVDA", "AMD"): 0.5}, symbols=["NVDA", "AMD"])

    order = ProposedOrder(symbol="NVDA", instrument_type="equity", side="buy", notional=4_000.0, rationale="test")
    results = execution.execute_proposed_orders(
        conn, portfolio, [order], model="test", prompt="p", raw_response="r",
        risk_config=risk_config, current_prices={"NVDA": 100.0, "AMD": 100.0}, start_of_run_nav=100_000.0,
        broker_client=ImmediateFillClient(filled_qty=40.0, filled_avg_price=100.0),
        correlation_matrix=correlation_matrix,
    )
    assert results[0].correlation_warnings == []


def test_execute_proposed_orders_no_correlation_check_for_sell():
    """Die Pruefung ist explizit auf BUY-Orders beschraenkt."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config()

    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="NVDA", instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10.0, fill_price=100.0,
    )
    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="AMD", instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10.0, fill_price=100.0,
    )
    correlation_matrix = _correlation_matrix({("NVDA", "AMD"): 0.95}, symbols=["NVDA", "AMD"])

    order = ProposedOrder(symbol="NVDA", instrument_type="equity", side="sell", quantity=10.0, rationale="test")
    results = execution.execute_proposed_orders(
        conn, portfolio, [order], model="test", prompt="p", raw_response="r",
        risk_config=risk_config, current_prices={"NVDA": 100.0, "AMD": 100.0}, start_of_run_nav=100_000.0,
        broker_client=ImmediateFillClient(filled_qty=10.0, filled_avg_price=100.0),
        correlation_matrix=correlation_matrix,
    )
    assert results[0].correlation_warnings == []


def test_execute_proposed_orders_without_correlation_matrix_is_noop():
    """`correlation_matrix=None` (Default) darf nicht crashen - z.B. wenn
    die Matrix-Berechnung fehlgeschlagen ist (siehe pipeline.py)."""
    conn = make_conn()
    portfolio = make_portfolio(conn, initial_cash=100_000.0)
    risk_config = make_risk_config()
    order = ProposedOrder(symbol="NVDA", instrument_type="equity", side="buy", notional=4_000.0, rationale="test")

    results = execution.execute_proposed_orders(
        conn, portfolio, [order], model="test", prompt="p", raw_response="r",
        risk_config=risk_config, current_prices={"NVDA": 100.0}, start_of_run_nav=100_000.0,
        broker_client=ImmediateFillClient(filled_qty=40.0, filled_avg_price=100.0),
    )
    assert results[0].correlation_warnings == []
