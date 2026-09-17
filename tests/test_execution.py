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

import pytest

from src import db, execution
from src.config import RiskConfig
from src.order_schema import ProposedOrder
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
