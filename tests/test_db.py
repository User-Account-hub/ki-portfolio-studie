"""Tests for src/db.py's nav_history support (migration + peak-NAV lookup).

Uses an in-memory SQLite DB built from the real db/schema.sql, so these
tests exercise the actual schema/FK constraints rather than a hand-rolled
stand-in.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src import db
from src.order_schema import ProposedOrder
from src.risk_guardrails import PortfolioContext, check_circuit_breaker

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def make_portfolio(conn: sqlite3.Connection, name: str = "test", initial_cash: float = 100_000.0) -> int:
    cur = conn.execute(
        "INSERT INTO portfolios (name, currency, initial_cash_balance, cash_balance) VALUES (?, ?, ?, ?)",
        (name, "USD", initial_cash, initial_cash),
    )
    conn.commit()
    return cur.lastrowid


# --- ensure_nav_history_table (Migration) --------------------------------------


def test_ensure_nav_history_table_is_idempotent():
    """schema.sql legt die Tabelle schon an; ein erneuter Aufruf (Migration
    fuer eine bereits deployte DB) darf nicht crashen und nichts zerstoeren."""
    conn = make_conn()
    db.ensure_nav_history_table(conn)
    db.ensure_nav_history_table(conn)

    portfolio_id = make_portfolio(conn)
    db.record_nav(conn, portfolio_id, 100_000.0)
    assert db.get_peak_nav(conn, portfolio_id) == 100_000.0


def test_ensure_nav_history_table_on_db_without_it():
    """Simuliert eine bereits deployte DB ohne nav_history (aeltere schema.sql-
    Version): Tabelle fehlt zunaechst komplett, muss aber sauber nachgezogen
    werden, ohne bestehende Daten (hier: das Portfolio) anzutasten."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    old_schema = SCHEMA_PATH.read_text(encoding="utf-8")
    # Nur den nav_history-Block herausschneiden, um den alten Zustand nachzustellen.
    without_nav_history = old_schema.split("CREATE TABLE IF NOT EXISTS nav_history")[0]
    conn.executescript(without_nav_history)

    with pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT * FROM nav_history").fetchone()

    portfolio_id = make_portfolio(conn, initial_cash=250_000.0)

    db.ensure_nav_history_table(conn)  # Migration

    # Bestehendes Portfolio unveraendert:
    row = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
    assert row["cash_balance"] == 250_000.0

    # Tabelle jetzt nutzbar:
    db.record_nav(conn, portfolio_id, 250_000.0)
    assert db.get_peak_nav(conn, portfolio_id) == 250_000.0


# --- get_peak_nav ----------------------------------------------------------------


def test_get_peak_nav_returns_none_without_history():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    assert db.get_peak_nav(conn, portfolio_id) is None


def test_get_peak_nav_scoped_per_portfolio():
    conn = make_conn()
    p1 = make_portfolio(conn, name="p1")
    p2 = make_portfolio(conn, name="p2")
    db.record_nav(conn, p1, 200_000.0)
    db.record_nav(conn, p2, 120_000.0)
    assert db.get_peak_nav(conn, p1) == 200_000.0
    assert db.get_peak_nav(conn, p2) == 120_000.0


def test_get_peak_nav_tracks_true_historical_high_not_just_latest_two_values():
    """Kernszenario: NAV steigt zwischenzeitlich stark an (Lauf 2), faellt
    danach wieder, bleibt aber ueber dem Startkapital (Lauf 3). Der Peak muss
    weiterhin das Zwischenhoch aus Lauf 2 sein.

    Die vorherige Naeherung `max(initial_cash_balance, start_of_run_nav)`
    haette in Lauf 3 faelschlich max(100_000, 110_000) = 110_000 ergeben -
    exakt der aktuelle Stand, also 0% Drawdown - und den Circuit-Breaker nie
    ausgeloest, obwohl der reale Drawdown vom echten Zwischenhoch (150_000)
    klar ueber der -25%-Schwelle liegt.
    """
    conn = make_conn()
    portfolio_id = make_portfolio(conn, initial_cash=100_000.0)

    db.record_nav(conn, portfolio_id, 100_000.0)  # Lauf 1: Start
    db.record_nav(conn, portfolio_id, 150_000.0)  # Lauf 2: Rally (Zwischenhoch)
    db.record_nav(conn, portfolio_id, 110_000.0)  # Lauf 3: Ruecksetzer, aber > Startkapital

    peak = db.get_peak_nav(conn, portfolio_id)
    assert peak == 150_000.0  # nicht 110_000 (alte Naeherung) und nicht 100_000 (Startkapital)

    current_nav = 110_000.0
    real_drawdown = (current_nav - peak) / peak
    naive_drawdown = (current_nav - max(100_000.0, current_nav)) / max(100_000.0, current_nav)

    assert real_drawdown == pytest.approx(-4 / 15)  # ~ -26.7%
    assert real_drawdown < -0.25
    assert naive_drawdown == 0.0  # zeigt konkret, was die alte Naeherung verpasst haette

    order = ProposedOrder(
        symbol="MINI-NVDA-LONG-1",
        instrument_type="mini_future",
        underlying_symbol="NVDA",
        side="buy",
        notional=1_000,
        rationale="test",
    )
    ctx = PortfolioContext(nav=current_nav, cash=current_nav, peak_nav=peak)
    result = check_circuit_breaker(order, ctx, drawdown_pct=-0.25)
    assert not result.approved  # mit dem echten Peak loest der Breaker korrekt aus

    naive_ctx = PortfolioContext(nav=current_nav, cash=current_nav, peak_nav=max(100_000.0, current_nav))
    naive_result = check_circuit_breaker(order, naive_ctx, drawdown_pct=-0.25)
    assert naive_result.approved  # die alte Naeherung haette es faelschlich durchgelassen


# --- record_nav --------------------------------------------------------------------


def test_record_nav_appends_without_overwriting():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    db.record_nav(conn, portfolio_id, 100_000.0)
    db.record_nav(conn, portfolio_id, 105_000.0)
    rows = conn.execute(
        "SELECT nav FROM nav_history WHERE portfolio_id = ? ORDER BY id", (portfolio_id,)
    ).fetchall()
    assert [r["nav"] for r in rows] == [100_000.0, 105_000.0]
