"""Tests for src/db.py's nav_history support (migration + peak-NAV lookup).

Uses an in-memory SQLite DB built from the real db/schema.sql, so these
tests exercise the actual schema/FK constraints rather than a hand-rolled
stand-in.
"""
from __future__ import annotations

import datetime
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


# --- ensure_trade_transaction_cost_column (Migration, 2026-09-17) --------------


def test_ensure_trade_transaction_cost_column_is_idempotent():
    """schema.sql legt die Spalte schon an; ein erneuter Aufruf (Migration
    fuer eine bereits deployte DB) darf nicht mit "duplicate column" abstuerzen."""
    conn = make_conn()
    db.ensure_trade_transaction_cost_column(conn)
    db.ensure_trade_transaction_cost_column(conn)

    portfolio_id = make_portfolio(conn)
    trade_id = db.insert_trade(
        conn, portfolio_id=portfolio_id, position_id=None, decision_id=None,
        symbol="AAPL", instrument_type="equity", side="buy", quantity=1, price=100.0,
        order_type="market", broker_order_id=None, source="alpaca", status="filled",
    )
    row = conn.execute("SELECT transaction_cost FROM trades WHERE id = ?", (trade_id,)).fetchone()
    assert row["transaction_cost"] == 0.0  # Default, wenn nicht explizit uebergeben


def test_ensure_trade_transaction_cost_column_on_db_without_it():
    """Simuliert eine bereits deployte DB ohne die transaction_cost-Spalte
    (aeltere schema.sql-Version): Spalte fehlt zunaechst, muss aber sauber
    nachgezogen werden, ohne bestehende Trades anzutasten."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    old_schema = SCHEMA_PATH.read_text(encoding="utf-8")
    # Entfernt jede Zeile, die "transaction_cost" erwaehnt (Spaltendefinition
    # UND ihre Kommentarzeilen) - robuster als ein exaktes Text-Match gegen
    # den aktuellen Kommentarwortlaut.
    without_cost_column = "\n".join(
        line for line in old_schema.splitlines() if "transaction_cost" not in line
    )
    assert "transaction_cost" not in without_cost_column  # Sanity: Filterung hat gegriffen
    conn.executescript(without_cost_column)

    portfolio_id = make_portfolio(conn)

    db.ensure_trade_transaction_cost_column(conn)  # Migration

    # Spalte jetzt nutzbar, mit Default 0 fuer neue Inserts ohne expliziten Wert:
    trade_id = db.insert_trade(
        conn, portfolio_id=portfolio_id, position_id=None, decision_id=None,
        symbol="MSFT", instrument_type="equity", side="buy", quantity=1, price=200.0,
        order_type="market", broker_order_id=None, source="alpaca", status="filled",
    )
    row = conn.execute("SELECT transaction_cost FROM trades WHERE id = ?", (trade_id,)).fetchone()
    assert row["transaction_cost"] == 0.0


# --- insert_trade records transaction_cost -------------------------------------


# --- Monatliche Tiefenreflexion (Kap. 6.12.3, 2026-09-17) ----------------------


def test_get_decisions_since_excludes_earlier_and_reflection_decisions():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    conn.execute(
        "INSERT INTO decisions (portfolio_id, created_at, model, prompt) VALUES (?, ?, ?, ?)",
        (portfolio_id, "2026-09-20 15:00:00", "claude-sonnet-5", "vor der Periode"),
    )
    conn.execute(
        "INSERT INTO decisions (portfolio_id, created_at, model, prompt) VALUES (?, ?, ?, ?)",
        (portfolio_id, "2026-09-25 15:00:00", "claude-sonnet-5", "in der Periode"),
    )
    conn.execute(
        "INSERT INTO decisions (portfolio_id, created_at, model, prompt) VALUES (?, ?, ?, ?)",
        (portfolio_id, "2026-09-26 15:00:00", "deep_reflection", "eigene Reflexion, nicht relevant"),
    )
    conn.commit()

    rows = db.get_decisions_since(conn, portfolio_id, "2026-09-21 00:00:00")
    assert [r["prompt"] for r in rows] == ["in der Periode"]


def test_get_decisions_since_includes_same_day_as_period_start():
    """Regressionstest: `since` muss im selben Textformat wie `created_at`
    vorliegen (Leerzeichen, kein 'T') - ein ISO-Format mit 'T' (z.B. via
    pd.Timestamp.isoformat()) sortiert lexikographisch NACH einem
    Leerzeichen-Zeitstempel desselben Tages und wuerde Eintraege vom
    Perioden-Start-Tag selbst faelschlich ausschliessen."""
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    conn.execute(
        "INSERT INTO decisions (portfolio_id, created_at, model, prompt) VALUES (?, ?, ?, ?)",
        (portfolio_id, "2026-09-21 08:03:00", "claude-sonnet-5", "am Perioden-Start-Tag selbst"),
    )
    conn.commit()

    # "since" im korrekten SQLite-Format (Leerzeichen, Mitternacht) - so wie
    # pipeline.py es via period_start.strftime("%Y-%m-%d %H:%M:%S") erzeugt.
    rows = db.get_decisions_since(conn, portfolio_id, "2026-09-21 00:00:00")
    assert len(rows) == 1

    # Zum Vergleich: das fehleranfaellige ISO-Format mit "T" wuerde den
    # Eintrag verlieren (dokumentiert den Bug, den die Docstring-Warnung in
    # db.get_decisions_since verhindern soll).
    rows_with_iso_bug = db.get_decisions_since(conn, portfolio_id, "2026-09-21T00:00:00")
    assert rows_with_iso_bug == []


def test_get_trades_since_filters_by_date():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    db.insert_trade(
        conn, portfolio_id=portfolio_id, position_id=None, decision_id=None,
        symbol="AAPL", instrument_type="equity", side="buy", quantity=1, price=100.0,
        order_type="market", broker_order_id=None, source="alpaca", status="filled",
    )
    conn.execute("UPDATE trades SET executed_at = ? WHERE symbol = 'AAPL'", ("2026-09-15 10:00:00",))
    db.insert_trade(
        conn, portfolio_id=portfolio_id, position_id=None, decision_id=None,
        symbol="MSFT", instrument_type="equity", side="buy", quantity=1, price=200.0,
        order_type="market", broker_order_id=None, source="alpaca", status="filled",
    )
    conn.execute("UPDATE trades SET executed_at = ? WHERE symbol = 'MSFT'", ("2026-09-25 10:00:00",))
    conn.commit()

    rows = db.get_trades_since(conn, portfolio_id, "2026-09-21 00:00:00")
    assert [r["symbol"] for r in rows] == ["MSFT"]


def test_count_and_get_latest_reflection():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    assert db.count_deep_reflections(conn, portfolio_id) == 0
    assert db.get_latest_reflection(conn, portfolio_id) is None

    db.insert_decision(
        conn, portfolio_id=portfolio_id, model="deep_reflection", prompt="p1",
        raw_response='{"reflection_commentary": "erste"}', proposed_orders=None,
        risk_check_result=None, rationale="erste", forced_action=False, approved=True, executed=False,
    )
    db.insert_decision(
        conn, portfolio_id=portfolio_id, model="deep_reflection", prompt="p2",
        raw_response='{"reflection_commentary": "zweite"}', proposed_orders=None,
        risk_check_result=None, rationale="zweite", forced_action=False, approved=True, executed=False,
    )
    # Eine normale Tagesentscheidung darf nicht mitgezaehlt werden.
    db.insert_decision(
        conn, portfolio_id=portfolio_id, model="claude-sonnet-5", prompt="p3",
        raw_response='{"orders": []}', proposed_orders=[], risk_check_result=[],
        rationale=None, forced_action=False, approved=True, executed=False,
    )

    assert db.count_deep_reflections(conn, portfolio_id) == 2
    latest = db.get_latest_reflection(conn, portfolio_id)
    assert latest["rationale"] == "zweite"


def test_insert_trade_records_explicit_transaction_cost():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    trade_id = db.insert_trade(
        conn, portfolio_id=portfolio_id, position_id=None, decision_id=None,
        symbol="AAPL", instrument_type="equity", side="buy", quantity=10, price=150.0,
        order_type="market", broker_order_id=None, source="alpaca", status="filled",
        transaction_cost=1.5,
    )
    row = conn.execute("SELECT transaction_cost, notional FROM trades WHERE id = ?", (trade_id,)).fetchone()
    assert row["transaction_cost"] == 1.5
    assert row["notional"] == 1500.0  # unveraendert - Kosten fliessen NICHT ins notional ein


# --- Randbedingungs-Tracking (Kap. 7, 2026-09-17) -------------------------------


def make_open_position(conn: sqlite3.Connection, portfolio_id: int, symbol: str = "NVDA") -> int:
    return db.upsert_open_position(
        conn, portfolio_id=portfolio_id, symbol=symbol, instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10.0, fill_price=100.0,
    )


def test_ensure_boundary_conditions_table_is_idempotent():
    conn = make_conn()
    db.ensure_boundary_conditions_table(conn)
    db.ensure_boundary_conditions_table(conn)

    portfolio_id = make_portfolio(conn)
    position_id = make_open_position(conn, portfolio_id)
    cond_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="test", check_type="qualitative", threshold_price=None,
    )
    assert cond_id is not None


def test_ensure_boundary_conditions_table_on_db_without_it():
    """Simuliert eine bereits deployte DB ohne die boundary_conditions-
    Tabelle (aeltere schema.sql-Version): Tabelle fehlt zunaechst komplett,
    muss aber sauber nachgezogen werden, ohne bestehende Daten anzutasten."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    old_schema = SCHEMA_PATH.read_text(encoding="utf-8")
    without_table = old_schema.split("CREATE TABLE IF NOT EXISTS boundary_conditions")[0]
    conn.executescript(without_table)

    with pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT * FROM boundary_conditions").fetchone()

    portfolio_id = make_portfolio(conn, initial_cash=250_000.0)
    db.ensure_boundary_conditions_table(conn)  # Migration

    row = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
    assert row["cash_balance"] == 250_000.0  # unveraendert

    position_id = make_open_position(conn, portfolio_id)
    cond_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="test", check_type="qualitative", threshold_price=None,
    )
    assert cond_id is not None


def test_get_open_boundary_conditions_excludes_triggered_and_closed():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    position_id = make_open_position(conn, portfolio_id)
    open_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="offen", check_type="qualitative", threshold_price=None,
    )
    triggered_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="ausgeloest", check_type="price_below", threshold_price=100.0,
    )
    db.mark_boundary_conditions_triggered(conn, [triggered_id])

    rows = db.get_open_boundary_conditions(conn, portfolio_id)
    assert [r["id"] for r in rows] == [open_id]


def test_mark_boundary_conditions_triggered_sets_status_and_timestamp():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    position_id = make_open_position(conn, portfolio_id)
    cond_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="test", check_type="price_below", threshold_price=100.0,
    )
    db.mark_boundary_conditions_triggered(conn, [cond_id])

    row = conn.execute("SELECT * FROM boundary_conditions WHERE id = ?", (cond_id,)).fetchone()
    assert row["status"] == "triggered"
    assert row["triggered_at"] is not None


def test_mark_boundary_conditions_triggered_empty_list_is_noop():
    conn = make_conn()
    db.mark_boundary_conditions_triggered(conn, [])  # darf nicht crashen


def test_close_boundary_conditions_for_position_only_touches_open_ones():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    position_id = make_open_position(conn, portfolio_id)
    open_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="offen", check_type="qualitative", threshold_price=None,
    )
    triggered_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="ausgeloest", check_type="price_below", threshold_price=100.0,
    )
    db.mark_boundary_conditions_triggered(conn, [triggered_id])

    db.close_boundary_conditions_for_position(conn, position_id)

    open_row = conn.execute("SELECT status FROM boundary_conditions WHERE id = ?", (open_id,)).fetchone()
    triggered_row = conn.execute("SELECT status FROM boundary_conditions WHERE id = ?", (triggered_id,)).fetchone()
    assert open_row["status"] == "closed_with_position"
    assert triggered_row["status"] == "triggered"  # bereits ausgeloest, bleibt unveraendert


def test_reduce_or_close_position_closes_boundary_conditions_on_full_close():
    """Integrationstest des Hooks in reduce_or_close_position: eine
    vollstaendig geschlossene Position darf keine offene Randbedingung mehr
    hinterlassen - unabhaengig vom Schliessungsgrund."""
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    position_id = make_open_position(conn, portfolio_id)
    cond_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="offen", check_type="qualitative", threshold_price=None,
    )

    db.reduce_or_close_position(
        conn, position_id, delta_quantity=10.0, fill_price=110.0, closure_reason="claude_decision"
    )

    row = conn.execute("SELECT status FROM boundary_conditions WHERE id = ?", (cond_id,)).fetchone()
    assert row["status"] == "closed_with_position"


def test_reduce_or_close_position_partial_reduction_keeps_conditions_open():
    """Eine Teil-Reduktion schliesst die Position NICHT - die genannte
    These/Randbedingung gilt fuer den verbleibenden Rest weiter."""
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    position_id = make_open_position(conn, portfolio_id)  # quantity=10
    cond_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio_id, position_id=position_id, symbol="NVDA",
        description="offen", check_type="qualitative", threshold_price=None,
    )

    db.reduce_or_close_position(
        conn, position_id, delta_quantity=4.0, fill_price=110.0, closure_reason="claude_decision"
    )

    row = conn.execute("SELECT status FROM boundary_conditions WHERE id = ?", (cond_id,)).fetchone()
    assert row["status"] == "open"


# --- get_successful_decision_today (2026-09-21, Kap. 12.7 Idempotenz-Sperre) --


def insert_raw_decision(
    conn: sqlite3.Connection,
    portfolio_id: int,
    model: str,
    created_at: str,
    proposed_orders: str | None,
) -> int:
    """Umgeht db.insert_decision (das immer datetime('now') als created_at
    verwendet) - fuer diese Tests muss created_at gezielt auf 'heute'/'gestern'
    gesetzt werden koennen."""
    cur = conn.execute(
        """
        INSERT INTO decisions (portfolio_id, created_at, model, prompt, proposed_orders, approved, executed)
        VALUES (?, ?, ?, 'p', ?, 1, 1)
        """,
        (portfolio_id, created_at, model, proposed_orders),
    )
    conn.commit()
    return cur.lastrowid


def test_get_successful_decision_today_none_when_no_decisions_exist():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    assert db.get_successful_decision_today(conn, portfolio_id, "claude-sonnet-5") is None


def test_get_successful_decision_today_finds_matching_decision():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    today = datetime.date.today().isoformat()
    decision_id = insert_raw_decision(
        conn, portfolio_id, "claude-sonnet-5", f"{today} 15:00:00", proposed_orders="[]"
    )
    result = db.get_successful_decision_today(conn, portfolio_id, "claude-sonnet-5")
    assert result is not None
    assert result["id"] == decision_id


def test_get_successful_decision_today_ignores_proposed_orders_null():
    """Boerse-geschlossen-Skip UND OrderParsingError setzen proposed_orders
    NICHT - beide duerfen nicht als abgeschlossene Entscheidung zaehlen (ein
    erneuter Versuch am selben Tag muss z.B. nach einem Parsing-Fehler
    moeglich bleiben)."""
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    today = datetime.date.today().isoformat()
    insert_raw_decision(conn, portfolio_id, "claude-sonnet-5", f"{today} 15:00:00", proposed_orders=None)
    assert db.get_successful_decision_today(conn, portfolio_id, "claude-sonnet-5") is None


def test_get_successful_decision_today_ignores_other_models():
    """deep_reflection/risk_guardrail/pipeline_guard-Eintraege duerfen nicht
    als tagesabschliessende Handelsentscheidung zaehlen."""
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    today = datetime.date.today().isoformat()
    insert_raw_decision(conn, portfolio_id, "deep_reflection", f"{today} 15:00:00", proposed_orders="[]")
    insert_raw_decision(conn, portfolio_id, "risk_guardrail", f"{today} 15:00:00", proposed_orders="[]")
    insert_raw_decision(conn, portfolio_id, "pipeline_guard", f"{today} 15:00:00", proposed_orders="[]")
    assert db.get_successful_decision_today(conn, portfolio_id, "claude-sonnet-5") is None


def test_get_successful_decision_today_ignores_previous_days():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    insert_raw_decision(conn, portfolio_id, "claude-sonnet-5", f"{yesterday} 15:00:00", proposed_orders="[]")
    assert db.get_successful_decision_today(conn, portfolio_id, "claude-sonnet-5") is None


def test_get_successful_decision_today_returns_most_recent_of_several():
    conn = make_conn()
    portfolio_id = make_portfolio(conn)
    today = datetime.date.today().isoformat()
    insert_raw_decision(conn, portfolio_id, "claude-sonnet-5", f"{today} 10:00:00", proposed_orders="[]")
    later_id = insert_raw_decision(conn, portfolio_id, "claude-sonnet-5", f"{today} 15:00:00", proposed_orders="[]")
    result = db.get_successful_decision_today(conn, portfolio_id, "claude-sonnet-5")
    assert result["id"] == later_id


def test_get_successful_decision_today_scoped_to_portfolio():
    conn = make_conn()
    portfolio_a = make_portfolio(conn, name="a")
    portfolio_b = make_portfolio(conn, name="b")
    today = datetime.date.today().isoformat()
    insert_raw_decision(conn, portfolio_a, "claude-sonnet-5", f"{today} 15:00:00", proposed_orders="[]")
    assert db.get_successful_decision_today(conn, portfolio_b, "claude-sonnet-5") is None
