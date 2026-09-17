"""Thin SQLite data-access layer on top of db/schema.sql."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Iterator, Optional

from src.risk_guardrails import OpenPosition


@contextmanager
def get_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def ensure_nav_history_table(conn: sqlite3.Connection) -> None:
    """Forward-compatible migration for a db/portfolio.db created before
    nav_history existed (e.g. an already-deployed DB from earlier pipeline
    runs). CREATE TABLE IF NOT EXISTS is a no-op when the table is already
    there (via a fresh db/init_db.py run using the current schema.sql) -
    safe to call unconditionally at the start of every pipeline run; never
    touches existing tables/data."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS nav_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id),
            recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
            nav             REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_nav_history_portfolio_recorded ON nav_history(portfolio_id, recorded_at);
        """
    )
    conn.commit()


def ensure_trade_transaction_cost_column(conn: sqlite3.Connection) -> None:
    """Forward-compatible migration (2026-09-17, feste Kosten-Annahme pro
    Trade) für ein db/portfolio.db, das vor der `transaction_cost`-Spalte
    erstellt wurde - analog zu ensure_nav_history_table oben, nur für eine
    Spalte statt eine Tabelle. SQLite kennt kein `ADD COLUMN IF NOT EXISTS`,
    daher der PRAGMA table_info-Check: ein wiederholter Aufruf bei jedem
    Pipeline-Lauf darf nicht mit "duplicate column name" abstürzen. DEFAULT 0
    gilt für bereits vorhandene Zeilen - historische Trades vor dieser
    Änderung rückwirkend Kosten zu unterstellen wäre falsch."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(trades)")}
    if "transaction_cost" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN transaction_cost REAL NOT NULL DEFAULT 0")
        conn.commit()


def ensure_boundary_conditions_table(conn: sqlite3.Connection) -> None:
    """Forward-compatible migration (2026-09-17, Randbedingungs-Tracking
    Kap. 7) für ein db/portfolio.db, das vor der `boundary_conditions`-
    Tabelle erstellt wurde - analog zu ensure_nav_history_table oben."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS boundary_conditions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id),
            position_id         INTEGER NOT NULL REFERENCES positions(id),
            decision_id         INTEGER REFERENCES decisions(id),
            symbol              TEXT NOT NULL,
            description         TEXT NOT NULL,
            check_type          TEXT NOT NULL CHECK (check_type IN ('price_above', 'price_below', 'qualitative')),
            threshold_price     REAL,
            status              TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'triggered', 'closed_with_position')),
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            triggered_at        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_boundary_conditions_portfolio_status ON boundary_conditions(portfolio_id, status);
        CREATE INDEX IF NOT EXISTS idx_boundary_conditions_position ON boundary_conditions(position_id);
        """
    )
    conn.commit()


def record_nav(conn: sqlite3.Connection, portfolio_id: int, nav: float) -> int:
    """Records one NAV data point (typically the NAV at pipeline run start)."""
    cur = conn.execute(
        "INSERT INTO nav_history (portfolio_id, nav) VALUES (?, ?)",
        (portfolio_id, nav),
    )
    conn.commit()
    return cur.lastrowid


def get_peak_nav(conn: sqlite3.Connection, portfolio_id: int) -> Optional[float]:
    """True historical NAV high-water mark (MAX over every recorded run for
    this portfolio) - used by risk_guardrails.check_circuit_breaker. Returns
    None if nav_history has no rows yet for this portfolio (fresh DB before
    the first record_nav call)."""
    row = conn.execute(
        "SELECT MAX(nav) AS peak FROM nav_history WHERE portfolio_id = ?", (portfolio_id,)
    ).fetchone()
    return row["peak"] if row is not None and row["peak"] is not None else None


def get_portfolio(conn: sqlite3.Connection, name: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM portfolios WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(
            f"Portfolio '{name}' existiert nicht. Bitte zuerst `python db/init_db.py` ausführen."
        )
    return row


def get_open_positions(conn: sqlite3.Connection, portfolio_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM positions WHERE portfolio_id = ? AND status = 'open'", (portfolio_id,)
    ).fetchall()


def open_positions_as_risk_objects(
    rows: list[sqlite3.Row], symbol_metadata: Optional[dict] = None
) -> list[OpenPosition]:
    """`symbol_metadata` maps symbol -> object with .segment/.cap_tier
    attributes (typically {s.symbol: s for s in watchlist.symbols}), used to
    enrich positions for the Kap.-6.8 segment/cap-tier guardrails. Symbols
    without an entry (e.g. structured products) get segment=cap_tier=None."""
    symbol_metadata = symbol_metadata or {}
    positions = []
    for r in rows:
        meta = symbol_metadata.get(r["symbol"])
        positions.append(
            OpenPosition(
                symbol=r["symbol"],
                instrument_type=r["instrument_type"],
                side=r["side"],
                quantity=r["quantity"],
                avg_entry_price=r["avg_entry_price"],
                segment=meta.segment if meta is not None else None,
                cap_tier=meta.cap_tier if meta is not None else None,
                leveraged=getattr(meta, "leveraged", False) if meta is not None else False,
            )
        )
    return positions


def count_trades_today_by_symbol(conn: sqlite3.Connection, portfolio_id: int) -> dict[str, int]:
    today = date.today().isoformat()
    rows = conn.execute(
        """
        SELECT symbol, COUNT(*) AS cnt
        FROM trades
        WHERE portfolio_id = ? AND date(executed_at) = ? AND status IN ('filled', 'simulated')
        GROUP BY symbol
        """,
        (portfolio_id, today),
    ).fetchall()
    return {r["symbol"]: r["cnt"] for r in rows}


def insert_decision(
    conn: sqlite3.Connection,
    portfolio_id: int,
    model: str,
    prompt: str,
    raw_response: Optional[str],
    proposed_orders: Optional[list[dict]],
    risk_check_result: Optional[list[dict]],
    rationale: Optional[str],
    forced_action: bool,
    approved: bool,
    executed: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO decisions
            (portfolio_id, model, prompt, raw_response, proposed_orders, risk_check_result,
             rationale, forced_action, approved, executed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            portfolio_id,
            model,
            prompt,
            raw_response,
            json.dumps(proposed_orders) if proposed_orders is not None else None,
            json.dumps(risk_check_result) if risk_check_result is not None else None,
            rationale,
            int(forced_action),
            int(approved),
            int(executed),
        ),
    )
    conn.commit()
    return cur.lastrowid


def upsert_open_position(
    conn: sqlite3.Connection,
    portfolio_id: int,
    symbol: str,
    instrument_type: str,
    underlying_symbol: Optional[str],
    side: str,
    delta_quantity: float,
    fill_price: float,
    stop_loss_price: Optional[float] = None,
) -> int:
    """Adds to (or creates) an open position for `symbol`/`side`.

    Position averaging uses a simple weighted-average entry price.
    """
    row = conn.execute(
        "SELECT * FROM positions WHERE portfolio_id = ? AND symbol = ? AND side = ? AND status = 'open'",
        (portfolio_id, symbol, side),
    ).fetchone()

    if row is None:
        cur = conn.execute(
            """
            INSERT INTO positions
                (portfolio_id, symbol, instrument_type, underlying_symbol, side,
                 quantity, avg_entry_price, stop_loss_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (portfolio_id, symbol, instrument_type, underlying_symbol, side, delta_quantity, fill_price, stop_loss_price),
        )
        conn.commit()
        return cur.lastrowid

    new_quantity = row["quantity"] + delta_quantity
    new_avg_price = ((row["quantity"] * row["avg_entry_price"]) + (delta_quantity * fill_price)) / new_quantity
    conn.execute(
        """
        UPDATE positions
        SET quantity = ?, avg_entry_price = ?, stop_loss_price = COALESCE(?, stop_loss_price),
            updated_at = datetime('now')
        WHERE id = ?
        """,
        (new_quantity, new_avg_price, stop_loss_price, row["id"]),
    )
    conn.commit()
    return row["id"]


def reduce_or_close_position(
    conn: sqlite3.Connection,
    position_id: int,
    delta_quantity: float,
    fill_price: float,
    closure_reason: Optional[str] = None,
    closure_notes: Optional[str] = None,
) -> None:
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    remaining = row["quantity"] - delta_quantity
    sign = 1 if row["side"] == "long" else -1
    realized_pnl_delta = sign * (fill_price - row["avg_entry_price"]) * delta_quantity

    if remaining <= 1e-9:
        conn.execute(
            """
            UPDATE positions
            SET quantity = 0, status = 'closed', closed_at = datetime('now'),
                closure_reason = ?, closure_notes = ?, realized_pnl = COALESCE(realized_pnl, 0) + ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (closure_reason, closure_notes, realized_pnl_delta, position_id),
        )
        # Kap. 7: eine geschlossene Position hat keine offene These mehr zu
        # schuetzen/beobachten - unabhaengig vom Schliessungsgrund werden
        # ihre noch offenen Randbedingungen als erledigt markiert, statt als
        # Karteileichen liegenzubleiben (siehe boundary_conditions.py).
        close_boundary_conditions_for_position(conn, position_id)
    else:
        conn.execute(
            """
            UPDATE positions
            SET quantity = ?, realized_pnl = COALESCE(realized_pnl, 0) + ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (remaining, realized_pnl_delta, position_id),
        )
    conn.commit()


def insert_trade(
    conn: sqlite3.Connection,
    portfolio_id: int,
    position_id: Optional[int],
    decision_id: Optional[int],
    symbol: str,
    instrument_type: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str,
    broker_order_id: Optional[str],
    source: str,
    status: str,
    rejection_reason: Optional[str] = None,
    transaction_cost: float = 0.0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO trades
            (portfolio_id, position_id, decision_id, symbol, instrument_type, side,
             quantity, price, notional, transaction_cost, order_type, broker_order_id,
             source, status, rejection_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            portfolio_id,
            position_id,
            decision_id,
            symbol,
            instrument_type,
            side,
            quantity,
            price,
            quantity * price,
            transaction_cost,
            order_type,
            broker_order_id,
            source,
            status,
            rejection_reason,
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_cash_balance(conn: sqlite3.Connection, portfolio_id: int, delta: float) -> None:
    conn.execute(
        "UPDATE portfolios SET cash_balance = cash_balance + ?, updated_at = datetime('now') WHERE id = ?",
        (delta, portfolio_id),
    )
    conn.commit()


def get_recent_decisions(conn: sqlite3.Connection, portfolio_id: int, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM decisions WHERE portfolio_id = ? ORDER BY created_at DESC LIMIT ?",
        (portfolio_id, limit),
    ).fetchall()


def get_all_trades(conn: sqlite3.Connection, portfolio_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM trades WHERE portfolio_id = ? ORDER BY executed_at ASC", (portfolio_id,)
    ).fetchall()


# --- Monatliche Tiefenreflexion (Kap. 6.12.3, 2026-09-17) -----------------------
# Kein eigenes Schema noetig: eine Tiefenreflexion ist ein 'decisions'-Eintrag
# mit model='deep_reflection' (proposed_orders/risk_check_result bleiben NULL,
# da rein analytisch - siehe pipeline.py._maybe_run_deep_reflection).


def get_decisions_since(conn: sqlite3.Connection, portfolio_id: int, since: str) -> list[sqlite3.Row]:
    """Decisions seit `since`, OHNE vorherige Tiefenreflexionen selbst - die
    Reflexion soll die taeglichen Handelsentscheidungen der Periode
    bewerten, nicht ihre eigenen frueheren Ausgaben.

    `since` muss im selben Textformat wie `created_at` vorliegen
    (`YYYY-MM-DD HH:MM:SS`, SQLite's `datetime('now')`-Format) - NICHT
    `pd.Timestamp.isoformat()`/`datetime.isoformat()` (die ein "T" statt
    eines Leerzeichens einfuegen, was den String-Vergleich >= fuer Zeiten am
    selben Tag falsch herum sortieren wuerde). Aufrufer sollten
    `.strftime("%Y-%m-%d %H:%M:%S")` verwenden, siehe pipeline.py."""
    return conn.execute(
        """
        SELECT * FROM decisions
        WHERE portfolio_id = ? AND created_at >= ? AND model != 'deep_reflection'
        ORDER BY created_at ASC
        """,
        (portfolio_id, since),
    ).fetchall()


def get_trades_since(conn: sqlite3.Connection, portfolio_id: int, since: str) -> list[sqlite3.Row]:
    """Siehe get_decisions_since fuer das erwartete Format von `since`."""
    return conn.execute(
        "SELECT * FROM trades WHERE portfolio_id = ? AND executed_at >= ? ORDER BY executed_at ASC",
        (portfolio_id, since),
    ).fetchall()


def count_deep_reflections(conn: sqlite3.Connection, portfolio_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM decisions WHERE portfolio_id = ? AND model = 'deep_reflection'",
        (portfolio_id,),
    ).fetchone()
    return row["n"]


def get_latest_reflection(conn: sqlite3.Connection, portfolio_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM decisions
        WHERE portfolio_id = ? AND model = 'deep_reflection'
        ORDER BY created_at DESC LIMIT 1
        """,
        (portfolio_id,),
    ).fetchone()


# --- Randbedingungs-Tracking (Kap. 7, 2026-09-17) -------------------------------


def insert_boundary_condition(
    conn: sqlite3.Connection,
    portfolio_id: int,
    position_id: int,
    symbol: str,
    description: str,
    check_type: str,
    threshold_price: Optional[float],
    decision_id: Optional[int] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO boundary_conditions
            (portfolio_id, position_id, decision_id, symbol, description, check_type, threshold_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (portfolio_id, position_id, decision_id, symbol, description, check_type, threshold_price),
    )
    conn.commit()
    return cur.lastrowid


def get_open_boundary_conditions(conn: sqlite3.Connection, portfolio_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM boundary_conditions WHERE portfolio_id = ? AND status = 'open'",
        (portfolio_id,),
    ).fetchall()


def mark_boundary_conditions_triggered(conn: sqlite3.Connection, ids: list[int]) -> None:
    if not ids:
        return
    conn.executemany(
        "UPDATE boundary_conditions SET status = 'triggered', triggered_at = datetime('now') WHERE id = ?",
        [(i,) for i in ids],
    )
    conn.commit()


def close_boundary_conditions_for_position(conn: sqlite3.Connection, position_id: int) -> None:
    """Markiert alle noch offenen Randbedingungen einer Position als
    'closed_with_position' - aufgerufen aus reduce_or_close_position bei
    vollstaendigem Schluss, unabhaengig vom Schliessungsgrund. Eigenstaendig
    commit-faehig fuer direkte Aufrufe/Tests."""
    conn.execute(
        "UPDATE boundary_conditions SET status = 'closed_with_position' WHERE position_id = ? AND status = 'open'",
        (position_id,),
    )
    conn.commit()
