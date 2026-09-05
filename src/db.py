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


def open_positions_as_risk_objects(rows: list[sqlite3.Row]) -> list[OpenPosition]:
    return [
        OpenPosition(
            symbol=r["symbol"],
            instrument_type=r["instrument_type"],
            side=r["side"],
            quantity=r["quantity"],
            avg_entry_price=r["avg_entry_price"],
        )
        for r in rows
    ]


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
) -> int:
    cur = conn.execute(
        """
        INSERT INTO trades
            (portfolio_id, position_id, decision_id, symbol, instrument_type, side,
             quantity, price, notional, order_type, broker_order_id, source, status, rejection_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
