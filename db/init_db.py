"""Initialize the SQLite database from schema.sql and seed the portfolio row.

Usage:
    python db/init_db.py
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: str, portfolio_name: str, initial_cash: float, currency: str, benchmark_symbol: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        existing = conn.execute(
            "SELECT id FROM portfolios WHERE name = ?", (portfolio_name,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO portfolios (name, currency, initial_cash_balance, cash_balance, benchmark_symbol)
                VALUES (?, ?, ?, ?, ?)
                """,
                (portfolio_name, currency, initial_cash, initial_cash, benchmark_symbol),
            )
            conn.commit()
            print(f"Portfolio '{portfolio_name}' angelegt mit Startkapital {initial_cash} {currency}.")
        else:
            print(f"Portfolio '{portfolio_name}' existiert bereits (id={existing[0]}), keine Änderung.")
    finally:
        conn.close()


if __name__ == "__main__":
    load_dotenv()
    init_db(
        db_path=os.getenv("DB_PATH", "./db/portfolio.db"),
        portfolio_name=os.getenv("PORTFOLIO_NAME", "ki-fallstudie-1"),
        initial_cash=float(os.getenv("INITIAL_CASH_BALANCE", "100000")),
        currency=os.getenv("PORTFOLIO_CURRENCY", "USD"),
        benchmark_symbol=os.getenv("BENCHMARK_SYMBOL", "SPY"),
    )
