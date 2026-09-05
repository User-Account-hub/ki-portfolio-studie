"""Performance metrics.

The DB schema deliberately has no dedicated NAV-history table (only the
four requested tables). Instead, the equity curve is reconstructed here by
replaying `trades` chronologically (cash + position bookkeeping mirrors
execution.py's cash-flow convention) and marking open positions to market
using historical closes from yfinance at each weekly checkpoint. This is an
approximation (intra-week price moves on already-closed positions are not
captured), acceptable for a weekly-cadence paper-trading case study.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from src.risk_guardrails import OpenPosition, compute_nav

CASH_SIGN = {"buy": -1, "sell": 1, "short": 1, "cover": -1}
OPEN_SIDES = {"buy": "long", "short": "short"}
CLOSE_SIDES = {"sell": "long", "cover": "short"}


@dataclass(frozen=True)
class NavHistory:
    dates: list[pd.Timestamp]
    nav: list[float]
    benchmark_normalized: list[float]


@dataclass(frozen=True)
class MetricsResult:
    current_nav: float
    total_return_pct: float
    last_period_return_pct: float | None
    annualized_volatility_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float
    benchmark_total_return_pct: float
    alpha_pct: float


def _replay_ledger(trades: list[sqlite3.Row], initial_cash: float) -> list[dict]:
    cash = initial_cash
    positions: dict[tuple[str, str], dict] = {}
    snapshots = []

    for t in trades:
        side, qty, price, symbol = t["side"], t["quantity"], t["price"], t["symbol"]
        cash += CASH_SIGN[side] * qty * price

        if side in OPEN_SIDES:
            key = (symbol, OPEN_SIDES[side])
            pos = positions.get(key)
            if pos is None:
                positions[key] = {
                    "instrument_type": t["instrument_type"],
                    "side": key[1],
                    "quantity": qty,
                    "avg_entry_price": price,
                }
            else:
                new_qty = pos["quantity"] + qty
                pos["avg_entry_price"] = (pos["quantity"] * pos["avg_entry_price"] + qty * price) / new_qty
                pos["quantity"] = new_qty
        else:
            key = (symbol, CLOSE_SIDES[side])
            pos = positions.get(key)
            if pos is not None:
                pos["quantity"] -= qty
                if pos["quantity"] <= 1e-9:
                    del positions[key]

        snapshots.append(
            {
                "timestamp": pd.Timestamp(t["executed_at"]),
                "cash": cash,
                "positions": [OpenPosition(symbol=k[0], **v) for k, v in positions.items()],
            }
        )
    return snapshots


def _price_lookup_symbols(snapshots: list[dict], watchlist_underlyings: dict[str, str]) -> list[str]:
    symbols = set()
    for snap in snapshots:
        for p in snap["positions"]:
            symbols.add(watchlist_underlyings.get(p.symbol, p.symbol))
    return sorted(symbols)


def reconstruct_nav_history(
    conn: sqlite3.Connection,
    portfolio_row: sqlite3.Row,
    trades: list[sqlite3.Row],
    watchlist_underlyings: dict[str, str],
    benchmark_symbol: str,
    freq: str = "W-MON",
) -> NavHistory:
    initial_cash = portfolio_row["initial_cash_balance"]

    if not trades:
        return NavHistory(dates=[pd.Timestamp.today()], nav=[initial_cash], benchmark_normalized=[initial_cash])

    snapshots = _replay_ledger(trades, initial_cash)
    start = snapshots[0]["timestamp"].normalize()
    end = pd.Timestamp.today().normalize()
    checkpoints = pd.date_range(start=start, end=end, freq=freq)
    if len(checkpoints) == 0 or checkpoints[-1] < end:
        checkpoints = checkpoints.append(pd.DatetimeIndex([end]))

    price_symbols = sorted(set(_price_lookup_symbols(snapshots, watchlist_underlyings)) | {benchmark_symbol})
    history = yf.download(
        tickers=price_symbols,
        start=start - pd.Timedelta(days=7),
        end=end + pd.Timedelta(days=1),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    def price_on(symbol: str, when: pd.Timestamp) -> float | None:
        try:
            series = history[symbol]["Close"] if len(price_symbols) > 1 else history["Close"]
            series = series.dropna()
            eligible = series[series.index <= when]
            return float(eligible.iloc[-1]) if not eligible.empty else None
        except (KeyError, IndexError):
            return None

    nav_values, dates = [], []
    for checkpoint in checkpoints:
        applicable = [s for s in snapshots if s["timestamp"] <= checkpoint]
        snap = applicable[-1] if applicable else {"cash": initial_cash, "positions": []}
        current_prices = {}
        for p in snap["positions"]:
            lookup_symbol = watchlist_underlyings.get(p.symbol, p.symbol)
            price = price_on(lookup_symbol, checkpoint)
            if price is not None:
                current_prices[p.symbol] = price
        nav = compute_nav(snap["cash"], snap["positions"], current_prices)
        nav_values.append(nav)
        dates.append(checkpoint)

    benchmark_start_price = price_on(benchmark_symbol, dates[0])
    benchmark_normalized = []
    for d in dates:
        price = price_on(benchmark_symbol, d)
        if price is None or benchmark_start_price is None:
            benchmark_normalized.append(initial_cash)
        else:
            benchmark_normalized.append(initial_cash * (price / benchmark_start_price))

    return NavHistory(dates=dates, nav=nav_values, benchmark_normalized=benchmark_normalized)


def compute_metrics(nav_history: NavHistory, risk_free_rate_annual: float = 0.0) -> MetricsResult:
    nav = pd.Series(nav_history.nav, index=nav_history.dates)
    benchmark = pd.Series(nav_history.benchmark_normalized, index=nav_history.dates)
    returns = nav.pct_change().dropna()

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    last_period_return = float(returns.iloc[-1]) if not returns.empty else None

    periods_per_year = 52  # wöchentliche Checkpoints
    if len(returns) >= 2 and returns.std() > 0:
        rf_per_period = risk_free_rate_annual / periods_per_year
        annualized_vol = float(returns.std() * np.sqrt(periods_per_year))
        sharpe = float((returns.mean() - rf_per_period) / returns.std() * np.sqrt(periods_per_year))
    else:
        annualized_vol = None
        sharpe = None

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_drawdown = float(drawdown.min())

    benchmark_total_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)

    return MetricsResult(
        current_nav=float(nav.iloc[-1]),
        total_return_pct=float(total_return),
        last_period_return_pct=last_period_return,
        annualized_volatility_pct=annualized_vol,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_drawdown,
        benchmark_total_return_pct=benchmark_total_return,
        alpha_pct=float(total_return - benchmark_total_return),
    )
