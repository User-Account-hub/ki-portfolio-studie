"""Performance metrics.

The equity curve for the report/chart is reconstructed here by replaying
`trades` chronologically (cash + position bookkeeping mirrors
execution.py's cash-flow convention) and marking open positions to market
using historical closes from yfinance at each checkpoint. This is an
approximation (intra-period price moves on already-closed positions are not
captured), acceptable for this paper-trading case study's cadence.

Checkpoint frequency (and therefore the annualization factor used in
compute_metrics) is driven by `reconstruct_nav_history`'s `freqs` parameter,
NOT hardcoded - it must match however often the pipeline actually runs (see
.github/workflows/weekly_pipeline.yml's cron schedule). Two pandas weekly
offset aliases, one per run day (default: Monday + Thursday), give 2
checkpoints/week for Phase 1; adding/removing a run day means updating
`freqs` here to match, nothing else in this module.

Regimewechsel Phase 1 -> Phase 2 (Thesis Kap. 6.3/11.2/15): ab `PHASE2_START`
(2026-09-10) laeuft die Studie mit taeglichem Handel (Mo-Fr) statt 2x/Woche.
`reconstruct_nav_history` erzeugt deshalb fuer Zeitpunkte vor `PHASE2_START`
weiterhin Mo/Do-Checkpoints (`freqs`) und ab `PHASE2_START` werktaegliche
Checkpoints. `compute_metrics` haelt Vol/Sharpe strikt getrennt: eine Rendite
zaehlt nur dann als Phase-2-Rendite (Annualisierung mit
`PHASE2_PERIODS_PER_YEAR` = 252), wenn auch ihr INTERVALL-START bereits auf
oder nach `PHASE2_START` liegt - sonst wuerde eine ~3.5-Tage-Phase-1-Rendite
mit einer 1-Tages-Phase-2-Rendite in derselben Standardabweichung vermischt
und mit dem falschen Faktor annualisiert. Solange nicht genug reine
Phase-2-Renditen vorliegen (< 2), faellt die Berechnung auf die alte
Phase-1-Annualisierung (`NavHistory.periods_per_year`) ueber die komplette
Historie zurueck. Vol/Sharpe sind die einzigen Kennzahlen, die von der
Checkpoint-Dichte/-Einteilung abhaengen: total_return vergleicht `nav[-1]`
gegen das feste `NavHistory.initial_nav` (Studien-Startkapital, siehe dortiger
Kommentar - NICHT `nav[0]`) und max_drawdown gegen das laufende Peak/Tal
innerhalb der Zeitreihe, beides unabhaengig vom Checkpoint-Abstand.
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

# Regimewechsel Phase 1 (2x/Woche) -> Phase 2 (taeglich, Mo-Fr), siehe Thesis
# Kap. 6.3/11.2/15. Wie `freqs` unten kein Config-Wert, sondern bewusst hier
# hart kodiert, direkt neben dem Code, der ihn auswertet.
PHASE2_START = pd.Timestamp("2026-09-10")
PHASE2_PERIODS_PER_YEAR = 252.0


@dataclass(frozen=True)
class NavHistory:
    dates: list[pd.Timestamp]
    nav: list[float]
    benchmark_normalized: list[float]
    # Checkpoints/Jahr, abgeleitet aus den `freqs` von reconstruct_nav_history -
    # treibt die Annualisierung in compute_metrics (Volatilitaet, Sharpe).
    periods_per_year: float
    # Bugfix 2026-09-12: das echte Startkapital der Studie
    # (portfolio_row["initial_cash_balance"]), NICHT dasselbe wie `nav[0]`.
    # `nav[0]` ist nur der erste *Checkpoint* der in diesem Lauf rekonstruierten
    # Zeitreihe - je nach Checkpoint-Dichte (siehe Phase 1/2-Regimewechsel)
    # kann das ein beliebiger spaeterer Zeitpunkt nach dem ersten Trade sein,
    # nicht der Studienbeginn. compute_metrics' total_return muss deshalb
    # gegen `initial_nav` rechnen, nicht gegen `nav[0]` - sonst misst
    # "Gesamtrendite" je nach Lauf mal die Rendite seit Studienbeginn, mal nur
    # die Rendite seit dem letzten Checkpoint (was `last_period_return_pct`
    # ohnehin schon abdeckt).
    initial_nav: float


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
    freqs: tuple[str, ...] = ("W-MON", "W-THU"),
    phase2_start: pd.Timestamp = PHASE2_START,
) -> NavHistory:
    """`freqs` are pandas weekly offset aliases, one per weekday the pipeline
    ran during Phase 1 (Monday + Thursday, matching the pre-2026-09-10 cron
    schedule - see module docstring). periods_per_year is derived as
    52 * len(freqs), since each alias contributes one checkpoint per week -
    this remains the Phase-1-only annualization factor stored on NavHistory;
    compute_metrics switches to PHASE2_PERIODS_PER_YEAR once enough
    Phase-2-only returns exist.

    Checkpoints before `phase2_start` follow `freqs` (Phase 1, 2x/week);
    checkpoints from `phase2_start` onward are every business day (Phase 2,
    daily), reflecting the Kap.-6.3 regime change to daily trading."""
    initial_cash = portfolio_row["initial_cash_balance"]
    periods_per_year = 52 * len(freqs)

    if not trades:
        return NavHistory(
            dates=[pd.Timestamp.today()],
            nav=[initial_cash],
            benchmark_normalized=[initial_cash],
            periods_per_year=periods_per_year,
            initial_nav=initial_cash,
        )

    snapshots = _replay_ledger(trades, initial_cash)
    start = snapshots[0]["timestamp"].normalize()
    end = pd.Timestamp.today().normalize()

    phase1_end = min(end, phase2_start)
    phase1_checkpoints = {ts for freq in freqs for ts in pd.date_range(start=start, end=phase1_end, freq=freq)}
    if end > phase2_start:
        phase2_checkpoints = set(
            pd.date_range(start=max(start, phase2_start + pd.Timedelta(days=1)), end=end, freq="B")
        )
    else:
        phase2_checkpoints = set()
    checkpoints = pd.DatetimeIndex(sorted(phase1_checkpoints | phase2_checkpoints))
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

    return NavHistory(
        dates=dates,
        nav=nav_values,
        benchmark_normalized=benchmark_normalized,
        periods_per_year=periods_per_year,
        initial_nav=initial_cash,
    )


def compute_metrics(
    nav_history: NavHistory,
    risk_free_rate_annual: float = 0.0,
    phase2_start: pd.Timestamp = PHASE2_START,
) -> MetricsResult:
    nav = pd.Series(nav_history.nav, index=nav_history.dates)
    benchmark = pd.Series(nav_history.benchmark_normalized, index=nav_history.dates)
    returns = nav.pct_change().dropna()

    # Bugfix 2026-09-12: vorher `nav.iloc[-1] / nav.iloc[0] - 1` - das
    # verwechselte "erster Checkpoint dieses Laufs" mit "Studienbeginn".
    # Fiel bisher nicht auf, weil `nav.iloc[0]` in fast allen bisherigen
    # Laeufen zufaellig gleich `initial_nav` war (nur 1 Checkpoint pro Lauf).
    # Seit dem Phase-1/2-Regimewechsel (mehr Checkpoints/Lauf) driftet
    # `nav.iloc[0]` vom echten Startkapital weg, wodurch "Gesamtrendite" nur
    # noch die Rendite seit dem vorletzten Checkpoint zeigte - identisch zu
    # `last_period_return_pct` statt zur kumulierten Rendite seit Studienstart.
    total_return = nav.iloc[-1] / nav_history.initial_nav - 1
    last_period_return = float(returns.iloc[-1]) if not returns.empty else None

    # Eine Rendite zaehlt nur als Phase-2-Rendite, wenn auch ihr Intervall-
    # Start >= phase2_start liegt - sonst waere es ein Phase-1->Phase-2-
    # Uebergangsintervall mit fremder Frequenz. `interval_start_dates` sind
    # positional (nicht per Label) an `returns` ausgerichtet, da beide gleich
    # lang und gleich sortiert aus derselben `nav`-Serie stammen.
    interval_start_dates = nav.index[:-1]
    is_phase2_return = interval_start_dates >= phase2_start
    phase2_returns = returns[is_phase2_return]

    if len(phase2_returns) >= 2 and phase2_returns.std() > 0:
        periods_per_year = PHASE2_PERIODS_PER_YEAR
        vol_returns = phase2_returns
    else:
        periods_per_year = nav_history.periods_per_year
        vol_returns = returns

    if len(vol_returns) >= 2 and vol_returns.std() > 0:
        rf_per_period = risk_free_rate_annual / periods_per_year
        annualized_vol = float(vol_returns.std() * np.sqrt(periods_per_year))
        sharpe = float((vol_returns.mean() - rf_per_period) / vol_returns.std() * np.sqrt(periods_per_year))
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
