"""Regelbasierte Momentum-Baseline (Thesis Kap. 6.9): Top-Quintil 12-Wochen-
Performance aus dem Aktien-Universum, gleichgewichtet, monatlich
rebalanciert - der nicht-KI-Vergleichsarm. Reine Kursdaten-Rekonstruktion,
kein LLM beteiligt - der Lookahead-Bias-Vorbehalt der Thesis betrifft nur
Claudes Trainingsdaten-Kontaminationsrisiko, nicht diese Regel, daher ist
rueckblickende Berechnung hier unproblematisch.

Architektonisch analog zu `benchmark_normalized` in metrics.py: eine reine,
aus vorhandenen Kursdaten berechnete Kennzahl statt eines separat gehandelten
Portfolios mit eigenen DB-Zeilen/Orders. `reconstruct_nav_history` ruft
`reconstruct_momentum_baseline` auf und haengt das Ergebnis als zusaetzliche
normalisierte Zeitreihe an `NavHistory` an - waechst automatisch mit jedem
Report ueber die volle Studienlaufzeit, kein isolierter Einmal-Snapshot.

Vereinfachung (dokumentiert, analog zu den anderen Approximationen in diesem
Projekt, siehe metrics.py-Modul-Docstring): volle Reinvestition bei jedem
Rebalance, kein Cash-Drag, keine Transaktionskosten.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

PriceLookup = Callable[[str, pd.Timestamp], "float | None"]


def select_top_quintile(returns: dict[str, float], top_fraction: float = 0.2) -> list[str]:
    """Rangiert absteigend nach Rendite und gibt die obersten `top_fraction`
    Symbole zurueck (mindestens 1, sofern `returns` nicht leer ist)."""
    if not returns:
        return []
    ranked = sorted(returns.items(), key=lambda kv: kv[1], reverse=True)
    n = max(1, round(len(ranked) * top_fraction))
    return [symbol for symbol, _ in ranked[:n]]


def monthly_rebalance_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Erster Rebalance-Termin ist `start` selbst, danach der 1. Kalendertag
    jedes weiteren Monats bis `end` (inklusive)."""
    if end < start:
        return [start]
    dates = {start} | set(pd.date_range(start=start, end=end, freq="MS"))
    return sorted(dates)


def reconstruct_momentum_baseline(
    price_on: PriceLookup,
    universe_symbols: list[str],
    dates: list[pd.Timestamp],
    start: pd.Timestamp,
    initial_cash: float,
    lookback_weeks: int = 12,
    top_quintile_fraction: float = 0.2,
) -> list[float]:
    """Rekonstruiert die Wertentwicklung von `initial_cash`, angelegt gemaess
    der Momentum-Regel (Top-Quintil `lookback_weeks`-Wochen-Performance aus
    `universe_symbols`, gleichgewichtet, monatlich rebalanciert seit `start`),
    zu jedem Zeitpunkt in `dates`.

    `price_on(symbol, when)` muss den zuletzt bekannten Schlusskurs bis
    (inklusive) `when` liefern (oder None) - analog zu den `price_on`-Helfern
    in metrics.reconstruct_nav_history/data_fetch. Kein Lookahead: die
    Rangliste an jedem Rebalance-Termin nutzt ausschliesslich Kurse bis zu
    diesem Termin.

    Symbole ohne ausreichende Historie an einem Rebalance-Termin (z.B. IPO
    juenger als `lookback_weeks`) werden nur fuer DIESEN Termin aus der
    Rangliste ausgeschlossen, nicht aus dem Universum insgesamt - sie koennen
    bei einem spaeteren Rebalance wieder teilnehmen.
    """
    if not dates:
        return []
    end = dates[-1]
    rebalance_dates = monthly_rebalance_dates(start, end)

    holdings_by_rebalance: dict[pd.Timestamp, dict[str, float]] = {}
    value_at_rebalance: dict[pd.Timestamp, float] = {}
    running_value = initial_cash

    for i, rdate in enumerate(rebalance_dates):
        if i > 0:
            prev_holdings = holdings_by_rebalance[rebalance_dates[i - 1]]
            if prev_holdings:
                running_value = sum(
                    shares * (price_on(symbol, rdate) or 0.0)
                    for symbol, shares in prev_holdings.items()
                )
            # else: voriger Rebalance konnte niemanden auswaehlen (siehe
            # unten) - Wert bleibt unangetastet in Cash bis zu diesem Termin.
        value_at_rebalance[rdate] = running_value

        lookback_date = rdate - pd.Timedelta(weeks=lookback_weeks)
        returns: dict[str, float] = {}
        for symbol in universe_symbols:
            price_now = price_on(symbol, rdate)
            price_then = price_on(symbol, lookback_date)
            if price_now is not None and price_then is not None and price_then > 0:
                returns[symbol] = price_now / price_then - 1

        selected = select_top_quintile(returns, top_quintile_fraction)
        weight_value = running_value / len(selected) if selected else 0.0
        holdings: dict[str, float] = {}
        for symbol in selected:
            price = price_on(symbol, rdate)
            if price:
                holdings[symbol] = weight_value / price
        holdings_by_rebalance[rdate] = holdings

    result = []
    for d in dates:
        applicable = [r for r in rebalance_dates if r <= d]
        if not applicable:
            result.append(initial_cash)
            continue
        rdate = applicable[-1]
        holdings = holdings_by_rebalance[rdate]
        if holdings:
            result.append(sum(shares * (price_on(symbol, d) or 0.0) for symbol, shares in holdings.items()))
        else:
            result.append(value_at_rebalance[rdate])
    return result
