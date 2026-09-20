"""Thematischer Segment-ETF-Korb (Thesis Kap. 6.9): gleichgewichtetes,
monatlich rebalanciertes Portfolio aus SMH (Halbleiter), URA (Uran) und ICLN
(Clean Energy) - ein zusätzlicher, rein informativer Vergleichspunkt,
analog zur Momentum-Baseline (src/momentum_baseline.py) berechnet: reine
Kursdaten-Rekonstruktion, kein LLM beteiligt, kein separat gehandeltes
Portfolio mit eigenen DB-Zeilen/Orders.

Anders als die Momentum-Baseline gibt es hier KEINE Auswahl/Rangliste - die
drei Symbole sind fest (Kap. 6.9 der Thesis nennt sie namentlich als
"Segment-ETF-Korb", eine thematische Referenz für das stark AI-/Energie-
lastige Anlage-Universum, Kap. 6.7) und werden bei jedem monatlichen
Rebalance-Termin zu gleichen Teilen gehalten. Nutzt denselben Rebalance-
Rhythmus (`monthly_rebalance_dates`) wie die Momentum-Baseline für
Konsistenz.

Vereinfachung (dokumentiert, analog zu momentum_baseline.py/metrics.py):
volle Reinvestition bei jedem Rebalance, kein Cash-Drag, keine
Transaktionskosten.
"""
from __future__ import annotations

import pandas as pd

from src.momentum_baseline import PriceLookup, monthly_rebalance_dates

# Kap. 6.9 der Thesis ("Segment-ETF-Korb"): Halbleiter, Uran, Clean Energy -
# bewusst fest kodiert (kein frei konfigurierbares Universum wie bei der
# Momentum-Baseline), analog zu metrics.SECONDARY_BENCHMARK_SYMBOL (QQQ).
SEGMENT_BASKET_SYMBOLS: tuple[str, ...] = ("SMH", "URA", "ICLN")


def reconstruct_segment_basket(
    price_on: PriceLookup,
    dates: list[pd.Timestamp],
    start: pd.Timestamp,
    initial_cash: float,
    symbols: tuple[str, ...] = SEGMENT_BASKET_SYMBOLS,
) -> list[float]:
    """Rekonstruiert die Wertentwicklung von `initial_cash`, gleichgewichtet
    in `symbols` angelegt seit `start`, monatlich rebalanciert, zu jedem
    Zeitpunkt in `dates`.

    `price_on(symbol, when)` muss den zuletzt bekannten Schlusskurs bis
    (inklusive) `when` liefern (oder None) - identische Semantik wie in
    momentum_baseline.reconstruct_momentum_baseline/metrics.
    reconstruct_nav_history. Kein Lookahead: die Gewichtung an jedem
    Rebalance-Termin nutzt ausschliesslich Kurse bis zu diesem Termin.

    Architektonisch dasselbe Modell wie die Momentum-Baseline, nur OHNE
    Auswahl/Rangliste (immer alle `symbols`, immer gleichgewichtet statt
    "Top-Quintil"). Ein Symbol ohne Kurs an einem gegebenen Rebalance-Termin
    wird für DIESEN Termin übersprungen (die verbleibenden teilen sich den
    Wert), nicht aus `symbols` insgesamt entfernt - es kann beim nächsten
    Rebalance wieder teilnehmen, sobald wieder ein Kurs verfügbar ist.
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
                    shares * (price_on(symbol, rdate) or 0.0) for symbol, shares in prev_holdings.items()
                )
            # else: voriger Rebalance konnte niemanden auswaehlen (siehe
            # unten) - Wert bleibt unangetastet in Cash bis zu diesem Termin.
        value_at_rebalance[rdate] = running_value

        available = [s for s in symbols if price_on(s, rdate) is not None]
        weight_value = running_value / len(available) if available else 0.0
        holdings: dict[str, float] = {}
        for symbol in available:
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
