"""Weicher Qualitäts-Score als zusätzlicher Prompt-Kontext: grobe
Fundamentaldaten (Umsatzwachstum, Verschuldungsgrad, Free Cashflow) via
yfinance, wo verfügbar, für jeden Titel im Universum.

Ausdrücklich KEIN Filter/Guardrail: es gibt keinen Schwellenwert, der einen
Titel ausschliesst - ein unprofitables oder hoch verschuldetes Symbol
erscheint genauso im Prompt-Kontext wie jedes andere, nur mit den
zusätzlichen Zahlen daneben. Ein automatischer Ausschluss würde dem bewusst
spekulativen Charakter des Anlage-Universums widersprechen. Claude
entscheidet selbst, wie es diese Zahlen gewichtet - dieselbe Philosophie
wie der Korrelations-Check (correlation.py), das Randbedingungs-Tracking
(Kap. 7) und der Event-Kalender-Hinweis (event_calendar.py).

Fehlende Daten (z.B. ETFs/strukturierte Produkte ohne Einzelunternehmens-
Fundamentaldaten, sehr junge Börsengänge ohne ausreichende Kennzahlen-
Historie) werden sauber übersprungen - weder ein Fehler noch ein
Platzhalterwert.
"""
from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass

import yfinance as yf

log = logging.getLogger("pipeline")

DEFAULT_FUNDAMENTALS_FETCH_MAX_WORKERS = 10


@dataclass(frozen=True)
class FundamentalSnapshot:
    symbol: str
    revenue_growth: float | None
    debt_to_equity: float | None
    free_cash_flow: float | None


def fetch_fundamentals(symbol: str) -> FundamentalSnapshot | None:
    """Grobe Fundamentaldaten für `symbol` aus yfinance's `Ticker.info`
    (ein einzelner, bereits von yfinance aggregierter Snapshot - keine
    eigene Berechnung aus Rohbilanzen, absichtlich "grob" statt exakt).

    None, falls KEINES der drei Felder verfügbar ist (z.B. ETFs,
    strukturierte Produkte, sehr junge Börsengänge). Einzelne fehlende
    Felder bei sonst vorhandenen Daten bleiben als `None` im Ergebnis
    stehen, statt das ganze Symbol zu verwerfen - "wo verfügbar" gilt pro
    Feld, nicht nur pro Symbol.

    Wirft NIE - ein Fehler für ein einzelnes Symbol darf den Abruf für alle
    anderen Symbole nicht gefährden (siehe fetch_universe_fundamentals)."""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        log.debug("Keine Fundamentaldaten für %s abrufbar.", symbol, exc_info=True)
        return None
    revenue_growth = info.get("revenueGrowth")
    debt_to_equity = info.get("debtToEquity")
    free_cash_flow = info.get("freeCashflow")
    if revenue_growth is None and debt_to_equity is None and free_cash_flow is None:
        return None
    return FundamentalSnapshot(
        symbol=symbol,
        revenue_growth=revenue_growth,
        debt_to_equity=debt_to_equity,
        free_cash_flow=free_cash_flow,
    )


def fetch_universe_fundamentals(
    symbols: list[str],
    max_workers: int = DEFAULT_FUNDAMENTALS_FETCH_MAX_WORKERS,
) -> dict[str, FundamentalSnapshot]:
    """Holt die Fundamentaldaten PARALLEL ab (ThreadPoolExecutor) - analog
    zu event_calendar.check_upcoming_earnings: yfinance bietet für
    `Ticker.info` keine gebündelte Mehr-Symbol-Abfrage wie `yf.download()`;
    bei 100+ Symbolen wäre eine sequenzielle Abfrage (ein HTTP-Request pro
    Symbol) spürbar langsam. `max_workers` begrenzt die Parallelität
    bewusst.

    Nur Symbole mit mindestens einem verfügbaren Feld sind im Ergebnis-
    Dict enthalten - kein Eintrag für vollständig unbekannte Symbole."""
    results: dict[str, FundamentalSnapshot] = {}
    if not symbols:
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_fundamentals, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                # fetch_fundamentals faengt intern bereits alles ab und
                # wirft nie - dieses try/except ist eine zusaetzliche
                # Absicherung, falls das je verletzt wird (siehe die
                # analoge Begruendung in event_calendar.check_upcoming_earnings).
                snapshot = future.result()
            except Exception:
                log.exception("Fundamentaldaten-Abfrage für %s fehlgeschlagen - wird übersprungen.", symbol)
                continue
            if snapshot is not None:
                results[symbol] = snapshot
    return results
