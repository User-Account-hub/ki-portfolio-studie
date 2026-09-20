"""Event-Kalender-Hinweis: bevorstehende Quartalsberichte (yfinance),
hardcodierte Makro-Termine (FOMC/CPI) und Options-Verfallstage (2026-09-21)
als zusätzlicher Kontext für Claude.

Rein informativ - KEIN Guardrail, KEIN automatisches Verbot. Ein erhöhtes
Ereignisrisiko wird im Prompt-Kontext genannt (siehe prompt_builder.py),
Claude entscheidet selbst, ob/wie es das in Positionsgrösse oder Timing
einbezieht - dieselbe Philosophie wie das Kap.-7-Randbedingungs-Tracking
und der Korrelations-Check (src/correlation.py): dokumentieren, nicht
automatisch entscheiden. Für Options-Verfallstage gilt das erst recht: das
Anlage-Universum handelt selbst keine Optionen (siehe README, "Wichtige
Einschränkungen") - der Hinweis dokumentiert nur die historisch erhöhte
Volatilität der zugrundeliegenden Aktien/ETFs an diesen Tagen.

"Handelstage" ist hier eine einfache Mo-Fr-Näherung (pandas `bdate_range`),
keine echte Börsenfeiertagsprüfung wie broker_alpaca.is_trading_day - für
eine Kontext-Anreicherung ausreichend, siehe auch metrics.py's `freqs`, das
denselben Kompromiss eingeht.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import logging
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

log = logging.getLogger("pipeline")

DEFAULT_EVENT_WINDOW_TRADING_DAYS = 3
DEFAULT_EARNINGS_FETCH_MAX_WORKERS = 10

# Hardcodierte FOMC-Zinsentscheid-Termine 2026 (jeweils der zweite Tag der
# zweitägigen Sitzung - dort liegt die eigentliche Marktbewegung). Quelle:
# federalreserve.gov / fedratecalc.com, recherchiert 2026-09-18. Kein
# Live-Feed - muss für ein weiteres Jahr manuell ergänzt werden (analog zu
# metrics.DEFAULT_RISK_FREE_RATE_ANNUAL).
FOMC_DECISION_DATES: list[datetime.date] = [
    datetime.date(2026, 1, 28),
    datetime.date(2026, 3, 18),
    datetime.date(2026, 4, 29),
    datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28),
    datetime.date(2026, 12, 9),
]

# Hardcodierte CPI-Veröffentlichungstermine 2026. Quelle: bls.gov/schedule/
# news_release/cpi.htm, recherchiert 2026-09-18.
CPI_RELEASE_DATES: list[datetime.date] = [
    datetime.date(2026, 1, 13),
    datetime.date(2026, 2, 13),
    datetime.date(2026, 3, 11),
    datetime.date(2026, 4, 10),
    datetime.date(2026, 5, 12),
    datetime.date(2026, 6, 10),
    datetime.date(2026, 7, 14),
    datetime.date(2026, 8, 12),
    datetime.date(2026, 9, 11),
    datetime.date(2026, 10, 14),
    datetime.date(2026, 11, 10),
    datetime.date(2026, 12, 10),
]

# Monate mit gleichzeitigem Verfall von Aktienindex-Futures, Index-Optionen
# UND Aktienoptionen ("Hexensabbat"/"Triple Witching") - die staerkste der
# beiden Verfallsauspraegungen unten. Anders als FOMC_DECISION_DATES/
# CPI_RELEASE_DATES oben ist das kein hartcodiertes Datum, das jaehrlich neu
# recherchiert werden muss, sondern eine feste, jedes Jahr gleich
# wiederkehrende Kalenderregel.
QUARTERLY_WITCHING_MONTHS = {3, 6, 9, 12}


def third_friday_of_month(year: int, month: int) -> datetime.date:
    """Regulaerer US-Options-Verfallstag (dritter Freitag des Monats,
    "Hexensabbat" in den Quartalsmonaten oben) - BERECHNET statt hartcodiert,
    da es sich (anders als FOMC-Sitzungen/CPI-Veroeffentlichungen) um eine
    feste, nie von einer externen Ankuendigung abhaengende Kalenderregel
    handelt."""
    first_of_month = datetime.date(year, month, 1)
    days_until_friday = (4 - first_of_month.weekday()) % 7  # Montag=0 ... Freitag=4
    first_friday = first_of_month + datetime.timedelta(days=days_until_friday)
    return first_friday + datetime.timedelta(weeks=2)


@dataclass(frozen=True)
class EarningsWarning:
    symbol: str
    earnings_date: datetime.date
    trading_days_until: int


@dataclass(frozen=True)
class MacroEvent:
    name: str  # "FOMC" | "CPI" | "Hexensabbat (Options-Quartalsverfall)" | "Optionsverfall (monatlich)"
    event_date: datetime.date
    trading_days_until: int


def trading_days_until(today: datetime.date, event_date: datetime.date) -> int:
    """Handelstage (Mo-Fr-Näherung) zwischen `today` und `event_date`
    (>= today vorausgesetzt) - 0, falls `event_date` == `today`."""
    return len(pd.bdate_range(start=today, end=event_date)) - 1


def _is_within_window(event_date: datetime.date, today: datetime.date, window_trading_days: int) -> bool:
    if event_date < today:
        return False
    window_dates = set(pd.bdate_range(start=today, periods=window_trading_days).date)
    return event_date in window_dates


def fetch_upcoming_earnings_date(
    symbol: str, today: datetime.date, limit: int = 12
) -> datetime.date | None:
    """Nächstes noch nicht vergangenes Earnings-Datum für `symbol`, oder
    None (kein bekanntes bevorstehendes Datum, oder yfinance kennt das
    Symbol nicht / hat keinen Earnings-Kalender - z.B. ETFs, strukturierte
    Produkte). Wirft NIE - ein Fehler für ein einzelnes Symbol darf den
    Check für alle anderen Symbole nicht gefährden (siehe
    check_upcoming_earnings)."""
    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    except Exception:
        log.debug("Kein Earnings-Kalender für %s abrufbar.", symbol, exc_info=True)
        return None
    if df is None or df.empty:
        return None
    upcoming = sorted(ts.date() for ts in df.index if ts.date() >= today)
    return upcoming[0] if upcoming else None


def check_upcoming_earnings(
    symbols: list[str],
    today: datetime.date,
    window_trading_days: int = DEFAULT_EVENT_WINDOW_TRADING_DAYS,
    max_workers: int = DEFAULT_EARNINGS_FETCH_MAX_WORKERS,
) -> list[EarningsWarning]:
    """Holt die Earnings-Termine PARALLEL ab (ThreadPoolExecutor) - anders
    als yf.download() bietet yfinance für get_earnings_dates() keine
    gebündelte Mehr-Symbol-Abfrage; bei 100+ Symbolen wäre eine
    sequenzielle Abfrage (ein HTTP-Request pro Symbol) spürbar langsam.
    `max_workers` begrenzt die Parallelität bewusst (kein Grund, den
    Datenanbieter mit 100+ gleichzeitigen Verbindungen zu belasten)."""
    warnings: list[EarningsWarning] = []
    if not symbols:
        return warnings
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_upcoming_earnings_date, symbol, today): symbol for symbol in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                # fetch_upcoming_earnings_date faengt intern bereits alles ab
                # und wirft nie - dieses try/except ist eine zusaetzliche
                # Absicherung, falls das je verletzt wird (z.B. durch einen
                # kuenftigen Bug dort): ein einzelner fehlerhafter Future darf
                # den Earnings-Check fuer die uebrigen Symbole nicht mitreissen.
                earnings_date = future.result()
            except Exception:
                log.exception("Earnings-Abfrage für %s fehlgeschlagen - wird übersprungen.", symbol)
                continue
            if earnings_date is not None and _is_within_window(earnings_date, today, window_trading_days):
                warnings.append(
                    EarningsWarning(
                        symbol=symbol,
                        earnings_date=earnings_date,
                        trading_days_until=trading_days_until(today, earnings_date),
                    )
                )
    return sorted(warnings, key=lambda w: (w.trading_days_until, w.symbol))


def _upcoming_third_fridays(today: datetime.date, count_months: int = 2) -> list[tuple[str, datetime.date]]:
    """Dritte Freitage des aktuellen und der naechsten `count_months - 1`
    Monate, mit Name je nach QUARTERLY_WITCHING_MONTHS - genug Vorlauf, um
    jedes uebliche kleine Event-Fenster (DEFAULT_EVENT_WINDOW_TRADING_DAYS)
    sicher abzudecken. Ein bereits vergangener dritter Freitag DIESES Monats
    wird trotzdem mitgeliefert; `_is_within_window` unten filtert ihn ueber
    das `event_date >= today`-Kriterium korrekt heraus."""
    results = []
    year, month = today.year, today.month
    for _ in range(count_months):
        expiration = third_friday_of_month(year, month)
        name = "Hexensabbat (Options-Quartalsverfall)" if month in QUARTERLY_WITCHING_MONTHS else "Optionsverfall (monatlich)"
        results.append((name, expiration))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return results


def get_upcoming_macro_events(
    today: datetime.date,
    window_trading_days: int = DEFAULT_EVENT_WINDOW_TRADING_DAYS,
) -> list[MacroEvent]:
    """Portfolioweiter Hinweis (nicht symbolspezifisch) - siehe
    FOMC_DECISION_DATES/CPI_RELEASE_DATES sowie (2026-09-21)
    _upcoming_third_fridays/third_friday_of_month für die Options-
    Verfallstage."""
    events = []
    for name, dates in (("FOMC", FOMC_DECISION_DATES), ("CPI", CPI_RELEASE_DATES)):
        for event_date in dates:
            if _is_within_window(event_date, today, window_trading_days):
                events.append(
                    MacroEvent(name=name, event_date=event_date, trading_days_until=trading_days_until(today, event_date))
                )
    for name, expiration in _upcoming_third_fridays(today):
        if _is_within_window(expiration, today, window_trading_days):
            events.append(
                MacroEvent(name=name, event_date=expiration, trading_days_until=trading_days_until(today, expiration))
            )
    return sorted(events, key=lambda e: e.event_date)
