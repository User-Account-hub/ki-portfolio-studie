"""Tests for src/event_calendar.py's earnings-date lookup (yfinance,
mocked - no real network) and hardcoded FOMC/CPI macro events. Purely
informational (Prompt-Kontext), no guardrail/veto behavior to test there -
only that the data is correctly computed and never crashes.
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest

from src.event_calendar import (
    CPI_RELEASE_DATES,
    FOMC_DECISION_DATES,
    QUARTERLY_WITCHING_MONTHS,
    EarningsWarning,
    MacroEvent,
    _upcoming_third_fridays,
    check_upcoming_earnings,
    fetch_upcoming_earnings_date,
    get_upcoming_macro_events,
    third_friday_of_month,
    trading_days_until,
)


# --- trading_days_until -----------------------------------------------------


def test_trading_days_until_same_day_is_zero():
    today = datetime.date(2026, 9, 18)  # Freitag
    assert trading_days_until(today, today) == 0


def test_trading_days_until_next_business_day():
    friday = datetime.date(2026, 9, 18)
    monday = datetime.date(2026, 9, 21)
    assert trading_days_until(friday, monday) == 1


def test_trading_days_until_counts_only_business_days():
    monday = datetime.date(2026, 9, 21)
    wednesday = datetime.date(2026, 9, 23)
    assert trading_days_until(monday, wednesday) == 2


# --- fetch_upcoming_earnings_date (mocked yf.Ticker) ------------------------


class FakeTicker:
    def __init__(self, earnings_df):
        self._earnings_df = earnings_df

    def get_earnings_dates(self, limit=12):
        return self._earnings_df


def _earnings_df(dates: list[str]) -> pd.DataFrame:
    index = pd.to_datetime(dates).tz_localize("America/New_York")
    return pd.DataFrame({"EPS Estimate": [1.0] * len(dates)}, index=index)


def test_fetch_upcoming_earnings_date_returns_next_future_date(monkeypatch):
    today = datetime.date(2026, 9, 18)
    df = _earnings_df(["2026-07-30", "2026-10-29", "2027-01-28"])  # gemischt vergangen/zukuenftig
    monkeypatch.setattr(
        "src.event_calendar.yf.Ticker", lambda symbol: FakeTicker(df)
    )
    result = fetch_upcoming_earnings_date("AAPL", today)
    assert result == datetime.date(2026, 10, 29)


def test_fetch_upcoming_earnings_date_none_when_all_dates_are_past(monkeypatch):
    today = datetime.date(2026, 9, 18)
    df = _earnings_df(["2026-07-30", "2026-04-30"])
    monkeypatch.setattr("src.event_calendar.yf.Ticker", lambda symbol: FakeTicker(df))
    assert fetch_upcoming_earnings_date("AAPL", today) is None


def test_fetch_upcoming_earnings_date_none_on_empty_dataframe(monkeypatch):
    today = datetime.date(2026, 9, 18)
    monkeypatch.setattr("src.event_calendar.yf.Ticker", lambda symbol: FakeTicker(pd.DataFrame()))
    assert fetch_upcoming_earnings_date("SPY", today) is None


def test_fetch_upcoming_earnings_date_none_when_dataframe_is_none(monkeypatch):
    today = datetime.date(2026, 9, 18)
    monkeypatch.setattr("src.event_calendar.yf.Ticker", lambda symbol: FakeTicker(None))
    assert fetch_upcoming_earnings_date("SPY", today) is None


def test_fetch_upcoming_earnings_date_never_raises_on_ticker_error(monkeypatch):
    today = datetime.date(2026, 9, 18)

    class BoomTicker:
        def get_earnings_dates(self, limit=12):
            raise RuntimeError("yfinance down")

    monkeypatch.setattr("src.event_calendar.yf.Ticker", lambda symbol: BoomTicker())
    assert fetch_upcoming_earnings_date("AAPL", today) is None


# --- check_upcoming_earnings (Fenster-Logik + Parallelitaet) ----------------


def test_check_upcoming_earnings_flags_symbol_within_window(monkeypatch):
    today = datetime.date(2026, 9, 18)  # Freitag

    def fake_fetch(symbol, today_arg, limit=12):
        return {"NVDA": datetime.date(2026, 9, 21), "AMD": datetime.date(2026, 11, 1)}.get(symbol)

    monkeypatch.setattr("src.event_calendar.fetch_upcoming_earnings_date", fake_fetch)
    warnings = check_upcoming_earnings(["NVDA", "AMD"], today, window_trading_days=3)
    assert warnings == [EarningsWarning(symbol="NVDA", earnings_date=datetime.date(2026, 9, 21), trading_days_until=1)]


def test_check_upcoming_earnings_no_warning_when_no_symbols_have_earnings(monkeypatch):
    monkeypatch.setattr("src.event_calendar.fetch_upcoming_earnings_date", lambda *a, **kw: None)
    assert check_upcoming_earnings(["AAPL", "SPY"], datetime.date(2026, 9, 18)) == []


def test_check_upcoming_earnings_empty_symbol_list():
    assert check_upcoming_earnings([], datetime.date(2026, 9, 18)) == []


def test_check_upcoming_earnings_sorted_by_urgency(monkeypatch):
    today = datetime.date(2026, 9, 18)  # Freitag - 3-Tage-Fenster deckt Fr/Mo/Di ab

    def fake_fetch(symbol, today_arg, limit=12):
        return {
            "LATER": datetime.date(2026, 9, 22),  # Dienstag, 2 Handelstage
            "SOONER": datetime.date(2026, 9, 21),  # Montag, 1 Handelstag
        }.get(symbol)

    monkeypatch.setattr("src.event_calendar.fetch_upcoming_earnings_date", fake_fetch)
    warnings = check_upcoming_earnings(["LATER", "SOONER"], today, window_trading_days=3)
    assert [w.symbol for w in warnings] == ["SOONER", "LATER"]


def test_check_upcoming_earnings_one_symbol_failing_does_not_affect_others(monkeypatch):
    """Verteidigt gegen eine verletzte Invariante: selbst wenn
    fetch_upcoming_earnings_date entgegen seiner eigenen Zusicherung fuer
    ein Symbol doch einmal wirft (z.B. ein kuenftiger Bug dort), darf der
    Check fuer die uebrigen Symbole nicht abstuerzen."""
    today = datetime.date(2026, 9, 18)

    def fake_fetch(symbol, today_arg, limit=12):
        if symbol == "BROKEN":
            raise RuntimeError("simuliert eine verletzte 'wirft nie'-Zusicherung")
        return {"GOOD": datetime.date(2026, 9, 21)}.get(symbol)

    monkeypatch.setattr("src.event_calendar.fetch_upcoming_earnings_date", fake_fetch)
    warnings = check_upcoming_earnings(["BROKEN", "GOOD"], today, window_trading_days=3)
    assert [w.symbol for w in warnings] == ["GOOD"]


# --- get_upcoming_macro_events ----------------------------------------------


def test_get_upcoming_macro_events_flags_fomc_within_window():
    # FOMC-Termin 2026-10-28 - 3 Handelstage vorher waere 2026-10-23 (Freitag).
    today = datetime.date(2026, 10, 26)  # Montag, 2 Handelstage vor 2026-10-28
    events = get_upcoming_macro_events(today, window_trading_days=3)
    assert MacroEvent(name="FOMC", event_date=datetime.date(2026, 10, 28), trading_days_until=2) in events


def test_get_upcoming_macro_events_flags_cpi_within_window():
    today = datetime.date(2026, 10, 13)  # Dienstag, CPI am 2026-10-14 (Mittwoch)
    events = get_upcoming_macro_events(today, window_trading_days=3)
    assert MacroEvent(name="CPI", event_date=datetime.date(2026, 10, 14), trading_days_until=1) in events


def test_get_upcoming_macro_events_empty_far_from_any_event():
    # Mitte September, weit weg vom naechsten FOMC (16.09., bereits vorbei)
    # und der naechsten CPI (14.10.).
    today = datetime.date(2026, 9, 22)
    assert get_upcoming_macro_events(today, window_trading_days=3) == []


def test_get_upcoming_macro_events_sorted_by_date():
    # Ein Zeitpunkt, an dem sowohl CPI (2026-01-13) als auch kein FOMC im Fenster liegen -
    # nur ueberpruefen, dass die generelle Sortierung nach Datum haelt.
    today = datetime.date(2026, 1, 12)
    events = get_upcoming_macro_events(today, window_trading_days=3)
    dates = [e.event_date for e in events]
    assert dates == sorted(dates)


def test_fomc_and_cpi_dates_are_hardcoded_and_nonempty():
    """Regressionsschutz: die hardcodierten Terminlisten muessen tatsaechlich
    befuellt sein (kein versehentlich leeres Modul-Level-Attribut)."""
    assert len(FOMC_DECISION_DATES) >= 8
    assert len(CPI_RELEASE_DATES) >= 12
    assert all(isinstance(d, datetime.date) for d in FOMC_DECISION_DATES)
    assert all(isinstance(d, datetime.date) for d in CPI_RELEASE_DATES)


# --- third_friday_of_month / Options-Verfallstage (2026-09-21) ---------------


def test_third_friday_of_month_known_dates():
    assert third_friday_of_month(2026, 1) == datetime.date(2026, 1, 16)
    assert third_friday_of_month(2026, 3) == datetime.date(2026, 3, 20)
    assert third_friday_of_month(2026, 9) == datetime.date(2026, 9, 18)
    assert third_friday_of_month(2026, 10) == datetime.date(2026, 10, 16)


def test_third_friday_of_month_is_always_a_friday():
    for month in range(1, 13):
        assert third_friday_of_month(2026, month).strftime("%A") == "Friday"


def test_third_friday_of_month_handles_year_rollover():
    assert third_friday_of_month(2027, 1) == datetime.date(2027, 1, 15)


def test_quarterly_witching_months_are_march_june_september_december():
    assert QUARTERLY_WITCHING_MONTHS == {3, 6, 9, 12}


def test_upcoming_third_fridays_labels_quarterly_month_as_hexensabbat():
    today = datetime.date(2026, 9, 1)  # September ist ein Hexensabbat-Monat
    results = _upcoming_third_fridays(today, count_months=1)
    assert results == [("Hexensabbat (Options-Quartalsverfall)", datetime.date(2026, 9, 18))]


def test_upcoming_third_fridays_labels_regular_month_as_monatlich():
    today = datetime.date(2026, 10, 1)  # Oktober ist KEIN Hexensabbat-Monat
    results = _upcoming_third_fridays(today, count_months=1)
    assert results == [("Optionsverfall (monatlich)", datetime.date(2026, 10, 16))]


def test_upcoming_third_fridays_covers_multiple_months_including_year_rollover():
    today = datetime.date(2026, 12, 1)
    results = _upcoming_third_fridays(today, count_months=2)
    assert results == [
        ("Hexensabbat (Options-Quartalsverfall)", datetime.date(2026, 12, 18)),
        ("Optionsverfall (monatlich)", datetime.date(2027, 1, 15)),
    ]


def test_get_upcoming_macro_events_flags_quarterly_witching_within_window():
    today = datetime.date(2026, 9, 16)  # Mittwoch, Hexensabbat am 2026-09-18 (Freitag)
    events = get_upcoming_macro_events(today, window_trading_days=3)
    assert MacroEvent(
        name="Hexensabbat (Options-Quartalsverfall)", event_date=datetime.date(2026, 9, 18), trading_days_until=2
    ) in events


def test_get_upcoming_macro_events_flags_regular_monthly_expiration_within_window():
    today = datetime.date(2026, 10, 14)  # Mittwoch, Verfall am 2026-10-16 (Freitag)
    events = get_upcoming_macro_events(today, window_trading_days=3)
    assert MacroEvent(
        name="Optionsverfall (monatlich)", event_date=datetime.date(2026, 10, 16), trading_days_until=2
    ) in events


def test_get_upcoming_macro_events_ignores_expiration_outside_window():
    today = datetime.date(2026, 9, 22)  # Hexensabbat (18.09.) bereits vorbei, naechster Verfall (16.10.) weit weg
    events = get_upcoming_macro_events(today, window_trading_days=3)
    assert not any("Verfall" in e.name or "Hexensabbat" in e.name for e in events)


def test_get_upcoming_macro_events_does_not_duplicate_fomc_or_cpi_with_expiration():
    """CPI-Termine liegen nicht zufaellig auf einem dritten Freitag - beide
    Ereignis-Arten muessen unabhaengig nebeneinander erscheinen koennen,
    ohne sich zu ueberschreiben."""
    today = datetime.date(2026, 10, 14)  # CPI (14.10., heute) UND Options-Verfall (16.10.) beide im 3-Tage-Fenster
    events = get_upcoming_macro_events(today, window_trading_days=3)
    names = {e.name for e in events}
    assert "CPI" in names
    assert "Optionsverfall (monatlich)" in names
