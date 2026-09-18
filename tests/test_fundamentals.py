"""Tests for src/fundamentals.py's soft quality-score context (yfinance
Ticker.info, mocked - no real network). Explicitly NOT a filter/guardrail -
only that the data is correctly fetched/skipped and never crashes.
"""
from __future__ import annotations

from src.fundamentals import (
    FundamentalSnapshot,
    fetch_fundamentals,
    fetch_universe_fundamentals,
)


class FakeTicker:
    def __init__(self, info: dict):
        self.info = info


# --- fetch_fundamentals -------------------------------------------------------


def test_fetch_fundamentals_returns_snapshot_when_all_fields_present(monkeypatch):
    info = {"revenueGrowth": 0.164, "debtToEquity": 78.445, "freeCashflow": 107_721_875_456}
    monkeypatch.setattr("src.fundamentals.yf.Ticker", lambda symbol: FakeTicker(info))
    result = fetch_fundamentals("AAPL")
    assert result == FundamentalSnapshot(
        symbol="AAPL", revenue_growth=0.164, debt_to_equity=78.445, free_cash_flow=107_721_875_456
    )


def test_fetch_fundamentals_keeps_partial_data_with_missing_fields_as_none(monkeypatch):
    """'Wo verfuegbar' gilt pro Feld, nicht nur pro Symbol - ein einzelnes
    fehlendes Feld darf die uebrigen verfuegbaren Werte nicht verwerfen."""
    info = {"revenueGrowth": 0.05, "debtToEquity": None, "freeCashflow": None}
    monkeypatch.setattr("src.fundamentals.yf.Ticker", lambda symbol: FakeTicker(info))
    result = fetch_fundamentals("SOME_SYMBOL")
    assert result.revenue_growth == 0.05
    assert result.debt_to_equity is None
    assert result.free_cash_flow is None


def test_fetch_fundamentals_none_when_all_fields_missing(monkeypatch):
    """z.B. ETFs wie NVDL/TSDD oder strukturierte Produkte - kein Fehler,
    einfach kein Eintrag."""
    info = {"revenueGrowth": None, "debtToEquity": None, "freeCashflow": None}
    monkeypatch.setattr("src.fundamentals.yf.Ticker", lambda symbol: FakeTicker(info))
    assert fetch_fundamentals("NVDL") is None


def test_fetch_fundamentals_none_when_fields_absent_from_info(monkeypatch):
    """Sehr junger Boersengang o.ae. - Felder fehlen im info-Dict komplett
    (nicht nur None), muss identisch behandelt werden."""
    monkeypatch.setattr("src.fundamentals.yf.Ticker", lambda symbol: FakeTicker({}))
    assert fetch_fundamentals("RECENT_IPO") is None


def test_fetch_fundamentals_never_raises_on_ticker_error(monkeypatch):
    class BoomTicker:
        @property
        def info(self):
            raise RuntimeError("yfinance down")

    monkeypatch.setattr("src.fundamentals.yf.Ticker", lambda symbol: BoomTicker())
    assert fetch_fundamentals("AAPL") is None


# --- fetch_universe_fundamentals (Parallelitaet + Fehlertoleranz) ----------


def test_fetch_universe_fundamentals_collects_available_symbols(monkeypatch):
    def fake_fetch(symbol):
        data = {
            "AAPL": FundamentalSnapshot("AAPL", 0.1, 50.0, 1_000_000.0),
            "NVDL": None,  # ETF, keine Daten
        }
        return data.get(symbol)

    monkeypatch.setattr("src.fundamentals.fetch_fundamentals", fake_fetch)
    result = fetch_universe_fundamentals(["AAPL", "NVDL"])
    assert list(result.keys()) == ["AAPL"]
    assert result["AAPL"].revenue_growth == 0.1


def test_fetch_universe_fundamentals_empty_symbol_list():
    assert fetch_universe_fundamentals([]) == {}


def test_fetch_universe_fundamentals_no_symbols_have_data(monkeypatch):
    monkeypatch.setattr("src.fundamentals.fetch_fundamentals", lambda symbol: None)
    assert fetch_universe_fundamentals(["NVDL", "TSDD"]) == {}


def test_fetch_universe_fundamentals_one_symbol_failing_does_not_affect_others(monkeypatch):
    """Verteidigt gegen eine verletzte Invariante: selbst wenn
    fetch_fundamentals entgegen seiner eigenen Zusicherung fuer ein Symbol
    doch einmal wirft, darf der Abruf fuer die uebrigen Symbole nicht
    abstuerzen (analoge Absicherung wie in event_calendar.py)."""
    def fake_fetch(symbol):
        if symbol == "BROKEN":
            raise RuntimeError("simuliert eine verletzte 'wirft nie'-Zusicherung")
        return {"GOOD": FundamentalSnapshot("GOOD", 0.2, 10.0, 500.0)}.get(symbol)

    monkeypatch.setattr("src.fundamentals.fetch_fundamentals", fake_fetch)
    result = fetch_universe_fundamentals(["BROKEN", "GOOD"])
    assert list(result.keys()) == ["GOOD"]
