"""Tests for the Kap.-6.9-Erweiterung thematischer Segment-ETF-Korb
(SMH/URA/ICLN) - reine Kursdaten-Rekonstruktion, kein Netzwerkzugriff
(Preise werden über ein Fake-`price_on` injiziert, analog zum Pattern in
test_momentum_baseline.py/metrics.reconstruct_nav_history).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.segment_basket import SEGMENT_BASKET_SYMBOLS, reconstruct_segment_basket


def test_segment_basket_symbols_are_fixed():
    """Dokumentiert die bewusste, feste Wahl (Kap. 6.9: Halbleiter/Uran/Clean
    Energy) - kein frei konfigurierbares Universum wie bei der Momentum-
    Baseline."""
    assert SEGMENT_BASKET_SYMBOLS == ("SMH", "URA", "ICLN")


def _make_price_on(prices: dict[str, dict[str, float]]):
    """`prices[symbol]` is a dict of ISO-date-string -> close price. Mimics
    the `price_on` contract: latest known price at or before `when`."""
    series_by_symbol = {
        symbol: pd.Series(list(values.values()), index=pd.to_datetime(list(values.keys()))).sort_index()
        for symbol, values in prices.items()
    }

    def price_on(symbol: str, when: pd.Timestamp) -> float | None:
        series = series_by_symbol.get(symbol)
        if series is None:
            return None
        eligible = series[series.index <= when]
        return float(eligible.iloc[-1]) if not eligible.empty else None

    return price_on


def test_reconstruct_segment_basket_invests_equally_at_start():
    prices = {
        "SMH": {"2026-01-01": 100.0},
        "URA": {"2026-01-01": 100.0},
        "ICLN": {"2026-01-01": 100.0},
    }
    price_on = _make_price_on(prices)
    result = reconstruct_segment_basket(
        price_on=price_on, dates=[pd.Timestamp("2026-01-01")], start=pd.Timestamp("2026-01-01"),
        initial_cash=90_000.0,
    )
    assert result[0] == pytest.approx(90_000.0)


def test_reconstruct_segment_basket_tracks_blended_equal_weighted_return():
    """Gleichgewichtet (je 1/3) - der Korb muss der DURCHSCHNITTLICHEN
    Rendite der drei Symbole folgen, nicht der eines einzelnen."""
    prices = {
        "SMH": {"2026-01-01": 100.0, "2026-01-15": 150.0},   # +50%
        "URA": {"2026-01-01": 100.0, "2026-01-15": 110.0},   # +10%
        "ICLN": {"2026-01-01": 100.0, "2026-01-15": 90.0},   # -10%
    }
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]
    result = reconstruct_segment_basket(
        price_on=price_on, dates=dates, start=pd.Timestamp("2026-01-01"), initial_cash=90_000.0,
    )
    # Durchschnittliche Rendite = (50% + 10% - 10%) / 3 = 16.667%
    assert result[1] == pytest.approx(90_000.0 * (1 + (0.5 + 0.1 - 0.1) / 3))


def test_reconstruct_segment_basket_rebalances_monthly_to_equal_weight():
    """Kernanforderung 'monatlich rebalanciert': Gewichte, die zwischen zwei
    Rebalance-Terminen auseinandergelaufen sind (hier: SMH verdoppelt sich),
    muessen am naechsten Rebalance-Termin wieder auf 1/3 zurueckgesetzt
    werden - nicht die alten (verzerrten) Stueckzahlen fortschreiben."""
    prices = {
        "SMH": {"2026-01-01": 100.0, "2026-02-01": 200.0, "2026-02-15": 220.0},
        "URA": {"2026-01-01": 100.0, "2026-02-01": 100.0, "2026-02-15": 110.0},
        "ICLN": {"2026-01-01": 100.0, "2026-02-01": 100.0, "2026-02-15": 90.0},
    }
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-15")]
    result = reconstruct_segment_basket(
        price_on=price_on, dates=dates, start=pd.Timestamp("2026-01-01"), initial_cash=90_000.0,
    )

    value_at_feb1 = result[1]
    assert value_at_feb1 == pytest.approx(120_000.0)  # 30k*2.0 + 30k*1.0 + 30k*1.0

    # Nach Re-Equalisierung auf 40k/40k/40k: +10%/+10%/-10% -> Durchschnitt +3.33%
    expected_feb15 = value_at_feb1 * (1 + (0.1 + 0.1 - 0.1) / 3)
    assert result[2] == pytest.approx(expected_feb15)
    assert result[2] == pytest.approx(124_000.0)
    # Ohne Re-Equalisierung (alte, verzerrte Jan-1-Stueckzahlen fortgeschrieben)
    # waere das Ergebnis 126_000.0 - explizit ausschliessen.
    assert result[2] != pytest.approx(126_000.0)


def test_reconstruct_segment_basket_skips_symbol_missing_at_rebalance():
    """Ein Symbol ohne Kurs an einem Rebalance-Termin wird NUR fuer diesen
    Termin uebersprungen - die verbleibenden teilen sich den Wert."""
    prices = {
        "SMH": {"2026-01-01": 100.0, "2026-01-15": 120.0},
        "ICLN": {"2026-01-01": 100.0, "2026-01-15": 80.0},
        # URA hat keinen Kurs am Rebalance-Termin -> nur SMH/ICLN je 1/2.
    }
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]
    result = reconstruct_segment_basket(
        price_on=price_on, dates=dates, start=pd.Timestamp("2026-01-01"), initial_cash=90_000.0,
    )
    # 45k in SMH (+20%), 45k in ICLN (-20%) -> Durchschnitt 0%, Wert unveraendert.
    assert result[1] == pytest.approx(90_000.0)


def test_reconstruct_segment_basket_no_symbols_available_stays_flat_cash():
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]
    result = reconstruct_segment_basket(
        price_on=lambda symbol, when: None, dates=dates, start=pd.Timestamp("2026-01-01"), initial_cash=90_000.0,
    )
    assert result == [90_000.0, 90_000.0]


def test_reconstruct_segment_basket_empty_dates_returns_empty_list():
    assert reconstruct_segment_basket(
        price_on=lambda symbol, when: 100.0, dates=[], start=pd.Timestamp("2026-01-01"), initial_cash=90_000.0,
    ) == []


def test_reconstruct_segment_basket_respects_custom_symbols_override():
    """`symbols` ist ueberschreibbar (z.B. fuer Tests) - der Default bleibt
    aber SEGMENT_BASKET_SYMBOLS, siehe test_segment_basket_symbols_are_fixed."""
    prices = {"AAA": {"2026-01-01": 100.0, "2026-01-15": 200.0}}
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]
    result = reconstruct_segment_basket(
        price_on=price_on, dates=dates, start=pd.Timestamp("2026-01-01"), initial_cash=1_000.0,
        symbols=("AAA",),
    )
    assert result[1] == pytest.approx(2_000.0)
