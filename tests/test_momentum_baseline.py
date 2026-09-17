"""Tests for the Kap.-6.9 regelbasierte Momentum-Baseline (reine
Kursdaten-Rekonstruktion, kein Netzwerkzugriff - Preise werden ueber ein
Fake-`price_on` injiziert, analog zum Pattern in metrics.reconstruct_nav_history).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.momentum_baseline import (
    monthly_rebalance_dates,
    reconstruct_momentum_baseline,
    select_top_quintile,
)


def test_select_top_quintile_picks_highest_returns():
    returns = {"A": 0.10, "B": 0.50, "C": -0.05, "D": 0.30, "E": 0.02}
    selected = select_top_quintile(returns, top_fraction=0.4)  # top 2 of 5
    assert selected == ["B", "D"]


def test_select_top_quintile_at_least_one_symbol():
    returns = {"A": 0.01, "B": -0.02, "C": 0.03}
    selected = select_top_quintile(returns, top_fraction=0.2)  # round(3*0.2)=1
    assert selected == ["C"]


def test_select_top_quintile_empty_returns_empty():
    assert select_top_quintile({}, top_fraction=0.2) == []


def test_monthly_rebalance_dates_includes_start_and_month_starts():
    start = pd.Timestamp("2026-09-21")
    end = pd.Timestamp("2026-11-30")
    dates = monthly_rebalance_dates(start, end)
    assert dates == [
        pd.Timestamp("2026-09-21"),
        pd.Timestamp("2026-10-01"),
        pd.Timestamp("2026-11-01"),
    ]


def test_monthly_rebalance_dates_no_duplicate_when_start_is_month_start():
    start = pd.Timestamp("2026-10-01")
    end = pd.Timestamp("2026-10-15")
    assert monthly_rebalance_dates(start, end) == [pd.Timestamp("2026-10-01")]


def test_monthly_rebalance_dates_single_when_end_before_start():
    start = pd.Timestamp("2026-09-21")
    end = pd.Timestamp("2026-09-01")
    assert monthly_rebalance_dates(start, end) == [start]


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


def test_reconstruct_momentum_baseline_selects_top_performer_and_tracks_its_price():
    """Kleines Universum aus 5 Symbolen (Quintil = 1 Symbol), keine
    Rebalance-Aenderung noetig (nur ein Zeitraum) - die Baseline muss exakt
    der Kursentwicklung des einzigen Top-Performers folgen."""
    start = pd.Timestamp("2026-01-01")
    lookback_start = (start - pd.Timedelta(weeks=12)).date().isoformat()
    prices = {
        "WINNER": {lookback_start: 100.0, "2026-01-01": 150.0, "2026-01-15": 180.0},
        "LOSER1": {lookback_start: 100.0, "2026-01-01": 90.0, "2026-01-15": 85.0},
        "LOSER2": {lookback_start: 100.0, "2026-01-01": 95.0, "2026-01-15": 92.0},
        "LOSER3": {lookback_start: 100.0, "2026-01-01": 98.0, "2026-01-15": 97.0},
        "LOSER4": {lookback_start: 100.0, "2026-01-01": 99.0, "2026-01-15": 99.5},
    }
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]

    result = reconstruct_momentum_baseline(
        price_on=price_on,
        universe_symbols=list(prices),
        dates=dates,
        start=start,
        initial_cash=100_000.0,
        top_quintile_fraction=0.2,
    )

    assert result[0] == pytest.approx(100_000.0)  # voll investiert am Rebalance-Tag selbst
    assert result[1] == pytest.approx(100_000.0 * (180.0 / 150.0))


def test_reconstruct_momentum_baseline_rebalances_monthly_into_new_winner():
    """Nach dem zweiten Rebalance (1. Feb) muss die Baseline auf den dann
    fuehrenden Performer wechseln, nicht dem alten Gewinner treu bleiben."""
    start = pd.Timestamp("2026-01-01")
    lookback_start = (start - pd.Timedelta(weeks=12)).date().isoformat()
    second_lookback = (pd.Timestamp("2026-02-01") - pd.Timedelta(weeks=12)).date().isoformat()
    prices = {
        # A fuehrt bis Ende Januar, dann seitwaerts.
        "A": {
            lookback_start: 100.0, "2026-01-01": 150.0,
            second_lookback: 145.0, "2026-02-01": 150.0, "2026-02-15": 151.0,
        },
        # B liegt anfangs zurueck, gewinnt aber im Februar deutlich (12-Wochen-
        # Rendite zum Feb-1-Rebalance: B +50% vs. A +3.4%).
        "B": {
            lookback_start: 100.0, "2026-01-01": 90.0,
            second_lookback: 60.0, "2026-02-01": 90.0, "2026-02-15": 180.0,
        },
    }
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-15")]

    result = reconstruct_momentum_baseline(
        price_on=price_on,
        universe_symbols=list(prices),
        dates=dates,
        start=start,
        initial_cash=100_000.0,
        top_quintile_fraction=0.5,  # Quintil-Fraktion so gewaehlt, dass bei 2 Symbolen genau 1 gewinnt
    )

    value_at_feb1 = result[1]
    # Zwischen Feb-Rebalance und Feb-15 muss die Rendite der von B stammen
    # (dem am 1. Feb neu fuehrenden Symbol), nicht von A.
    expected_feb15 = value_at_feb1 * (180.0 / 90.0)
    assert result[2] == pytest.approx(expected_feb15)


def test_reconstruct_momentum_baseline_excludes_symbol_without_enough_history():
    """Ein Symbol ohne Kurs vor `lookback_weeks` (z.B. juengerer Datenbeginn)
    darf an diesem Rebalance-Termin nicht in die Rangliste aufgenommen werden,
    selbst wenn es sonst die hoechste (unberechenbare) Rendite haette."""
    start = pd.Timestamp("2026-01-01")
    lookback_start = (start - pd.Timedelta(weeks=12)).date().isoformat()
    prices = {
        "ESTABLISHED": {lookback_start: 100.0, "2026-01-01": 110.0, "2026-01-15": 120.0},
        # Keine Kurshistorie vor dem Rebalance-Termin -> darf nicht gewaehlt werden.
        "TOO_NEW": {"2026-01-01": 50.0, "2026-01-15": 500.0},
    }
    price_on = _make_price_on(prices)
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]

    result = reconstruct_momentum_baseline(
        price_on=price_on,
        universe_symbols=list(prices),
        dates=dates,
        start=start,
        initial_cash=100_000.0,
        top_quintile_fraction=0.5,
    )

    assert result[1] == pytest.approx(100_000.0 * (120.0 / 110.0))


def test_reconstruct_momentum_baseline_empty_universe_stays_flat_cash():
    dates = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15")]
    result = reconstruct_momentum_baseline(
        price_on=lambda symbol, when: None,
        universe_symbols=[],
        dates=dates,
        start=pd.Timestamp("2026-01-01"),
        initial_cash=100_000.0,
    )
    assert result == [100_000.0, 100_000.0]
