"""Tests for src/stress_test.py's mechanical (contamination-free) backtest
logic (Thesis Kap. 11.2) - pure functions, no network/DB/LLM.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.stress_test import (
    PositionWeight,
    StressPeriod,
    compute_position_weights,
    compute_stress_period_result,
)


def _series(dates: list[str], prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=pd.to_datetime(dates))


# --- compute_position_weights -------------------------------------------------


def test_compute_position_weights_single_long_position():
    positions = [{"symbol": "NVDA", "side": "long", "quantity": 10.0}]
    weights = compute_position_weights(positions, {"NVDA": 100.0})
    assert weights == [PositionWeight(symbol="NVDA", weight=1.0)]


def test_compute_position_weights_signs_short_negative():
    positions = [
        {"symbol": "NVDA", "side": "long", "quantity": 10.0},   # 1000
        {"symbol": "TSDD", "side": "short", "quantity": 5.0},   # -500 (bei price=100)
    ]
    weights = compute_position_weights(positions, {"NVDA": 100.0, "TSDD": 100.0})
    by_symbol = {w.symbol: w.weight for w in weights}
    assert by_symbol["NVDA"] == pytest.approx(1000 / 1500)
    assert by_symbol["TSDD"] == pytest.approx(-500 / 1500)


def test_compute_position_weights_aggregates_same_symbol_both_sides():
    """Ein Symbol mit gleichzeitig offener Long- UND Short-Position (Hedge)
    muss zu einem einzigen Netto-Gewicht zusammengefasst werden."""
    positions = [
        {"symbol": "NVDA", "side": "long", "quantity": 10.0},   # +1000
        {"symbol": "NVDA", "side": "short", "quantity": 4.0},   # -400
    ]
    weights = compute_position_weights(positions, {"NVDA": 100.0})
    assert len(weights) == 1
    assert weights[0].symbol == "NVDA"
    assert weights[0].weight == pytest.approx(1.0)  # (1000-400)/|1000-400| normalisiert auf sich selbst


def test_compute_position_weights_skips_symbols_without_current_price():
    positions = [
        {"symbol": "NVDA", "side": "long", "quantity": 10.0},
        {"symbol": "UNKNOWN", "side": "long", "quantity": 5.0},
    ]
    weights = compute_position_weights(positions, {"NVDA": 100.0})
    assert [w.symbol for w in weights] == ["NVDA"]


def test_compute_position_weights_empty_when_no_prices_available():
    positions = [{"symbol": "NVDA", "side": "long", "quantity": 10.0}]
    assert compute_position_weights(positions, {}) == []


def test_compute_position_weights_empty_input():
    assert compute_position_weights([], {"NVDA": 100.0}) == []


# --- compute_stress_period_result ----------------------------------------------


def test_stress_period_single_long_position_tracks_drawdown():
    period = StressPeriod("Test-Krise", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))
    weights = [PositionWeight(symbol="NVDA", weight=1.0)]
    histories = {
        "NVDA": _series(
            ["2020-02-19", "2020-03-01", "2020-03-23"],
            [100.0, 80.0, 65.0],  # -35% peak-to-trough
        )
    }
    result = compute_stress_period_result(period, weights, histories)
    assert result.max_drawdown_pct == pytest.approx(-0.35)
    assert result.included_symbols == ["NVDA"]
    assert result.excluded_symbols_no_data == []


def test_stress_period_short_position_gains_when_price_falls():
    """Eine geshortete Position (negatives Gewicht) muss den Portfolio-Index
    bei fallendem Kurs steigen lassen, nicht fallen - kein Drawdown."""
    period = StressPeriod("Test-Krise", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))
    weights = [PositionWeight(symbol="TSDD", weight=-1.0)]
    histories = {"TSDD": _series(["2020-02-19", "2020-03-23"], [100.0, 65.0])}  # -35% Kurs -> +35% fuer den Short
    result = compute_stress_period_result(period, weights, histories)
    assert result.max_drawdown_pct == pytest.approx(0.0)  # Index steigt monoton, kein Drawdown


def test_stress_period_excludes_symbol_without_any_history():
    period = StressPeriod("GFC 2008", pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-09"))
    weights = [
        PositionWeight(symbol="ESTABLISHED", weight=0.5),
        PositionWeight(symbol="RECENT_IPO", weight=0.5),
    ]
    histories = {"ESTABLISHED": _series(["2008-09-01", "2009-03-09"], [100.0, 70.0])}
    result = compute_stress_period_result(period, weights, histories)
    assert result.included_symbols == ["ESTABLISHED"]
    assert result.excluded_symbols_no_data == ["RECENT_IPO"]
    # Nur ESTABLISHED zaehlt, auf 100% renormalisiert -> -30% Drawdown.
    assert result.max_drawdown_pct == pytest.approx(-0.30)


def test_stress_period_excludes_symbol_with_history_outside_window():
    period = StressPeriod("Test-Krise", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))
    weights = [PositionWeight(symbol="NVDA", weight=1.0)]
    # Historie existiert, aber komplett ausserhalb des Krisenfensters.
    histories = {"NVDA": _series(["2019-01-01", "2019-06-01"], [50.0, 60.0])}
    result = compute_stress_period_result(period, weights, histories)
    assert result.included_symbols == []
    assert result.excluded_symbols_no_data == ["NVDA"]
    assert result.max_drawdown_pct is None


def test_stress_period_returns_none_drawdown_when_all_symbols_excluded():
    period = StressPeriod("GFC 2008", pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-09"))
    weights = [PositionWeight(symbol="RECENT_IPO", weight=1.0)]
    result = compute_stress_period_result(period, weights, {})
    assert result.max_drawdown_pct is None
    assert result.included_symbols == []
    assert result.excluded_symbols_no_data == ["RECENT_IPO"]


def test_stress_period_two_positions_diversify_drawdown():
    period = StressPeriod("Test-Krise", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))
    weights = [
        PositionWeight(symbol="LOSER", weight=0.5),
        PositionWeight(symbol="STABLE", weight=0.5),
    ]
    histories = {
        "LOSER": _series(["2020-02-19", "2020-03-23"], [100.0, 50.0]),   # -50%
        "STABLE": _series(["2020-02-19", "2020-03-23"], [100.0, 100.0]),  # 0%
    }
    result = compute_stress_period_result(period, weights, histories)
    # Gewichteter Effekt: 0.5 * -50% + 0.5 * 0% = -25%
    assert result.max_drawdown_pct == pytest.approx(-0.25)
