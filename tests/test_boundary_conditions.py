"""Tests for src/boundary_conditions.py's mechanical price-threshold check
(Thesis Kap. 7) - pure logic, no DB/network.
"""
from __future__ import annotations

from src.boundary_conditions import BoundaryConditionCheck, evaluate_boundary_conditions


def make_check(**overrides) -> BoundaryConditionCheck:
    defaults = dict(
        id=1, position_id=10, symbol="NVDA", description="test",
        check_type="price_below", threshold_price=150.0,
    )
    defaults.update(overrides)
    return BoundaryConditionCheck(**defaults)


def test_price_below_triggers_when_price_under_threshold():
    cond = make_check(check_type="price_below", threshold_price=150.0)
    triggered, still_open = evaluate_boundary_conditions([cond], {"NVDA": 149.0})
    assert triggered == [cond]
    assert still_open == []


def test_price_below_does_not_trigger_at_or_above_threshold():
    cond = make_check(check_type="price_below", threshold_price=150.0)
    triggered, still_open = evaluate_boundary_conditions([cond], {"NVDA": 150.0})
    assert triggered == []
    assert still_open == [cond]


def test_price_above_triggers_when_price_over_threshold():
    cond = make_check(check_type="price_above", threshold_price=250.0)
    triggered, still_open = evaluate_boundary_conditions([cond], {"NVDA": 251.0})
    assert triggered == [cond]
    assert still_open == []


def test_price_above_does_not_trigger_at_or_below_threshold():
    cond = make_check(check_type="price_above", threshold_price=250.0)
    triggered, still_open = evaluate_boundary_conditions([cond], {"NVDA": 250.0})
    assert triggered == []
    assert still_open == [cond]


def test_qualitative_never_triggers_automatically():
    """Kernaussage von Kap. 7: qualitative Randbedingungen werden NIE
    automatisch geprueft, unabhaengig von current_prices."""
    cond = make_check(check_type="qualitative", threshold_price=None)
    triggered, still_open = evaluate_boundary_conditions([cond], {"NVDA": 1.0})
    assert triggered == []
    assert still_open == [cond]


def test_missing_current_price_leaves_condition_open_not_triggered():
    """Fehlt der aktuelle Kurs fuer das Symbol in diesem Lauf, darf die
    Randbedingung NICHT stillschweigend als ausgeloest gelten."""
    cond = make_check(check_type="price_below", threshold_price=150.0)
    triggered, still_open = evaluate_boundary_conditions([cond], {})
    assert triggered == []
    assert still_open == [cond]


def test_evaluates_multiple_conditions_independently():
    below = make_check(id=1, symbol="NVDA", check_type="price_below", threshold_price=150.0)
    above = make_check(id=2, symbol="AMD", check_type="price_above", threshold_price=250.0)
    qualitative = make_check(id=3, symbol="CCJ", check_type="qualitative", threshold_price=None)

    triggered, still_open = evaluate_boundary_conditions(
        [below, above, qualitative],
        {"NVDA": 140.0, "AMD": 200.0},  # below triggers, above does not, CCJ has no price anyway
    )
    assert triggered == [below]
    assert still_open == [above, qualitative]


def test_empty_input_returns_empty_lists():
    assert evaluate_boundary_conditions([], {"NVDA": 100.0}) == ([], [])


# --- v18 (17-Punkte-Audit Fund #12): Basiswert-Kurs-Fallback -------------------


def test_falls_back_to_underlying_symbol_price():
    """Strukturiertes Produkt ohne eigenen Kurs, aber mit Kurs fuer seinen
    Basiswert - muss trotzdem geprueft werden, nicht permanent offen
    bleiben."""
    cond = make_check(
        symbol="MINI-NVDA-LONG-1", check_type="price_above", threshold_price=12.0,
        underlying_symbol="NVDA",
    )
    triggered, still_open = evaluate_boundary_conditions([cond], {"NVDA": 15.0})
    assert triggered == [cond]
    assert still_open == []


def test_uses_own_price_over_underlying_when_both_available():
    cond = make_check(
        symbol="MINI-NVDA-LONG-1", check_type="price_above", threshold_price=12.0,
        underlying_symbol="NVDA",
    )
    triggered, still_open = evaluate_boundary_conditions(
        [cond], {"MINI-NVDA-LONG-1": 10.0, "NVDA": 15.0}
    )
    assert triggered == []
    assert still_open == [cond]


def test_stays_open_when_neither_own_nor_underlying_price_available():
    cond = make_check(
        symbol="MINI-NVDA-LONG-1", check_type="price_above", threshold_price=12.0,
        underlying_symbol="NVDA",
    )
    triggered, still_open = evaluate_boundary_conditions([cond], {})
    assert triggered == []
    assert still_open == [cond]
