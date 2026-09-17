"""Tests for src/order_schema.py's Kap.-7 BoundaryCondition addition to
ProposedOrder - optional/best-effort (no order is rejected for omitting it),
but internally consistent when provided (a price-based check_type needs a
threshold_price).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.order_schema import BoundaryCondition, ProposedOrder


def make_order(**overrides) -> ProposedOrder:
    defaults = dict(symbol="NVDA", instrument_type="equity", side="buy", quantity=10, rationale="test")
    defaults.update(overrides)
    return ProposedOrder(**defaults)


def test_boundary_conditions_default_to_empty_list():
    """Optional/best-effort (bestaetigt): eine Order ohne jede Randbedingung
    muss weiterhin gueltig sein."""
    order = make_order()
    assert order.boundary_conditions == []


def test_boundary_condition_qualitative_needs_no_threshold():
    bc = BoundaryCondition(description="Q3-Earnings enttäuschen", check_type="qualitative")
    assert bc.threshold_price is None


def test_boundary_condition_price_below_requires_threshold():
    with pytest.raises(ValidationError, match="threshold_price"):
        BoundaryCondition(description="Fällt unter Marke", check_type="price_below")


def test_boundary_condition_price_above_requires_threshold():
    with pytest.raises(ValidationError, match="threshold_price"):
        BoundaryCondition(description="Steigt über Marke", check_type="price_above")


def test_boundary_condition_price_below_with_threshold_is_valid():
    bc = BoundaryCondition(description="Fällt unter $150", check_type="price_below", threshold_price=150.0)
    assert bc.threshold_price == 150.0


def test_order_accepts_multiple_boundary_conditions():
    order = make_order(
        boundary_conditions=[
            {"description": "Fällt unter $150", "check_type": "price_below", "threshold_price": 150.0},
            {"description": "Fed pausiert Zinssenkungen", "check_type": "qualitative"},
        ]
    )
    assert len(order.boundary_conditions) == 2
    assert order.boundary_conditions[0].check_type.value == "price_below"
    assert order.boundary_conditions[1].check_type.value == "qualitative"


def test_boundary_condition_defaults_to_qualitative_when_check_type_omitted():
    bc = BoundaryCondition(description="etwas Vages")
    assert bc.check_type.value == "qualitative"
