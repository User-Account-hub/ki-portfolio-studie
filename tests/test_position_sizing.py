"""Tests for src/position_sizing.py's volatilitätsadjustierte Positionsgrössen-
Skalierung. Pure functions, no network/DB - Volatilitäten werden hier direkt
als Eingabe-Dict übergeben (die eigentliche Berechnung aus Kurshistorien
läuft bereits in data_fetch._annualized_volatility und wird dort getestet)."""
from __future__ import annotations

import pytest

from src.position_sizing import VolatilityScaling, compute_scaling_factors, scale_order_size

# --- compute_scaling_factors -------------------------------------------------


def test_average_volatility_symbol_gets_factor_one():
    volatilities = {"A": 0.20, "B": 0.20, "C": 0.20}
    factors = compute_scaling_factors(volatilities)
    for scaling in factors.values():
        assert scaling.scaling_factor == pytest.approx(1.0)
        assert scaling.universe_avg_volatility == pytest.approx(0.20)


def test_below_average_volatility_scales_up():
    volatilities = {"LOW": 0.10, "HIGH": 0.30}
    factors = compute_scaling_factors(volatilities)
    # avg = 0.20; LOW: 0.20/0.10 = 2.0 -> geclippt auf max_factor (1.5)
    assert factors["LOW"].scaling_factor == pytest.approx(1.5)
    # HIGH: 0.20/0.30 = 0.667
    assert factors["HIGH"].scaling_factor == pytest.approx(0.20 / 0.30)


def test_above_average_volatility_scales_down():
    volatilities = {"CALM": 0.10, "WILD": 1.00}
    factors = compute_scaling_factors(volatilities)
    assert factors["WILD"].scaling_factor < 1.0
    assert factors["CALM"].scaling_factor > 1.0


def test_scaling_factor_clipped_to_configured_band():
    volatilities = {"ULTRA_LOW": 0.01, "ULTRA_HIGH": 5.0, "NORMAL": 0.15}
    factors = compute_scaling_factors(volatilities, min_factor=0.5, max_factor=1.5)
    assert factors["ULTRA_LOW"].scaling_factor == pytest.approx(1.5)
    assert factors["ULTRA_HIGH"].scaling_factor == pytest.approx(0.5)


def test_custom_scaling_band_is_respected():
    volatilities = {"LOW": 0.05, "HIGH": 0.50}
    factors = compute_scaling_factors(volatilities, min_factor=0.8, max_factor=1.2)
    assert factors["LOW"].scaling_factor == pytest.approx(1.2)
    assert factors["HIGH"].scaling_factor == pytest.approx(0.8)


def test_zero_or_negative_volatility_is_excluded_not_divided_by():
    volatilities = {"ZERO_VOL": 0.0, "NORMAL": 0.20}
    factors = compute_scaling_factors(volatilities)
    assert "ZERO_VOL" not in factors
    assert "NORMAL" in factors


def test_none_volatility_entries_are_excluded():
    volatilities = {"NO_DATA": None, "NORMAL": 0.20}
    factors = compute_scaling_factors(volatilities)
    assert "NO_DATA" not in factors


def test_empty_volatilities_returns_empty_dict():
    assert compute_scaling_factors({}) == {}


def test_single_symbol_gets_factor_one():
    factors = compute_scaling_factors({"ONLY": 0.42})
    assert factors["ONLY"].scaling_factor == pytest.approx(1.0)


# --- scale_order_size ---------------------------------------------------------


def test_scale_order_size_scales_quantity():
    scaling = VolatilityScaling("A", annualized_volatility=0.1, universe_avg_volatility=0.2, scaling_factor=1.5)
    qty, notional = scale_order_size(10.0, None, scaling)
    assert qty == pytest.approx(15.0)
    assert notional is None


def test_scale_order_size_scales_notional():
    scaling = VolatilityScaling("A", annualized_volatility=0.4, universe_avg_volatility=0.2, scaling_factor=0.5)
    qty, notional = scale_order_size(None, 1000.0, scaling)
    assert qty is None
    assert notional == pytest.approx(500.0)


def test_scale_order_size_leaves_none_fields_as_none():
    scaling = VolatilityScaling("A", annualized_volatility=0.2, universe_avg_volatility=0.2, scaling_factor=1.0)
    qty, notional = scale_order_size(None, None, scaling)
    assert qty is None
    assert notional is None
