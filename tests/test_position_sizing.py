"""Tests for src/position_sizing.py's volatilitätsadjustierte Positionsgrössen-
Skalierung und den (bewusst schwachen) Konviktions-Multiplikator. Pure
functions, no network/DB - Volatilitäten werden hier direkt als Eingabe-Dict
übergeben (die eigentliche Berechnung aus Kurshistorien läuft bereits in
data_fetch._annualized_volatility und wird dort getestet)."""
from __future__ import annotations

import pytest

from src.order_schema import ConvictionLevel
from src.position_sizing import (
    CONVICTION_SCALING_FACTORS,
    VolatilityScaling,
    compute_scaling_factors,
    conviction_scaling_factor,
    scale_order_size,
)

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
    qty, notional = scale_order_size(10.0, None, 1.5)
    assert qty == pytest.approx(15.0)
    assert notional is None


def test_scale_order_size_scales_notional():
    qty, notional = scale_order_size(None, 1000.0, 0.5)
    assert qty is None
    assert notional == pytest.approx(500.0)


def test_scale_order_size_leaves_none_fields_as_none():
    qty, notional = scale_order_size(None, None, 1.0)
    assert qty is None
    assert notional is None


def test_scale_order_size_composes_multiple_factors():
    """execution.py kombiniert Volatilitaets- und Konviktions-Faktor
    multiplikativ VOR einem einzigen scale_order_size-Aufruf."""
    combined = 1.2 * CONVICTION_SCALING_FACTORS[ConvictionLevel.HIGH]
    qty, _ = scale_order_size(10.0, None, combined)
    assert qty == pytest.approx(10.0 * 1.2 * 1.15)


# --- conviction_scaling_factor --------------------------------------------------


def test_conviction_scaling_factor_none_is_neutral():
    """Fehlende Konviktions-Angabe darf keine Skalierung ausloesen - siehe
    ProposedOrder.conviction's Optional-Semantik (order_schema.py)."""
    assert conviction_scaling_factor(None) == pytest.approx(1.0)


def test_conviction_scaling_factor_medium_is_neutral():
    assert conviction_scaling_factor(ConvictionLevel.MEDIUM) == pytest.approx(1.0)


def test_conviction_scaling_factor_high_scales_up_slightly():
    factor = conviction_scaling_factor(ConvictionLevel.HIGH)
    assert factor == pytest.approx(1.15)
    assert factor < 1.2  # bewusst schwach - siehe CONVICTION_SCALING_FACTORS' Kommentar


def test_conviction_scaling_factor_low_scales_down_slightly():
    factor = conviction_scaling_factor(ConvictionLevel.LOW)
    assert factor == pytest.approx(0.8)
    assert factor > 0.5  # deutlich enger als das Volatilitaets-Band (0.5x-1.5x)


def test_conviction_scaling_band_is_narrower_than_volatility_band():
    """Kernanforderung: die Konviktions-Selbsteinschaetzung eines LLM ist
    bekanntermassen schlecht kalibriert und darf die Positionsgroesse
    deshalb deutlich schwaecher beeinflussen als ein gemessenes
    Marktsignal wie die Volatilitaet (DEFAULT_MIN/MAX_SCALING_FACTOR)."""
    from src.position_sizing import DEFAULT_MAX_SCALING_FACTOR, DEFAULT_MIN_SCALING_FACTOR

    conviction_spread = CONVICTION_SCALING_FACTORS[ConvictionLevel.HIGH] - CONVICTION_SCALING_FACTORS[
        ConvictionLevel.LOW
    ]
    volatility_spread = DEFAULT_MAX_SCALING_FACTOR - DEFAULT_MIN_SCALING_FACTOR
    assert conviction_spread < volatility_spread
