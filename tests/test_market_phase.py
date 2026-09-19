"""Tests for src/market_phase.py's regelbasierte Bull/Bear/Seitwärts-
Klassifikation und den Abgleich gegen Claudes optionale cycle_position.
Pure functions, no network/DB.
"""
from __future__ import annotations

import pytest

from src.data_fetch import MarketSnapshot
from src.market_phase import (
    MarketPhase,
    MarketPhaseContradiction,
    check_cycle_position_against_market_phase,
    classify_market_phase,
    classify_universe_market_phases,
)
from src.order_schema import CyclePosition

# --- classify_market_phase ----------------------------------------------------


def test_classify_bull_when_price_and_sma20_above_sma50():
    phase = classify_market_phase(price=120.0, sma20=115.0, sma50=100.0, volatility_20d_annualized=0.20)
    assert phase == MarketPhase.BULL


def test_classify_bear_when_price_and_sma20_below_sma50():
    phase = classify_market_phase(price=80.0, sma20=85.0, sma50=100.0, volatility_20d_annualized=0.20)
    assert phase == MarketPhase.BEAR


def test_classify_sideways_when_price_within_volatility_band_of_sma50():
    # Band = max(0.03, 0.5 * 0.20) = 0.10 -> Abweichung von 5% liegt innerhalb.
    phase = classify_market_phase(price=105.0, sma20=104.0, sma50=100.0, volatility_20d_annualized=0.20)
    assert phase == MarketPhase.SIDEWAYS


def test_classify_sideways_on_mixed_signal_outside_band():
    """Kurs > sma50, aber sma20 < sma50 (moeglicher Trendwechsel im Gange) -
    konservativ Seitwaerts statt geraten, auch ausserhalb der Bandbreite."""
    phase = classify_market_phase(price=130.0, sma20=95.0, sma50=100.0, volatility_20d_annualized=0.05)
    assert phase == MarketPhase.SIDEWAYS


def test_classify_uses_minimum_band_when_volatility_missing():
    # Ohne Vola-Angabe greift die Mindestbandbreite (3%): 2% Abweichung -> Seitwaerts.
    phase = classify_market_phase(price=102.0, sma20=101.0, sma50=100.0, volatility_20d_annualized=None)
    assert phase == MarketPhase.SIDEWAYS


def test_classify_bull_outside_minimum_band_without_volatility():
    phase = classify_market_phase(price=110.0, sma20=106.0, sma50=100.0, volatility_20d_annualized=None)
    assert phase == MarketPhase.BULL


def test_classify_higher_volatility_widens_sideways_band():
    """Dieselbe 12%-Abweichung ist bei niedriger Vola ein Trend, bei hoher
    Vola noch Rauschen - derselbe Sachverhalt, unterschiedliche Klassifikation."""
    low_vol_phase = classify_market_phase(price=112.0, sma20=111.0, sma50=100.0, volatility_20d_annualized=0.10)
    high_vol_phase = classify_market_phase(price=112.0, sma20=111.0, sma50=100.0, volatility_20d_annualized=0.60)
    assert low_vol_phase == MarketPhase.BULL
    assert high_vol_phase == MarketPhase.SIDEWAYS


def test_classify_unknown_without_sma20():
    assert classify_market_phase(price=100.0, sma20=None, sma50=100.0, volatility_20d_annualized=0.2) == (
        MarketPhase.UNKNOWN
    )


def test_classify_unknown_without_sma50():
    assert classify_market_phase(price=100.0, sma20=100.0, sma50=None, volatility_20d_annualized=0.2) == (
        MarketPhase.UNKNOWN
    )


def test_classify_unknown_when_sma50_is_zero():
    assert classify_market_phase(price=100.0, sma20=100.0, sma50=0.0, volatility_20d_annualized=0.2) == (
        MarketPhase.UNKNOWN
    )


# --- classify_universe_market_phases -------------------------------------------


def make_snapshot(**overrides) -> MarketSnapshot:
    defaults = dict(
        symbol="AAPL", last_price=100.0, change_1d_pct=0.0, sma20=100.0, sma50=100.0,
        volatility_20d_annualized=0.2, volume=1000,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


def test_classify_universe_maps_every_symbol():
    snapshots = {
        "AAPL": make_snapshot(symbol="AAPL", last_price=120.0, sma20=115.0, sma50=100.0),
        "TSLA": make_snapshot(symbol="TSLA", last_price=80.0, sma20=85.0, sma50=100.0),
    }
    result = classify_universe_market_phases(snapshots)
    assert result["AAPL"].phase == MarketPhase.BULL
    assert result["TSLA"].phase == MarketPhase.BEAR
    assert result["AAPL"].symbol == "AAPL"
    assert result["AAPL"].price == 120.0


def test_classify_universe_empty_snapshots_returns_empty():
    assert classify_universe_market_phases({}) == {}


# --- check_cycle_position_against_market_phase ---------------------------------


def test_no_check_without_claude_cycle_position():
    assert check_cycle_position_against_market_phase("AAPL", None, MarketPhase.BEAR) is None


def test_no_check_when_rule_based_phase_unknown():
    assert check_cycle_position_against_market_phase("AAPL", CyclePosition.MANIA, MarketPhase.UNKNOWN) is None


@pytest.mark.parametrize(
    "cycle_position,plausible_phase",
    [
        (CyclePosition.ACCUMULATION, MarketPhase.SIDEWAYS),
        (CyclePosition.ACCUMULATION, MarketPhase.BEAR),
        (CyclePosition.ATTENTION, MarketPhase.BULL),
        (CyclePosition.ATTENTION, MarketPhase.SIDEWAYS),
        (CyclePosition.MANIA, MarketPhase.BULL),
        (CyclePosition.CRASH, MarketPhase.BEAR),
        (CyclePosition.REVERSION_TO_MEAN, MarketPhase.SIDEWAYS),
    ],
)
def test_plausible_combinations_produce_no_contradiction(cycle_position, plausible_phase):
    assert check_cycle_position_against_market_phase("AAPL", cycle_position, plausible_phase) is None


@pytest.mark.parametrize(
    "cycle_position,implausible_phase",
    [
        (CyclePosition.ACCUMULATION, MarketPhase.BULL),
        (CyclePosition.ATTENTION, MarketPhase.BEAR),
        (CyclePosition.MANIA, MarketPhase.BEAR),
        (CyclePosition.MANIA, MarketPhase.SIDEWAYS),
        (CyclePosition.CRASH, MarketPhase.BULL),
        (CyclePosition.CRASH, MarketPhase.SIDEWAYS),
        (CyclePosition.REVERSION_TO_MEAN, MarketPhase.BULL),
        (CyclePosition.REVERSION_TO_MEAN, MarketPhase.BEAR),
    ],
)
def test_implausible_combinations_produce_contradiction(cycle_position, implausible_phase):
    result = check_cycle_position_against_market_phase("AAPL", cycle_position, implausible_phase)
    assert isinstance(result, MarketPhaseContradiction)
    assert result.symbol == "AAPL"
    assert result.claude_cycle_position == cycle_position
    assert result.rule_based_phase == implausible_phase
    assert "AAPL" in result.detail
    assert cycle_position.value in result.detail
    assert implausible_phase.value in result.detail


def test_contradiction_is_never_a_veto_by_construction():
    """Dieser Check gibt AUSSCHLIESSLICH ein dokumentarisches Objekt oder
    None zurueck - es existiert keine Rueckgabemoeglichkeit, die eine Order
    ablehnen wuerde (kein RiskCheckResult, kein approved-Flag)."""
    result = check_cycle_position_against_market_phase("AAPL", CyclePosition.MANIA, MarketPhase.BEAR)
    assert not hasattr(result, "approved")
