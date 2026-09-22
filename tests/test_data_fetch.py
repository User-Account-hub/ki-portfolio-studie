"""Tests for src/data_fetch.py's pure calculation helpers (_pct_change, _sma,
_annualized_volatility) - previously untested (17-point audit Fund #14). No
network access: operates directly on hand-built pandas Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_fetch import _annualized_volatility, _pct_change, _sma

# --- _pct_change ---------------------------------------------------------------


def test_pct_change_computes_correct_ratio():
    close = pd.Series([100.0, 110.0])
    assert _pct_change(close, periods=1) == pytest.approx(0.10)


def test_pct_change_over_multiple_periods():
    close = pd.Series([100.0, 105.0, 90.0, 120.0])
    assert _pct_change(close, periods=3) == pytest.approx(120.0 / 100.0 - 1)


def test_pct_change_none_when_not_enough_history():
    close = pd.Series([100.0, 110.0])
    assert _pct_change(close, periods=2) is None


def test_pct_change_none_at_exact_boundary():
    """len(close) == periods (nicht > periods) muss noch als 'nicht genug'
    gelten - siehe die `<=`-Bedingung in _pct_change."""
    close = pd.Series([100.0, 110.0])
    assert _pct_change(close, periods=len(close)) is None


def test_pct_change_works_at_minimum_sufficient_length():
    close = pd.Series([100.0, 110.0])
    assert _pct_change(close, periods=len(close) - 1) == pytest.approx(0.10)


# --- _sma ------------------------------------------------------------------------


def test_sma_computes_rolling_mean_of_last_window():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _sma(close, window=3) == pytest.approx((3.0 + 4.0 + 5.0) / 3)


def test_sma_none_when_not_enough_history():
    close = pd.Series([1.0, 2.0])
    assert _sma(close, window=3) is None


def test_sma_works_at_exact_boundary_length():
    close = pd.Series([1.0, 2.0, 3.0])
    assert _sma(close, window=3) == pytest.approx(2.0)


# --- _annualized_volatility -------------------------------------------------------


def test_annualized_volatility_zero_for_constant_returns():
    """Zwei identische Tagesrenditen hintereinander (100 -> 110 -> 121, je
    +10%) ergeben eine Stichproben-Standardabweichung von exakt 0."""
    close = pd.Series([100.0, 110.0, 121.0])
    assert _annualized_volatility(close, window=2) == pytest.approx(0.0)


def test_annualized_volatility_matches_manual_pandas_calculation():
    close = pd.Series([100.0, 102.0, 101.0, 103.0, 100.0, 105.0])
    window = 5
    expected = close.pct_change().dropna().iloc[-window:].std() * np.sqrt(252)
    assert _annualized_volatility(close, window) == pytest.approx(expected)


def test_annualized_volatility_none_when_not_enough_history():
    close = pd.Series([100.0, 101.0, 99.0])
    assert _annualized_volatility(close, window=3) is None


def test_annualized_volatility_works_at_exact_boundary_length():
    """len(close) == window + 1 ist die minimal ausreichende Laenge (window
    Renditen aus window+1 Kursen)."""
    close = pd.Series([100.0, 101.0, 99.0, 102.0])
    assert _annualized_volatility(close, window=3) is not None
