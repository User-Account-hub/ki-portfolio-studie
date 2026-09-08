"""Tests for src/metrics.py's annualization factor.

compute_metrics must use NavHistory.periods_per_year (derived from however
many checkpoints/week the pipeline actually produces), not a hardcoded
constant - otherwise a cadence change (e.g. 1x/week -> 2x/week) silently
skews volatility/Sharpe without anyone noticing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import NavHistory, compute_metrics


def make_nav_history(nav_values: list[float], periods_per_year: float) -> NavHistory:
    dates = pd.date_range("2026-01-05", periods=len(nav_values), freq="D")
    return NavHistory(
        dates=list(dates),
        nav=nav_values,
        benchmark_normalized=nav_values,  # Benchmark irrelevant fuer diesen Test
        periods_per_year=periods_per_year,
    )


def test_compute_metrics_uses_nav_history_periods_per_year_for_annualization():
    """Gleiche Renditereihe, unterschiedliche periods_per_year (52 vs. 104,
    z.B. 1x vs. 2x woechentlich) muss unterschiedliche annualisierte
    Volatilitaet/Sharpe ergeben - sqrt(104) != sqrt(52)."""
    nav_values = [100_000, 101_000, 99_500, 102_000, 101_500]

    weekly = compute_metrics(make_nav_history(nav_values, periods_per_year=52))
    twice_weekly = compute_metrics(make_nav_history(nav_values, periods_per_year=104))

    returns = pd.Series(nav_values).pct_change().dropna()
    expected_vol_weekly = float(returns.std() * np.sqrt(52))
    expected_vol_twice_weekly = float(returns.std() * np.sqrt(104))

    assert weekly.annualized_volatility_pct == pytest.approx(expected_vol_weekly)
    assert twice_weekly.annualized_volatility_pct == pytest.approx(expected_vol_twice_weekly)
    assert twice_weekly.annualized_volatility_pct == pytest.approx(
        weekly.annualized_volatility_pct * np.sqrt(2)
    )


def test_compute_metrics_total_return_independent_of_periods_per_year():
    """Gesamtrendite/Max-Drawdown haengen nicht von der Annualisierung ab -
    nur Vol/Sharpe tun das."""
    nav_values = [100_000, 105_000, 95_000, 110_000]

    weekly = compute_metrics(make_nav_history(nav_values, periods_per_year=52))
    twice_weekly = compute_metrics(make_nav_history(nav_values, periods_per_year=104))

    assert weekly.total_return_pct == pytest.approx(twice_weekly.total_return_pct)
    assert weekly.max_drawdown_pct == pytest.approx(twice_weekly.max_drawdown_pct)
