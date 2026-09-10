"""Tests for src/metrics.py's annualization factor.

compute_metrics must use NavHistory.periods_per_year (derived from however
many checkpoints/week the pipeline actually produces), not a hardcoded
constant - otherwise a cadence change (e.g. 1x/week -> 2x/week) silently
skews volatility/Sharpe without anyone noticing.

Also covers the Phase 1 -> Phase 2 regime change (2026-09-10, Thesis Kap.
6.3/11.2/15: 2x/Woche -> taeglich) - compute_metrics must not blend Phase-1
(longer-interval) and Phase-2 (daily) returns into one annualized figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import PHASE2_PERIODS_PER_YEAR, NavHistory, compute_metrics


def make_nav_history(
    nav_values: list[float], periods_per_year: float, dates: list[pd.Timestamp] | None = None
) -> NavHistory:
    if dates is None:
        dates = list(pd.date_range("2026-01-05", periods=len(nav_values), freq="D"))
    return NavHistory(
        dates=dates,
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


def test_compute_metrics_uses_phase2_annualization_once_enough_daily_returns_exist():
    """Regimewechsel Phase 1 (2x/Woche) -> Phase 2 (taeglich) am 2026-09-10
    (Thesis Kap. 6.3/11.2/15): sobald genug reine Phase-2-Renditen (Intervall-
    Start >= phase2_start) vorliegen, muss Vol/Sharpe mit
    PHASE2_PERIODS_PER_YEAR (252) annualisiert werden - nicht mit der alten
    Phase-1-Rate (104) und nicht mit den laengeren Phase-1-Intervallen
    vermischt."""
    phase2_start = pd.Timestamp("2026-09-10")
    # 2 Phase-1-Checkpoints (Mo/Do) vor dem Regimewechsel, dann 4 taegliche
    # Phase-2-Checkpoints (3 reine Phase-2-Renditen).
    dates = [
        pd.Timestamp("2026-09-03"),  # Phase 1
        pd.Timestamp("2026-09-07"),  # Phase 1 (letzter Mo/Do-Checkpoint)
        pd.Timestamp("2026-09-10"),  # phase2_start - Rendite dorthin ist Phase 1 -> Phase 2 (Uebergang)
        pd.Timestamp("2026-09-11"),  # reine Phase-2-Rendite (Start >= phase2_start)
        pd.Timestamp("2026-09-14"),  # reine Phase-2-Rendite
        pd.Timestamp("2026-09-15"),  # reine Phase-2-Rendite
    ]
    nav_values = [100_000, 100_800, 101_500, 101_200, 101_900, 101_600]
    nav_history = make_nav_history(nav_values, periods_per_year=104, dates=dates)

    result = compute_metrics(nav_history, phase2_start=phase2_start)

    nav = pd.Series(nav_values, index=dates)
    returns = nav.pct_change().dropna()
    interval_start_dates = nav.index[:-1]
    phase2_returns = returns[interval_start_dates >= phase2_start]
    assert len(phase2_returns) == 3  # Sept10->11, 11->14, 14->15 - Sept07->10 ausgeschlossen

    expected_vol = float(phase2_returns.std() * np.sqrt(PHASE2_PERIODS_PER_YEAR))
    expected_sharpe = float(phase2_returns.mean() / phase2_returns.std() * np.sqrt(PHASE2_PERIODS_PER_YEAR))
    assert result.annualized_volatility_pct == pytest.approx(expected_vol)
    assert result.sharpe_ratio == pytest.approx(expected_sharpe)

    # total_return/max_drawdown bleiben ueber die volle Historie (Phase 1 + 2)
    expected_total_return = nav.iloc[-1] / nav.iloc[0] - 1
    assert result.total_return_pct == pytest.approx(expected_total_return)


def test_compute_metrics_falls_back_to_phase1_annualization_before_regime_change():
    """Vor dem Regimewechsel (oder mit zu wenigen reinen Phase-2-Renditen)
    darf sich am bisherigen Verhalten nichts aendern - reine Phase-1-Historie
    nutzt weiterhin NavHistory.periods_per_year."""
    nav_values = [100_000, 101_000, 99_500, 102_000, 101_500]
    nav_history = make_nav_history(nav_values, periods_per_year=104)  # Datumsreihe: Jan 2026, vor PHASE2_START

    result = compute_metrics(nav_history)

    returns = pd.Series(nav_values).pct_change().dropna()
    expected_vol = float(returns.std() * np.sqrt(104))
    assert result.annualized_volatility_pct == pytest.approx(expected_vol)
