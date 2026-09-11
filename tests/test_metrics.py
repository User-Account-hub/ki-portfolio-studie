"""Tests for src/metrics.py's annualization factor.

compute_metrics must use NavHistory.periods_per_year (derived from however
many checkpoints/week the pipeline actually produces), not a hardcoded
constant - otherwise a cadence change (e.g. 1x/week -> 2x/week) silently
skews volatility/Sharpe without anyone noticing.

Also covers the Phase 1 -> Phase 2 regime change (2026-09-10, Thesis Kap.
6.3/11.2/15: 2x/Woche -> taeglich) - compute_metrics must not blend Phase-1
(longer-interval) and Phase-2 (daily) returns into one annualized figure.

And the 2026-09-12 fixes for the same underlying bug class (a fixed,
run-independent reference point instead of "first local checkpoint"):
- total_return_pct against NavHistory.initial_nav, not nav[0] - see
  test_compute_metrics_total_return_uses_initial_nav_not_first_checkpoint.
- max_drawdown_pct against NavHistory.historical_peak_nav (the real
  high-water mark, reused from db.get_peak_nav/the Kap.-6.8 circuit
  breaker), not nav.cummax() alone - see
  test_compute_metrics_max_drawdown_uses_historical_peak_not_local_cummax.
- benchmark_total_return_pct/alpha_pct against NavHistory.initial_nav, not
  benchmark[0] - see
  test_compute_metrics_benchmark_total_return_uses_initial_nav_not_first_checkpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import PHASE2_PERIODS_PER_YEAR, NavHistory, compute_metrics


def make_nav_history(
    nav_values: list[float],
    periods_per_year: float,
    dates: list[pd.Timestamp] | None = None,
    initial_nav: float | None = None,
    historical_peak_nav: float | None = None,
    benchmark_normalized: list[float] | None = None,
) -> NavHistory:
    if dates is None:
        dates = list(pd.date_range("2026-01-05", periods=len(nav_values), freq="D"))
    return NavHistory(
        dates=dates,
        nav=nav_values,
        # Default = nav_values (Benchmark irrelevant fuer die meisten Tests hier).
        benchmark_normalized=benchmark_normalized if benchmark_normalized is not None else nav_values,
        periods_per_year=periods_per_year,
        # Default = nav_values[0]: die meisten Tests hier pruefen Annualisierung/
        # Phasenlogik, nicht total_return, und sollen sich nicht aendern, wenn
        # initial_nav nicht explizit gesetzt wird (siehe test_total_return_uses_
        # initial_nav_not_first_checkpoint fuer den eigentlichen Bugfix-Test).
        initial_nav=initial_nav if initial_nav is not None else nav_values[0],
        # Default = nav_values[0]: cummax()[0] == nav_values[0] immer, ein Floor
        # darauf ist also fuer alle bisherigen Tests ein No-Op (siehe
        # test_compute_metrics_max_drawdown_uses_historical_peak_not_local_cummax
        # fuer den Fall, in dem historical_peak_nav > nav.cummax() ist).
        historical_peak_nav=historical_peak_nav if historical_peak_nav is not None else nav_values[0],
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


def test_compute_metrics_total_return_uses_initial_nav_not_first_checkpoint():
    """Bugfix 2026-09-12: total_return_pct muss gegen NavHistory.initial_nav
    (das echte Studien-Startkapital) rechnen, nicht gegen nav[0] (den ersten
    Checkpoint DIESES Laufs). Reproduziert den realen Fall vom 11.09.2026:
    ein Lauf, dessen Zeitreihe erst ab einem spaeteren Checkpoint beginnt
    (z.B. weil Phase 2 mehr Checkpoints pro Lauf erzeugt als Phase 1), darf
    "Gesamtrendite" nicht mit der letzten Perioden-Rendite verwechseln."""
    # Nachgebildet an den echten Werten: Startkapital 1'000'000, aber der
    # erste Checkpoint DIESES Laufs (nav[0]) liegt schon bei ~969'273 (nach
    # Kursverlusten vor Laufbeginn) - nav[0] != initial_nav.
    nav_history = make_nav_history(
        nav_values=[969_272.89, 971_148.52],
        periods_per_year=252,
        initial_nav=1_000_000.0,
    )

    result = compute_metrics(nav_history)

    # Vor dem Fix waere das hier faelschlich nav[-1]/nav[0] - 1 (~+0.19%,
    # identisch zu last_period_return_pct) statt der echten kumulierten
    # Rendite seit Studienbeginn (~-2.89%).
    expected_total_return = 971_148.52 / 1_000_000.0 - 1
    assert expected_total_return < 0  # sanity: NAV liegt unter dem Startkapital
    assert result.total_return_pct == pytest.approx(expected_total_return)
    assert result.total_return_pct != pytest.approx(result.last_period_return_pct)


def test_compute_metrics_max_drawdown_uses_historical_peak_not_local_cummax():
    """Bugfix 2026-09-12 (Teil 2): max_drawdown_pct muss gegen den WIRKLICH
    bekannten historischen Hoechststand (NavHistory.historical_peak_nav)
    rechnen, nicht nur gegen nav.cummax() ueber die lokalen Checkpoints
    DIESES Laufs. Reproduziert den realen Fall vom 11.09.2026: beide lokalen
    Checkpoints liegen schon unter dem echten Hoechststand UND steigen lokal
    an - nav.cummax() allein haette daraus faelschlich 0.00% Drawdown
    gemacht, obwohl das Portfolio seit seinem Hoechststand bereits im Minus
    war."""
    nav_values = [969_272.89, 971_148.52]  # steigt lokal -> nav.cummax()-only waere 0% Drawdown
    historical_peak = 1_000_212.08  # echter Hoechststand, siehe db.get_peak_nav

    nav_history = make_nav_history(
        nav_values=nav_values,
        periods_per_year=252,
        initial_nav=1_000_000.0,
        historical_peak_nav=historical_peak,
    )

    result = compute_metrics(nav_history)

    # Beide Checkpoints liegen unter historical_peak, der lokale cummax
    # uebertrifft ihn nirgends -> running_max ist an jedem Punkt konstant
    # historical_peak, und der schlechteste (kleinste) Drawdown ist der am
    # niedrigsten liegende NAV-Wert (hier: der ERSTE Checkpoint, nicht der
    # letzte - max_drawdown ist der schlechteste Punkt der ganzen Serie).
    expected_drawdown = (min(nav_values) - historical_peak) / historical_peak
    assert expected_drawdown < 0  # sanity: NAV liegt unter dem echten Hoechststand
    assert result.max_drawdown_pct == pytest.approx(expected_drawdown)
    assert result.max_drawdown_pct != pytest.approx(0.0)


def test_compute_metrics_benchmark_total_return_uses_initial_nav_not_first_checkpoint():
    """Bugfix 2026-09-12 (Teil 3): benchmark_total_return_pct (und damit
    alpha_pct) muessen gegen NavHistory.initial_nav rechnen, nicht gegen
    benchmark[0]. benchmark[0] ist per Konstruktion IMMER gleich dem
    Dollarwert am jeweils gewaehlten Anker-Datum - unabhaengig davon, ob
    dieser Anker (frueher: dates[0], jetzt: der echte Studienbeginn) korrekt
    ist. Reproduziert die realen SPY-Werte vom 11.09.2026: bei falscher Basis
    (benchmark[0]) kommt eine positive Benchmark-Rendite heraus, bei
    korrekter Basis (initial_nav) eine leicht negative - und damit ein
    komplett anderes alpha_pct."""
    nav_history = make_nav_history(
        nav_values=[969_272.89, 971_148.52],
        periods_per_year=252,
        initial_nav=1_000_000.0,
        # Nachgebildet an echten SPY-Kursen 08./10./11.09.2026 (765.96 /
        # 757.83 / 764.29), korrekt auf initial_nav am echten Studienbeginn
        # (08.09.) normalisiert - benchmark[0] (989'384.90) ist bewusst
        # NICHT gleich initial_nav, um den Bug von der Fix-Variante zu
        # unterscheiden.
        benchmark_normalized=[989_384.90, 997_819.60],
    )

    result = compute_metrics(nav_history)

    expected_benchmark_return = 997_819.60 / 1_000_000.0 - 1
    wrong_benchmark_return = 997_819.60 / 989_384.90 - 1  # alte, dates[0]-relative Formel
    assert expected_benchmark_return < 0 < wrong_benchmark_return  # unterschiedliches Vorzeichen!
    assert result.benchmark_total_return_pct == pytest.approx(expected_benchmark_return)
    assert result.benchmark_total_return_pct != pytest.approx(wrong_benchmark_return)
    assert result.alpha_pct == pytest.approx(result.total_return_pct - expected_benchmark_return)


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
