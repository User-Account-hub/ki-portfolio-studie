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

from src.metrics import (
    DEFAULT_RISK_FREE_RATE_ANNUAL,
    PHASE2_PERIODS_PER_YEAR,
    SECONDARY_BENCHMARK_SYMBOL,
    NavHistory,
    _normalize_symbol_to_initial_cash,
    _replay_ledger,
    compute_metrics,
)


def make_nav_history(
    nav_values: list[float],
    periods_per_year: float,
    dates: list[pd.Timestamp] | None = None,
    initial_nav: float | None = None,
    historical_peak_nav: float | None = None,
    benchmark_normalized: list[float] | None = None,
    qqq_normalized: list[float] | None = None,
    baseline_normalized: list[float] | None = None,
    segment_basket_normalized: list[float] | None = None,
) -> NavHistory:
    if dates is None:
        dates = list(pd.date_range("2026-01-05", periods=len(nav_values), freq="D"))
    return NavHistory(
        dates=dates,
        nav=nav_values,
        # Default = nav_values (Benchmark irrelevant fuer die meisten Tests hier).
        benchmark_normalized=benchmark_normalized if benchmark_normalized is not None else nav_values,
        # Default = nav_values (QQQ-Vergleich irrelevant fuer die meisten Tests hier).
        qqq_normalized=qqq_normalized if qqq_normalized is not None else nav_values,
        # Default = nav_values (Momentum-Baseline irrelevant fuer die meisten Tests hier).
        baseline_normalized=baseline_normalized if baseline_normalized is not None else nav_values,
        # Default = nav_values (Segment-ETF-Korb irrelevant fuer die meisten Tests hier).
        segment_basket_normalized=segment_basket_normalized if segment_basket_normalized is not None else nav_values,
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

    # risk_free_rate_annual=0.0 explizit: dieser Test prueft die Annualisierung/
    # Phasenlogik, nicht die Risk-free-Rate-Annahme (siehe test_compute_metrics_
    # default_risk_free_rate_annual_matches_documented_constant dafuer) - soll
    # sich nicht aendern, wenn DEFAULT_RISK_FREE_RATE_ANNUAL angepasst wird.
    result = compute_metrics(nav_history, risk_free_rate_annual=0.0, phase2_start=phase2_start)

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


def test_compute_metrics_baseline_total_return_uses_initial_nav_not_first_checkpoint():
    """Kap. 6.9: baseline_total_return_pct/baseline_alpha_pct muessen wie
    benchmark_total_return_pct gegen NavHistory.initial_nav rechnen, nicht
    gegen baseline_normalized[0] - siehe
    test_compute_metrics_benchmark_total_return_uses_initial_nav_not_first_checkpoint
    fuer die analoge Benchmark-Begruendung."""
    nav_history = make_nav_history(
        nav_values=[969_272.89, 971_148.52],
        periods_per_year=252,
        initial_nav=1_000_000.0,
        baseline_normalized=[989_384.90, 997_819.60],
    )

    result = compute_metrics(nav_history)

    expected_baseline_return = 997_819.60 / 1_000_000.0 - 1
    wrong_baseline_return = 997_819.60 / 989_384.90 - 1
    assert expected_baseline_return < 0 < wrong_baseline_return
    assert result.baseline_total_return_pct == pytest.approx(expected_baseline_return)
    assert result.baseline_total_return_pct != pytest.approx(wrong_baseline_return)
    assert result.baseline_alpha_pct == pytest.approx(result.total_return_pct - expected_baseline_return)


def test_replay_ledger_deducts_transaction_cost_from_cash():
    """2026-09-17: die feste Spread/Slippage-Pauschale muss beim Replay
    mitgezogen werden, sonst driftet die rekonstruierte NAV-Historie von der
    tatsaechlichen (live gefuehrten) cash_balance weg - siehe execution.py."""
    trades = [
        {
            "executed_at": "2026-01-05 10:00:00", "side": "buy", "quantity": 10.0,
            "price": 100.0, "symbol": "AAPL", "instrument_type": "equity",
            "transaction_cost": 1.0,
        },
    ]
    snapshots = _replay_ledger(trades, initial_cash=100_000.0)
    # 100_000 - (10*100) - 1.0 Kosten = 98_999.0
    assert snapshots[-1]["cash"] == pytest.approx(98_999.0)


def test_replay_ledger_defaults_transaction_cost_to_zero_when_column_missing():
    """Rueckwirkende Kompatibilitaet: Trades ohne transaction_cost-Schluessel
    (z.B. ein Fake-Row-Objekt ohne diese Spalte) duerfen den Replay nicht
    crashen und keine Kosten erfinden."""
    trades = [
        {
            "executed_at": "2026-01-05 10:00:00", "side": "buy", "quantity": 10.0,
            "price": 100.0, "symbol": "AAPL", "instrument_type": "equity",
        },
    ]
    snapshots = _replay_ledger(trades, initial_cash=100_000.0)
    assert snapshots[-1]["cash"] == pytest.approx(99_000.0)


# --- risk_free_rate_annual (2026-09-17: realistischer Default statt 0.0) -----


def test_compute_metrics_default_risk_free_rate_is_no_longer_zero():
    """Vorher war risk_free_rate_annual=0.0 der Default - das unterstellte,
    "risikofrei" wuerfe 0% Rendite ab, und ueberzeichnete damit den Sharpe.
    Ohne explizite Angabe muss compute_metrics jetzt DEFAULT_RISK_FREE_RATE_
    ANNUAL (~4%, US-3-Monats-T-Bill-Anker) verwenden, was den Sharpe
    gegenueber der alten 0%-Annahme senkt (hoehere Rendite-Huerde)."""
    nav_values = [100_000, 101_000, 99_500, 102_000, 101_500]

    with_default = compute_metrics(make_nav_history(nav_values, periods_per_year=104))
    with_zero_rf = compute_metrics(
        make_nav_history(nav_values, periods_per_year=104), risk_free_rate_annual=0.0
    )

    assert DEFAULT_RISK_FREE_RATE_ANNUAL > 0.0
    assert with_default.sharpe_ratio != pytest.approx(with_zero_rf.sharpe_ratio)
    assert with_default.sharpe_ratio < with_zero_rf.sharpe_ratio
    # annualized_volatility_pct haengt nicht von der Risk-free-Rate ab.
    assert with_default.annualized_volatility_pct == pytest.approx(with_zero_rf.annualized_volatility_pct)


def test_compute_metrics_uses_default_risk_free_rate_when_not_specified():
    nav_values = [100_000, 101_000, 99_500, 102_000, 101_500]
    result_implicit = compute_metrics(make_nav_history(nav_values, periods_per_year=104))
    result_explicit = compute_metrics(
        make_nav_history(nav_values, periods_per_year=104),
        risk_free_rate_annual=DEFAULT_RISK_FREE_RATE_ANNUAL,
    )
    assert result_implicit.sharpe_ratio == pytest.approx(result_explicit.sharpe_ratio)


# --- Information Ratio (2026-09-20: Alpha / Tracking Error) ------------------


def test_compute_metrics_information_ratio_matches_alpha_over_tracking_error():
    nav_values = [100_000, 101_000, 102_500, 101_800, 103_000]
    benchmark_values = [100_000, 100_500, 101_000, 101_200, 101_500]
    nav_history = make_nav_history(nav_values, periods_per_year=104, benchmark_normalized=benchmark_values)

    result = compute_metrics(nav_history, risk_free_rate_annual=0.0)

    nav = pd.Series(nav_values)
    benchmark = pd.Series(benchmark_values)
    returns = nav.pct_change().dropna()
    benchmark_returns = benchmark.pct_change()
    active_returns = returns - benchmark_returns.loc[returns.index]
    expected_tracking_error = float(active_returns.std() * np.sqrt(104))
    expected_information_ratio = result.alpha_pct / expected_tracking_error

    assert expected_tracking_error > 0  # sanity: Portfolio und Benchmark divergieren tatsaechlich
    assert result.information_ratio == pytest.approx(expected_information_ratio)


def test_compute_metrics_information_ratio_none_when_portfolio_tracks_benchmark_exactly():
    """std() der aktiven Rendite ist 0, wenn Portfolio und Benchmark exakt
    gleichlaufen (hier: Benchmark = Portfolio, der Default von
    make_nav_history) - Tracking Error und damit Information Ratio sind dann
    nicht definiert, NICHT +-inf oder 0."""
    nav_values = [100_000, 101_000, 102_500, 101_800, 103_000]
    nav_history = make_nav_history(nav_values, periods_per_year=104)
    result = compute_metrics(nav_history)
    assert result.information_ratio is None


def test_compute_metrics_information_ratio_none_with_single_return():
    nav_history = make_nav_history(
        [100_000, 101_000], periods_per_year=104, benchmark_normalized=[100_000, 100_500]
    )
    result = compute_metrics(nav_history)
    assert result.information_ratio is None


def test_compute_metrics_information_ratio_negative_when_underperforming():
    nav_values = [100_000, 99_000, 98_500, 97_800]
    benchmark_values = [100_000, 100_500, 101_000, 101_200]
    nav_history = make_nav_history(nav_values, periods_per_year=104, benchmark_normalized=benchmark_values)

    result = compute_metrics(nav_history)

    assert result.alpha_pct < 0
    assert result.information_ratio is not None
    assert result.information_ratio < 0


def test_compute_metrics_information_ratio_uses_phase2_annualization_once_enough_daily_returns_exist():
    """Tracking Error/Information Ratio muessen denselben Phase-1/2-Split wie
    Vol/Sharpe verwenden (siehe
    test_compute_metrics_uses_phase2_annualization_once_enough_daily_returns_exist) -
    nicht unabhaengig davon annualisiert werden."""
    phase2_start = pd.Timestamp("2026-09-10")
    dates = [
        pd.Timestamp("2026-09-03"),
        pd.Timestamp("2026-09-07"),
        pd.Timestamp("2026-09-10"),
        pd.Timestamp("2026-09-11"),
        pd.Timestamp("2026-09-14"),
        pd.Timestamp("2026-09-15"),
    ]
    nav_values = [100_000, 100_800, 101_500, 101_200, 101_900, 101_600]
    benchmark_values = [100_000, 100_200, 100_400, 100_600, 100_500, 100_700]
    nav_history = make_nav_history(
        nav_values, periods_per_year=104, dates=dates, benchmark_normalized=benchmark_values,
    )

    result = compute_metrics(nav_history, risk_free_rate_annual=0.0, phase2_start=phase2_start)

    nav = pd.Series(nav_values, index=dates)
    benchmark = pd.Series(benchmark_values, index=dates)
    returns = nav.pct_change().dropna()
    benchmark_returns = benchmark.pct_change()
    active_returns = returns - benchmark_returns.loc[returns.index]
    interval_start_dates = nav.index[:-1]
    phase2_active_returns = active_returns[interval_start_dates >= phase2_start]
    assert len(phase2_active_returns) == 3

    expected_tracking_error = float(phase2_active_returns.std() * np.sqrt(PHASE2_PERIODS_PER_YEAR))
    expected_information_ratio = result.alpha_pct / expected_tracking_error
    assert result.information_ratio == pytest.approx(expected_information_ratio)


# --- QQQ-Vergleichsindex (Kap. 6.9 Erweiterung, 2026-09-21) -------------------


def test_secondary_benchmark_symbol_is_qqq():
    """Dokumentiert die bewusste, feste Wahl (siehe Modul-Kommentar) - kein
    frei konfigurierbarer Wert wie benchmark_symbol."""
    assert SECONDARY_BENCHMARK_SYMBOL == "QQQ"


def test_normalize_symbol_to_initial_cash_basic():
    prices = {"2026-01-05": 100.0, "2026-01-06": 110.0, "2026-01-07": 90.0}

    def price_on(symbol, when):
        return prices.get(when.strftime("%Y-%m-%d"))

    dates = [pd.Timestamp(d) for d in prices]
    result = _normalize_symbol_to_initial_cash(price_on, "QQQ", dates[0], dates, initial_cash=1_000.0)
    assert result == [pytest.approx(1_000.0), pytest.approx(1_100.0), pytest.approx(900.0)]


def test_normalize_symbol_to_initial_cash_falls_back_when_start_price_missing():
    def price_on(symbol, when):
        return None  # Symbol nicht aufloesbar

    dates = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    result = _normalize_symbol_to_initial_cash(price_on, "QQQ", dates[0], dates, initial_cash=1_000.0)
    assert result == [1_000.0, 1_000.0]


def test_normalize_symbol_to_initial_cash_falls_back_for_missing_individual_date():
    def price_on(symbol, when):
        if when == pd.Timestamp("2026-01-05"):
            return 100.0
        return None  # Luecke an diesem einen Datum

    dates = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    result = _normalize_symbol_to_initial_cash(price_on, "QQQ", dates[0], dates, initial_cash=1_000.0)
    assert result == [pytest.approx(1_000.0), 1_000.0]


def test_compute_metrics_qqq_fields_computed_like_benchmark():
    """qqq_total_return_pct/alpha_vs_qqq_pct muessen dieselbe initial_nav-
    Ankerung wie benchmark_total_return_pct/alpha_pct verwenden (siehe
    Kommentar in compute_metrics)."""
    nav_history = make_nav_history(
        nav_values=[100_000, 105_000],
        periods_per_year=252,
        qqq_normalized=[100_000, 112_000],
    )
    result = compute_metrics(nav_history)
    assert result.qqq_total_return_pct == pytest.approx(0.12)
    assert result.alpha_vs_qqq_pct == pytest.approx(result.total_return_pct - 0.12)


def test_compute_metrics_qqq_alpha_negative_when_qqq_outperforms():
    nav_history = make_nav_history(
        nav_values=[100_000, 104_000],
        periods_per_year=252,
        qqq_normalized=[100_000, 115_000],
    )
    result = compute_metrics(nav_history)
    assert result.alpha_vs_qqq_pct < 0


def test_compute_metrics_qqq_independent_of_primary_benchmark():
    """QQQ ERGAENZT SPY, ist aber davon unabhaengig - unterschiedliche
    Kursverlaeufe fuer benchmark_normalized vs. qqq_normalized duerfen sich
    nicht gegenseitig beeinflussen."""
    nav_history = make_nav_history(
        nav_values=[100_000, 100_000],
        periods_per_year=252,
        benchmark_normalized=[100_000, 105_000],
        qqq_normalized=[100_000, 120_000],
    )
    result = compute_metrics(nav_history)
    assert result.benchmark_total_return_pct == pytest.approx(0.05)
    assert result.qqq_total_return_pct == pytest.approx(0.20)
    assert result.alpha_pct == pytest.approx(-0.05)
    assert result.alpha_vs_qqq_pct == pytest.approx(-0.20)


def test_reconstruct_nav_history_no_trades_includes_flat_qqq_line():
    """Der fruehe 'keine Trades'-Rueckgabepfad muss ebenfalls
    qqq_normalized befuellen, nicht nur benchmark_normalized/
    baseline_normalized (sonst crasht die NavHistory-Konstruktion)."""
    import sqlite3

    from src.metrics import reconstruct_nav_history

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE nav_history (id INTEGER PRIMARY KEY, portfolio_id INTEGER, recorded_at TEXT, nav REAL)"
    )
    portfolio_row = {"id": 1, "initial_cash_balance": 100_000.0}
    result = reconstruct_nav_history(conn, portfolio_row, trades=[], watchlist_underlyings={}, benchmark_symbol="SPY")
    assert result.qqq_normalized == [100_000.0]


def test_reconstruct_nav_history_anchors_on_reset_nav_not_stale_pilot_phase_history():
    """Bugfix 2026-09-21 (Kap.-6.3-Reset): reproduziert den gemeldeten Fall
    vom 21.09. - der Aufrufer uebergibt bereits (via db.get_trades_since)
    keine Trades seit dem Reset (alle heutigen Order-Vorschlaege wurden wegen
    Notional-Limit abgelehnt), aber die DB hat noch eine viel hoehere
    Pilotphase-nav_history-Zeile VOR dem Reset. `initial_cash`/`initial_nav`
    muessen trotzdem auf den Reset-NAV (1'000'000) verankert sein, NICHT auf
    das Pilotphase-Zwischenhoch UND NICHT auf portfolio_row["initial_cash_balance"]."""
    import sqlite3

    from src.metrics import OFFICIAL_STUDY_START, reconstruct_nav_history

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE nav_history (id INTEGER PRIMARY KEY, portfolio_id INTEGER, recorded_at TEXT, nav REAL)"
    )
    conn.execute(
        "INSERT INTO nav_history (portfolio_id, recorded_at, nav) VALUES (1, '2026-09-17 15:15:35', 1058900.0)"
    )  # Pilotphase-Zwischenhoch, VOR dem Reset
    conn.execute(
        "INSERT INTO nav_history (portfolio_id, recorded_at, nav) VALUES (1, ?, 1000000.0)",
        (OFFICIAL_STUDY_START.strftime("%Y-%m-%d %H:%M:%S"),),
    )  # der Reset-Eintrag selbst
    conn.commit()

    # portfolio_row["initial_cash_balance"] absichtlich abweichend (100_000)
    # gesetzt, um sicherzustellen, dass NICHT dieser statische Fallback-Wert
    # verwendet wird, solange ein nav_history-Eintrag ab dem Reset existiert.
    portfolio_row = {"id": 1, "initial_cash_balance": 100_000.0}
    result = reconstruct_nav_history(conn, portfolio_row, trades=[], watchlist_underlyings={}, benchmark_symbol="SPY")

    assert result.nav == [1_000_000.0]
    assert result.initial_nav == 1_000_000.0


# --- Segment-ETF-Korb (Kap. 6.9 Erweiterung, 2026-09-21) ----------------------


def test_compute_metrics_segment_basket_fields_computed_like_baseline():
    """segment_basket_total_return_pct/alpha_vs_segment_basket_pct muessen
    dieselbe initial_nav-Ankerung wie baseline_total_return_pct/
    baseline_alpha_pct verwenden (siehe Kommentar in compute_metrics)."""
    nav_history = make_nav_history(
        nav_values=[100_000, 105_000],
        periods_per_year=252,
        segment_basket_normalized=[100_000, 118_000],
    )
    result = compute_metrics(nav_history)
    assert result.segment_basket_total_return_pct == pytest.approx(0.18)
    assert result.alpha_vs_segment_basket_pct == pytest.approx(result.total_return_pct - 0.18)


def test_compute_metrics_segment_basket_alpha_negative_when_basket_outperforms():
    nav_history = make_nav_history(
        nav_values=[100_000, 104_000],
        periods_per_year=252,
        segment_basket_normalized=[100_000, 130_000],
    )
    result = compute_metrics(nav_history)
    assert result.alpha_vs_segment_basket_pct < 0


def test_compute_metrics_segment_basket_independent_of_other_comparators():
    """Der Segment-Korb ERGAENZT SPY/QQQ/Momentum-Baseline, ist aber davon
    unabhaengig - unterschiedliche Kursverlaeufe duerfen sich nicht
    gegenseitig beeinflussen."""
    nav_history = make_nav_history(
        nav_values=[100_000, 100_000],
        periods_per_year=252,
        benchmark_normalized=[100_000, 105_000],
        qqq_normalized=[100_000, 120_000],
        baseline_normalized=[100_000, 108_000],
        segment_basket_normalized=[100_000, 130_000],
    )
    result = compute_metrics(nav_history)
    assert result.benchmark_total_return_pct == pytest.approx(0.05)
    assert result.qqq_total_return_pct == pytest.approx(0.20)
    assert result.baseline_total_return_pct == pytest.approx(0.08)
    assert result.segment_basket_total_return_pct == pytest.approx(0.30)
    assert result.alpha_vs_segment_basket_pct == pytest.approx(-0.30)


def test_reconstruct_nav_history_no_trades_includes_flat_segment_basket_line():
    """Der fruehe 'keine Trades'-Rueckgabepfad muss ebenfalls
    segment_basket_normalized befuellen (sonst crasht die
    NavHistory-Konstruktion)."""
    import sqlite3

    from src.metrics import reconstruct_nav_history

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE nav_history (id INTEGER PRIMARY KEY, portfolio_id INTEGER, recorded_at TEXT, nav REAL)"
    )
    portfolio_row = {"id": 1, "initial_cash_balance": 100_000.0}
    result = reconstruct_nav_history(conn, portfolio_row, trades=[], watchlist_underlyings={}, benchmark_symbol="SPY")
    assert result.segment_basket_normalized == [100_000.0]


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
