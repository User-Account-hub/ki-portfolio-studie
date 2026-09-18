"""Tests for src/reporting.py's Kap.-6.12.3 report section: the monthly deep
reflection must appear only on the run it was actually generated on, and
render its fields when present.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.boundary_conditions import BoundaryConditionCheck
from src.correlation import CorrelationWarning
from src.data_quality import DataQualityReport
from src.deep_reflection_schema import DeepReflectionOutput
from src.execution import ExecutedOrderResult
from src.metrics import MetricsResult
from src.order_schema import ProposedOrder
from src.reporting import generate_report, generate_stress_test_report
from src.stress_test import PositionWeight, StressPeriodResult
from src.stress_test_schema import StressTestCommentary

EMPTY_DQ_REPORT = DataQualityReport(price_deviations=[], missing_trading_days=[], outlier_moves=[])


def make_metrics() -> MetricsResult:
    return MetricsResult(
        current_nav=100_000.0,
        total_return_pct=0.0,
        last_period_return_pct=None,
        annualized_volatility_pct=None,
        sharpe_ratio=None,
        max_drawdown_pct=0.0,
        benchmark_total_return_pct=0.0,
        alpha_pct=0.0,
        baseline_total_return_pct=0.0,
        baseline_alpha_pct=0.0,
    )


def _generate(
    deep_reflection,
    triggered_boundary_conditions=None,
    still_open_boundary_conditions=None,
    executed_results=None,
    correlation_cluster_count=None,
) -> str:
    portfolio_row = {"name": "test", "currency": "USD", "benchmark_symbol": "SPY"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = generate_report(
            portfolio_row, [], executed_results or [], [], "", make_metrics(), EMPTY_DQ_REPORT, deep_reflection,
            triggered_boundary_conditions or [], still_open_boundary_conditions or [], correlation_cluster_count,
            tmp_dir,
        )
        return Path(report_path).read_text(encoding="utf-8")


def test_report_omits_reflection_section_when_none():
    content = _generate(deep_reflection=None)
    assert "Monatliche Tiefenreflexion" not in content


def test_report_includes_reflection_section_when_present():
    reflection = DeepReflectionOutput(
        theses_confirmed=["NVDA-These bestätigt"],
        theses_falsified_or_overdue=["CCJ-These überfällig"],
        pattern_matching_concerns="keine",
        portfolio_stance_assessment="kohärent",
        reflection_commentary="Alles im Rahmen der Methodik.",
    )
    content = _generate(deep_reflection=reflection)
    assert "## Monatliche Tiefenreflexion (Kap. 6.12.3)" in content
    assert "NVDA-These bestätigt" in content
    assert "CCJ-These überfällig" in content
    assert "Alles im Rahmen der Methodik." in content


# --- Randbedingungen (Kap. 7) -----------------------------------------------


def test_report_shows_no_open_boundary_conditions_when_empty():
    content = _generate(deep_reflection=None)
    assert "## Randbedingungen (Kap. 7)" in content
    assert "_Keine offenen Randbedingungen._" in content


def test_report_shows_triggered_and_still_open_boundary_conditions():
    triggered = [
        BoundaryConditionCheck(
            id=1, position_id=10, symbol="NVDA", description="Fällt unter $150",
            check_type="price_below", threshold_price=150.0,
        )
    ]
    still_open = [
        BoundaryConditionCheck(
            id=2, position_id=11, symbol="CCJ", description="Q3-Earnings enttäuschen",
            check_type="qualitative", threshold_price=None,
        ),
        BoundaryConditionCheck(
            id=3, position_id=12, symbol="AMD", description="Steigt über $250",
            check_type="price_above", threshold_price=250.0,
        ),
    ]
    content = _generate(
        deep_reflection=None, triggered_boundary_conditions=triggered, still_open_boundary_conditions=still_open
    )
    assert "**Ausgelöst in diesem Lauf:**" in content
    assert "NVDA" in content and "Fällt unter $150" in content
    assert "**Weiterhin offen:**" in content
    assert "nicht automatisch prüfbar" in content
    assert "Schwelle 250.00" in content
    assert "_Keine offenen Randbedingungen._" not in content


# --- generate_stress_test_report (Kap. 11.2) --------------------------------


def test_stress_test_report_always_shows_contamination_caveat_banner():
    """Kernanforderung: der Kontaminations-Hinweis muss IMMER erscheinen,
    unabhaengig vom Inhalt der Ergebnisse."""
    portfolio_row = {"name": "test"}
    weights = [PositionWeight(symbol="NVDA", weight=1.0)]
    results = [
        StressPeriodResult(
            name="Covid-Crash 2020", start=pd.Timestamp("2020-02-19"), end=pd.Timestamp("2020-03-23"),
            max_drawdown_pct=-0.30, included_symbols=["NVDA"], excluded_symbols_no_data=[],
        )
    ]
    commentary = StressTestCommentary(
        resilience_assessment="Konzentriert im KI-Segment.",
        most_vulnerable_exposure="NVDA-Position.",
        contamination_caveat="Kenne den tatsaechlichen Verlauf aus Trainingsdaten.",
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = generate_stress_test_report(portfolio_row, weights, results, commentary, tmp_dir)
        content = Path(report_path).read_text(encoding="utf-8")

    assert "Kontaminations-Vorbehalt" in content
    assert "Kap. 11.2" in content
    assert "nicht" in content.lower()  # Abgrenzung: mechanische Zahlen NICHT betroffen
    assert "Covid-Crash 2020" in content
    assert "-30.00%" in content
    assert "NVDA-Position." in content
    assert "Kenne den tatsaechlichen Verlauf aus Trainingsdaten." in content


def test_stress_test_report_shows_na_for_periods_without_usable_data():
    portfolio_row = {"name": "test"}
    weights = [PositionWeight(symbol="RECENT_IPO", weight=1.0)]
    results = [
        StressPeriodResult(
            name="GFC 2008", start=pd.Timestamp("2008-09-01"), end=pd.Timestamp("2009-03-09"),
            max_drawdown_pct=None, included_symbols=[], excluded_symbols_no_data=["RECENT_IPO"],
        )
    ]
    commentary = StressTestCommentary(
        resilience_assessment="Keine Aussage moeglich.",
        most_vulnerable_exposure="-",
        contamination_caveat="Keine Daten, daher kein Kontaminationsrisiko fuer diese Periode.",
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = generate_stress_test_report(portfolio_row, weights, results, commentary, tmp_dir)
        content = Path(report_path).read_text(encoding="utf-8")

    assert "| n/a |" in content
    assert "RECENT_IPO" in content


# --- Korrelations-Beobachtungen -----------------------------------------------


def _make_buy_order(symbol: str) -> ProposedOrder:
    return ProposedOrder(symbol=symbol, instrument_type="equity", side="buy", quantity=1, rationale="test")


def test_report_shows_no_correlation_warnings_when_none():
    content = _generate(deep_reflection=None, executed_results=[], correlation_cluster_count=1)
    assert "## Korrelations-Beobachtungen" in content
    assert "_Keine Korrelationswarnungen in diesem Lauf._" in content
    assert "**Korrelations-Cluster im aktuellen Portfolio:** 1" in content


def test_report_shows_correlation_warnings_from_executed_results():
    result = ExecutedOrderResult(
        order=_make_buy_order("NVDA"), approved=True, reasons=[], fill_price=100.0,
        correlation_warnings=[CorrelationWarning(candidate_symbol="NVDA", existing_symbol="AMD", correlation=0.91)],
    )
    content = _generate(deep_reflection=None, executed_results=[result], correlation_cluster_count=2)
    assert "NVDA vs. bestehende Position AMD: 0.91" in content
    assert "**Korrelations-Cluster im aktuellen Portfolio:** 2" in content
    assert "_Keine Korrelationswarnungen in diesem Lauf._" not in content


def test_report_shows_na_for_cluster_count_when_unavailable():
    content = _generate(deep_reflection=None, executed_results=[], correlation_cluster_count=None)
    assert "**Korrelations-Cluster im aktuellen Portfolio:** n/a" in content


def test_report_mentions_no_veto_for_correlation():
    """Kernaussage: der Report muss explizit klarstellen, dass dies kein
    Guardrail mit Ablehnungswirkung ist."""
    content = _generate(deep_reflection=None, executed_results=[], correlation_cluster_count=0)
    assert "KEINE automatische Ablehnung" in content
