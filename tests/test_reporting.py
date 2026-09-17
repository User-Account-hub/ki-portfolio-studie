"""Tests for src/reporting.py's Kap.-6.12.3 report section: the monthly deep
reflection must appear only on the run it was actually generated on, and
render its fields when present.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.boundary_conditions import BoundaryConditionCheck
from src.data_quality import DataQualityReport
from src.deep_reflection_schema import DeepReflectionOutput
from src.metrics import MetricsResult
from src.reporting import generate_report

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


def _generate(deep_reflection, triggered_boundary_conditions=None, still_open_boundary_conditions=None) -> str:
    portfolio_row = {"name": "test", "currency": "USD", "benchmark_symbol": "SPY"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = generate_report(
            portfolio_row, [], [], [], "", make_metrics(), EMPTY_DQ_REPORT, deep_reflection,
            triggered_boundary_conditions or [], still_open_boundary_conditions or [], tmp_dir,
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
