"""Tests for src/reporting.py's Kap.-6.12.3 report section: the monthly deep
reflection must appear only on the run it was actually generated on, and
render its fields when present.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

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


def _generate(deep_reflection) -> str:
    portfolio_row = {"name": "test", "currency": "USD", "benchmark_symbol": "SPY"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = generate_report(
            portfolio_row, [], [], [], "", make_metrics(), EMPTY_DQ_REPORT, deep_reflection, tmp_dir
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
