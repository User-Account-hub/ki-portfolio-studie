"""Tests for src/reporting.py's Kap.-6.12.3 report section: the monthly deep
reflection must appear only on the run it was actually generated on, and
render its fields when present.
"""
from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pandas as pd

from src.boundary_conditions import BoundaryConditionCheck
from src.correlation import CorrelationWarning
from src.data_quality import DataQualityReport, StalePosition
from src.event_calendar import EarningsWarning, MacroEvent
from src.deep_reflection_schema import DeepReflectionOutput, DeepReflectionRunResult, SelfConsistencyCheckResult
from src.execution import ExecutedOrderResult
from src.market_phase import MarketPhase, MarketPhaseContradiction
from src.metrics import MetricsResult
from src.order_schema import CyclePosition, ProposedOrder
from src.reporting import generate_report, generate_stress_test_report
from src.stress_test import PositionWeight, StressPeriodResult
from src.stress_test_schema import StressTestCommentary

EMPTY_DQ_REPORT = DataQualityReport(price_deviations=[], missing_trading_days=[], outlier_moves=[])


def make_metrics(**overrides) -> MetricsResult:
    defaults = dict(
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
        information_ratio=None,
        qqq_total_return_pct=0.0,
        alpha_vs_qqq_pct=0.0,
        segment_basket_total_return_pct=0.0,
        alpha_vs_segment_basket_pct=0.0,
    )
    defaults.update(overrides)
    return MetricsResult(**defaults)


def _generate(
    deep_reflection,
    triggered_boundary_conditions=None,
    still_open_boundary_conditions=None,
    executed_results=None,
    correlation_cluster_count=None,
    earnings_warnings=None,
    macro_events=None,
    metrics=None,
    open_positions=None,
    data_quality_report=None,
) -> str:
    portfolio_row = {"name": "test", "currency": "USD", "benchmark_symbol": "SPY"}
    metrics = metrics if metrics is not None else make_metrics()
    data_quality_report = data_quality_report if data_quality_report is not None else EMPTY_DQ_REPORT
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = generate_report(
            portfolio_row, open_positions or [], executed_results or [], [], "", metrics, data_quality_report,
            deep_reflection,
            triggered_boundary_conditions or [], still_open_boundary_conditions or [], correlation_cluster_count,
            earnings_warnings or [], macro_events or [],
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
    run_result = DeepReflectionRunResult(
        primary=reflection, secondary=reflection, consistency=SelfConsistencyCheckResult(consistent=True)
    )
    content = _generate(deep_reflection=run_result)
    assert "## Monatliche Tiefenreflexion (Kap. 6.12.3)" in content
    assert "NVDA-These bestätigt" in content
    assert "CCJ-These überfällig" in content
    assert "Alles im Rahmen der Methodik." in content


# --- Selbstkonsistenz-Prüfung (2026-09-19) -----------------------------------


def test_report_shows_consistent_when_both_calls_agree():
    reflection = DeepReflectionOutput(reflection_commentary="Alles im Rahmen der Methodik.")
    run_result = DeepReflectionRunResult(
        primary=reflection, secondary=reflection, consistency=SelfConsistencyCheckResult(consistent=True)
    )
    content = _generate(deep_reflection=run_result)
    assert "Selbstkonsistenz-Prüfung" in content
    assert "Beide Aufrufe stimmen in den Kernaussagen überein." in content
    assert "Zweite Antwort" not in content  # nur bei Abweichung vollständig ausgegeben


def test_report_documents_both_responses_on_mismatch():
    """Kernanforderung: keine automatische Konfliktlösung - bei Abweichung
    müssen BEIDE Antworten vollständig im Report erscheinen."""
    primary = DeepReflectionOutput(
        theses_confirmed=["NVDA-These bestätigt"],
        reflection_commentary="Erster Aufruf: These bestätigt.",
    )
    secondary = DeepReflectionOutput(
        theses_falsified_or_overdue=["NVDA-These widerlegt"],
        reflection_commentary="Zweiter Aufruf: These widerlegt.",
    )
    run_result = DeepReflectionRunResult(
        primary=primary,
        secondary=secondary,
        consistency=SelfConsistencyCheckResult(
            consistent=False,
            mismatch_details=["theses_confirmed: nur im 1. Aufruf genannt: ['nvda-these bestätigt']"],
        ),
    )
    content = _generate(deep_reflection=run_result)
    assert "Abweichung zwischen den beiden Aufrufen" in content
    assert "nur im 1. Aufruf genannt" in content
    # Erste Antwort (oben, als "die" Reflexion) UND zweite Antwort (vollständig) müssen beide vorkommen.
    assert "Erster Aufruf: These bestätigt." in content
    assert "Zweiter Aufruf: These widerlegt." in content
    assert "NVDA-These widerlegt" in content


# --- Kennzahlen: Information Ratio (2026-09-20) -------------------------------


def test_report_omits_information_ratio_row_when_not_computable():
    content = _generate(deep_reflection=None, metrics=make_metrics(information_ratio=None))
    assert "Information Ratio" not in content


def test_report_shows_information_ratio_next_to_alpha():
    """'Neben' der Benchmark-Zeilen = innerhalb desselben Kennzahlen-Blocks,
    direkt nach Alpha vs. QQQ (siehe test_report_shows_qqq_benchmark_rows für
    die Reihenfolge Benchmark -> Alpha vs. Benchmark -> QQQ -> Alpha vs. QQQ
    -> Information Ratio)."""
    content = _generate(deep_reflection=None, metrics=make_metrics(alpha_pct=0.05, information_ratio=0.42))
    lines = content.splitlines()
    alpha_qqq_index = next(i for i, line in enumerate(lines) if line.startswith("| Alpha vs. QQQ"))
    assert "0.42" in lines[alpha_qqq_index + 1]
    assert "Information Ratio" in lines[alpha_qqq_index + 1]


def test_report_formats_negative_information_ratio():
    content = _generate(deep_reflection=None, metrics=make_metrics(information_ratio=-1.23))
    assert "| Information Ratio (Alpha / Tracking Error) | -1.23 |" in content


# --- QQQ-Vergleichsindex (Kap. 6.9 Erweiterung, 2026-09-21) -------------------


def test_report_shows_qqq_benchmark_rows_alongside_spy():
    """Kernanforderung: QQQ ERGAENZT die bestehende SPY-Benchmark-Zeile,
    ersetzt sie nicht - beide muessen gleichzeitig im Report stehen."""
    content = _generate(
        deep_reflection=None,
        metrics=make_metrics(
            benchmark_total_return_pct=0.08, alpha_pct=0.02,
            qqq_total_return_pct=0.11, alpha_vs_qqq_pct=-0.01,
        ),
    )
    assert "| Benchmark-Rendite (SPY) | 8.00% |" in content
    assert "| Alpha vs. Benchmark | 2.00% |" in content
    assert "| Benchmark-Rendite (QQQ, Nasdaq-100, sektorspezifisch) | 11.00% |" in content
    assert "| Alpha vs. QQQ | -1.00% |" in content


def test_report_qqq_rows_always_shown_not_conditional():
    """Anders als Information Ratio (optional, kann None sein) sind die
    QQQ-Kennzahlen Pflichtfelder auf MetricsResult - die Zeilen muessen
    immer erscheinen, nicht nur bei einem bestimmten Wert."""
    content = _generate(deep_reflection=None, metrics=make_metrics())
    assert "Benchmark-Rendite (QQQ" in content
    assert "Alpha vs. QQQ" in content


# --- Segment-ETF-Korb (Kap. 6.9 Erweiterung, 2026-09-21) ----------------------


def test_report_shows_segment_basket_row_alongside_other_comparators():
    """Kernanforderung: der Segment-Korb ERGAENZT die bestehenden
    Vergleichspunkte (SPY, QQQ, Momentum-Baseline), ersetzt sie nicht - alle
    muessen gleichzeitig im Report stehen."""
    content = _generate(
        deep_reflection=None,
        metrics=make_metrics(
            benchmark_total_return_pct=0.08, alpha_pct=0.02,
            qqq_total_return_pct=0.11, alpha_vs_qqq_pct=-0.01,
            baseline_total_return_pct=0.09, baseline_alpha_pct=0.01,
            segment_basket_total_return_pct=0.15, alpha_vs_segment_basket_pct=-0.05,
        ),
    )
    assert "| Benchmark-Rendite (SPY) | 8.00% |" in content
    assert "| Benchmark-Rendite (QQQ, Nasdaq-100, sektorspezifisch) | 11.00% |" in content
    assert "| Momentum-Baseline-Rendite (Kap. 6.9) | 9.00% |" in content
    assert "Segment-ETF-Korb-Rendite (SMH/URA/ICLN" in content
    assert "| 15.00% |" in content
    assert "| Alpha vs. Segment-ETF-Korb | -5.00% |" in content


def test_report_segment_basket_rows_always_shown_not_conditional():
    content = _generate(deep_reflection=None, metrics=make_metrics())
    assert "Segment-ETF-Korb-Rendite" in content
    assert "Alpha vs. Segment-ETF-Korb" in content


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


# --- Markt-Phasen-Abgleich (2026-09-20) ---------------------------------------


def test_report_shows_no_market_phase_contradictions_when_none():
    content = _generate(deep_reflection=None, executed_results=[])
    assert "## Markt-Phasen-Abgleich" in content
    assert "_Keine Widersprüche in diesem Lauf._" in content


def test_report_shows_market_phase_contradiction_from_executed_results():
    contradiction = MarketPhaseContradiction(
        symbol="NVDA",
        claude_cycle_position=CyclePosition.MANIA,
        rule_based_phase=MarketPhase.BEAR,
        detail="NVDA: Claude ordnet die Zyklus-Position als 'mania' ein, die regelbasierte "
        "Marktphasen-Klassifikation (SMA20/50 + Volatilität) sieht das Symbol aber in 'bear'.",
    )
    result = ExecutedOrderResult(
        order=_make_buy_order("NVDA"), approved=True, reasons=[], fill_price=100.0,
        market_phase_contradiction=contradiction,
    )
    content = _generate(deep_reflection=None, executed_results=[result])
    assert "mania" in content
    assert "bear" in content
    assert "_Keine Widersprüche in diesem Lauf._" not in content


def test_report_mentions_no_veto_for_market_phase_check():
    """Kernaussage: der Report muss im Markt-Phasen-Abschnitt selbst
    klarstellen, dass ein Widerspruch keine automatische Ablehnung ausloest
    (nicht nur an anderer Stelle im Dokument, z.B. beim Korrelations-Check)."""
    content = _generate(deep_reflection=None, executed_results=[])
    section = content.split("## Markt-Phasen-Abgleich")[1].split("## Event-Kalender-Hinweis")[0]
    assert "KEINE automatische Ablehnung" in section


# --- Event-Kalender-Hinweis --------------------------------------------------


def test_report_shows_no_upcoming_events_when_none():
    content = _generate(deep_reflection=None)
    assert "## Event-Kalender-Hinweis" in content
    assert "_Keine bevorstehenden Ereignisse innerhalb des Zeitfensters._" in content


def test_report_shows_earnings_and_macro_events():
    earnings = [EarningsWarning(symbol="NVDA", earnings_date=datetime.date(2026, 9, 21), trading_days_until=1)]
    macro = [MacroEvent(name="FOMC", event_date=datetime.date(2026, 10, 28), trading_days_until=2)]
    content = _generate(deep_reflection=None, earnings_warnings=earnings, macro_events=macro)
    assert "NVDA berichtet am 2026-09-21 (1 Handelstag(e))" in content
    assert "FOMC am 2026-10-28 (2 Handelstag(e))" in content
    assert "_Keine bevorstehenden Ereignisse innerhalb des Zeitfensters._" not in content


def test_report_event_calendar_section_mentions_no_veto():
    content = _generate(deep_reflection=None)
    assert "kein Verbot" in content or "KEIN automatisches Verbot" in content


# --- Fehlende Kursdaten / offene Positionen (Kap. 12.7-Nachbarthema, 2026-09-21) ---


def make_open_position_row(**overrides) -> dict:
    defaults = dict(
        symbol="ACME", instrument_type="equity", side="long", quantity=10.0,
        avg_entry_price=100.0, stop_loss_price=None,
    )
    defaults.update(overrides)
    return defaults


def test_report_omits_stale_positions_section_when_none():
    content = _generate(deep_reflection=None)
    assert "Kursdaten fehlen - manuelle Prüfung nötig" not in content


def test_report_shows_stale_positions_section_when_present():
    dq_report = DataQualityReport(
        price_deviations=[], missing_trading_days=[], outlier_moves=[],
        stale_positions=[
            StalePosition(symbol="DELISTED", instrument_type="equity", side="long", quantity=10.0, avg_entry_price=50.0)
        ],
    )
    content = _generate(deep_reflection=None, data_quality_report=dq_report)
    assert "## ⚠️ Kursdaten fehlen - manuelle Prüfung nötig" in content
    assert "**DELISTED**" in content
    assert "Ø-Einstand 50.00" in content


def test_report_mentions_no_automatic_order_for_stale_positions():
    """Kernaussage des Auftrags: der Report muss explizit klarstellen, dass
    KEINE automatische Order ausgelöst wurde."""
    dq_report = DataQualityReport(
        price_deviations=[], missing_trading_days=[], outlier_moves=[],
        stale_positions=[
            StalePosition(symbol="DELISTED", instrument_type="equity", side="long", quantity=10.0, avg_entry_price=50.0)
        ],
    )
    content = _generate(deep_reflection=None, data_quality_report=dq_report)
    assert "KEINE automatische Order" in content


def test_report_marks_affected_row_in_open_positions_table():
    dq_report = DataQualityReport(
        price_deviations=[], missing_trading_days=[], outlier_moves=[],
        stale_positions=[
            StalePosition(symbol="DELISTED", instrument_type="equity", side="long", quantity=10.0, avg_entry_price=50.0)
        ],
    )
    positions = [make_open_position_row(symbol="AAPL"), make_open_position_row(symbol="DELISTED")]
    content = _generate(deep_reflection=None, open_positions=positions, data_quality_report=dq_report)
    table_lines = {
        line.split("|")[1].strip(): line
        for line in content.splitlines()
        if line.startswith("| ") and "Symbol" not in line and "---" not in line
    }
    assert "⚠️ Kursdaten fehlen - manuelle Prüfung nötig" not in table_lines["AAPL"]  # unbetroffen
    assert "⚠️ Kursdaten fehlen - manuelle Prüfung nötig" in table_lines["DELISTED"]


def test_report_datenqualitaet_section_cross_references_stale_positions():
    """Die allgemeine 'Datenqualität'-Sektion darf bei ausschliesslich
    fehlenden Kursdaten nicht faelschlich leer erscheinen (has_findings=True,
    aber keiner der drei alten Unterabschnitte greift) - sie muss auf den
    eigenen, prominenteren Abschnitt verweisen."""
    dq_report = DataQualityReport(
        price_deviations=[], missing_trading_days=[], outlier_moves=[],
        stale_positions=[
            StalePosition(symbol="DELISTED", instrument_type="equity", side="long", quantity=10.0, avg_entry_price=50.0)
        ],
    )
    content = _generate(deep_reflection=None, data_quality_report=dq_report)
    section = content.split("## Datenqualität")[1].split("## ")[0]
    assert "1 offene Position(en)" in section
    assert "_Keine Auffälligkeiten._" not in section
