"""Generates a Markdown report per pipeline run."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.boundary_conditions import BoundaryConditionCheck
from src.data_quality import DataQualityReport
from src.deep_reflection_schema import DeepReflectionRunResult
from src.event_calendar import EarningsWarning, MacroEvent
from src.execution import ExecutedOrderResult
from src.metrics import MetricsResult
from src.risk_guardrails import ForcedStopLossAction
from src.stress_test_schema import StressTestCommentary

if TYPE_CHECKING:
    # Nur für Typannotationen - ein Modulebene-Import von src.stress_test
    # hier würde einen Zirkularimport erzeugen, da stress_test.py umgekehrt
    # generate_stress_test_report aus diesem Modul aufruft.
    from src.stress_test import PositionWeight, StressPeriodResult


def generate_report(
    portfolio_row: sqlite3.Row,
    open_positions: list[sqlite3.Row],
    executed_results: list[ExecutedOrderResult],
    forced_actions: list[ForcedStopLossAction],
    portfolio_commentary: str,
    metrics: MetricsResult,
    data_quality_report: DataQualityReport,
    deep_reflection: DeepReflectionRunResult | None,
    triggered_boundary_conditions: list[BoundaryConditionCheck],
    still_open_boundary_conditions: list[BoundaryConditionCheck],
    correlation_cluster_count: int | None,
    earnings_warnings: list[EarningsWarning],
    macro_events: list[MacroEvent],
    reports_dir: str,
) -> Path:
    lines = []
    now = datetime.now()
    lines.append(f"# Portfolio-Report - {portfolio_row['name']}")
    lines.append(f"_Erstellt am {now.strftime('%Y-%m-%d %H:%M')}_\n")

    lines.append("## Kennzahlen")
    lines.append("| Kennzahl | Wert |")
    lines.append("|---|---|")
    lines.append(f"| NAV | {metrics.current_nav:,.2f} {portfolio_row['currency']} |")
    lines.append(f"| Gesamtrendite | {metrics.total_return_pct:.2%} |")
    if metrics.last_period_return_pct is not None:
        lines.append(f"| Rendite letzte Periode | {metrics.last_period_return_pct:.2%} |")
    if metrics.annualized_volatility_pct is not None:
        lines.append(f"| Volatilität (annualisiert) | {metrics.annualized_volatility_pct:.2%} |")
    if metrics.sharpe_ratio is not None:
        lines.append(f"| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |")
    lines.append(f"| Max. Drawdown | {metrics.max_drawdown_pct:.2%} |")
    lines.append(f"| Benchmark-Rendite ({portfolio_row['benchmark_symbol']}) | {metrics.benchmark_total_return_pct:.2%} |")
    lines.append(f"| Alpha vs. Benchmark | {metrics.alpha_pct:.2%} |")
    lines.append(f"| Momentum-Baseline-Rendite (Kap. 6.9) | {metrics.baseline_total_return_pct:.2%} |")
    lines.append(f"| Alpha vs. Momentum-Baseline | {metrics.baseline_alpha_pct:.2%} |\n")

    lines.append("## Datenqualität")
    if data_quality_report.has_findings:
        if data_quality_report.price_deviations:
            lines.append("**Kursabweichungen yfinance vs. Alpaca (Stichprobe, >1%):**")
            lines.append("| Symbol | yfinance | Alpaca | Abweichung |")
            lines.append("|---|---|---|---|")
            for d in data_quality_report.price_deviations:
                lines.append(
                    f"| {d.symbol} | {d.yfinance_price:.2f} | {d.alpaca_price:.2f} | {d.deviation_pct:+.2%} |"
                )
            lines.append("")
        if data_quality_report.missing_trading_days:
            lines.append("**Fehlende Handelstage (yfinance-Kurshistorie):**")
            for issue in data_quality_report.missing_trading_days:
                lines.append(f"- **{issue.symbol}**: {issue.detail}")
            lines.append("")
        if data_quality_report.outlier_moves:
            lines.append("**Ausreisser-Tagesbewegungen (möglicher Datenfehler):**")
            for issue in data_quality_report.outlier_moves:
                lines.append(f"- **{issue.symbol}**: {issue.detail}")
            lines.append("")
    else:
        lines.append("_Keine Auffälligkeiten._")
    lines.append("")

    lines.append("## Randbedingungen (Kap. 7)")
    if triggered_boundary_conditions:
        lines.append("**Ausgelöst in diesem Lauf:**")
        for c in triggered_boundary_conditions:
            lines.append(f"- **{c.symbol}**: {c.description}")
    if still_open_boundary_conditions:
        lines.append("**Weiterhin offen:**")
        for c in still_open_boundary_conditions:
            note = "nicht automatisch prüfbar" if c.check_type == "qualitative" else f"Schwelle {c.threshold_price:.2f}"
            lines.append(f"- **{c.symbol}** ({note}): {c.description}")
    if not triggered_boundary_conditions and not still_open_boundary_conditions:
        lines.append("_Keine offenen Randbedingungen._")
    lines.append("")

    lines.append("## Korrelations-Beobachtungen (Beobachtungsgrösse, kein Guardrail)")
    lines.append(
        "Rollierende 60-Tage-Korrelation der Tagesrenditen über das Anlage-Universum. "
        "Eine Korrelation > 0.85 zu einer bestehenden Position wird hier dokumentiert, "
        "löst aber KEINE automatische Ablehnung aus - anders als die harten Kap.-6.8-Limiten."
    )
    correlation_warnings = [w for r in executed_results for w in r.correlation_warnings]
    if correlation_warnings:
        lines.append("**Warnungen in diesem Lauf:**")
        for w in correlation_warnings:
            lines.append(f"- {w.candidate_symbol} vs. bestehende Position {w.existing_symbol}: {w.correlation:.2f}")
    else:
        lines.append("_Keine Korrelationswarnungen in diesem Lauf._")
    cluster_display = str(correlation_cluster_count) if correlation_cluster_count is not None else "n/a"
    lines.append(f"**Korrelations-Cluster im aktuellen Portfolio:** {cluster_display}")
    lines.append("")

    lines.append("## Markt-Phasen-Abgleich (Beobachtungsgrösse, kein Guardrail)")
    lines.append(
        "Regelbasierte Bull/Bear/Seitwärts-Klassifikation je Symbol (SMA20/50 + rollierende "
        "20-Tage-Volatilität, siehe src/market_phase.py) gegen Claudes optionale, je Order "
        "angegebene Zyklus-Position (Kap. 3, SYSTEM_PROMPT-Anforderung 1) verglichen. Ein "
        "Widerspruch wird hier dokumentiert, löst aber KEINE automatische Ablehnung aus - die "
        "Regel-Klassifikation ist ihrerseits nur ein einfaches technisches Signal, kein Beweis, "
        "dass Claudes Einschätzung falsch liegt."
    )
    phase_contradictions = [r.market_phase_contradiction for r in executed_results if r.market_phase_contradiction]
    if phase_contradictions:
        lines.append("**Widersprüche in diesem Lauf:**")
        for c in phase_contradictions:
            lines.append(f"- {c.detail}")
    else:
        lines.append("_Keine Widersprüche in diesem Lauf._")
    lines.append("")

    lines.append("## Event-Kalender-Hinweis (informativ, kein Verbot)")
    lines.append(
        "Bevorstehende Quartalsberichte (yfinance) und hardcodierte FOMC-/CPI-Termine "
        "innerhalb der nächsten 3 Handelstage - rein informativ, löst KEIN automatisches "
        "Verbot aus. Claude erhält denselben Hinweis im Tagesprompt."
    )
    if earnings_warnings:
        lines.append("**Bevorstehende Quartalsberichte:**")
        for w in earnings_warnings:
            lines.append(f"- {w.symbol} berichtet am {w.earnings_date} ({w.trading_days_until} Handelstag(e))")
    if macro_events:
        lines.append("**Bevorstehende Makro-Termine:**")
        for m in macro_events:
            lines.append(f"- {m.name} am {m.event_date} ({m.trading_days_until} Handelstag(e))")
    if not earnings_warnings and not macro_events:
        lines.append("_Keine bevorstehenden Ereignisse innerhalb des Zeitfensters._")
    lines.append("")

    if deep_reflection is not None:
        primary = deep_reflection.primary
        lines.append("## Monatliche Tiefenreflexion (Kap. 6.12.3)")
        if primary.theses_confirmed:
            lines.append("**Bestätigte Thesen:**")
            for thesis in primary.theses_confirmed:
                lines.append(f"- {thesis}")
        if primary.theses_falsified_or_overdue:
            lines.append("**Widerlegte/überfällige Thesen:**")
            for thesis in primary.theses_falsified_or_overdue:
                lines.append(f"- {thesis}")
        lines.append(f"**Pattern-Matching-Bedenken:** {primary.pattern_matching_concerns or '-'}")
        lines.append(f"**Portfolio-Haltung:** {primary.portfolio_stance_assessment or '-'}")
        lines.append(f"**Zusammenfassung:** {primary.reflection_commentary}")
        lines.append("")

        lines.append("### Selbstkonsistenz-Prüfung (zwei unabhängige Aufrufe desselben Prompts)")
        lines.append(
            "Ausschliesslich für die monatliche Tiefenreflexion (aus Kostengründen nicht im "
            "täglichen Ablauf): Claude wird für dieselbe Periode zweimal mit identischem Prompt "
            "aufgerufen und die Kernaussagen (bestätigte/widerlegte Thesen) verglichen. Bei einer "
            "Abweichung erfolgt KEINE automatische Konfliktlösung - beide Antworten werden unten "
            "dokumentiert; die Reflexion oben (1. Aufruf) dient unverändert als Grundlage für den "
            "nächsten Lauf."
        )
        if deep_reflection.consistency.consistent:
            lines.append("✅ Beide Aufrufe stimmen in den Kernaussagen überein.")
        else:
            lines.append("⚠️ **Abweichung zwischen den beiden Aufrufen:**")
            for detail in deep_reflection.consistency.mismatch_details:
                lines.append(f"- {detail}")
            secondary = deep_reflection.secondary
            lines.append("")
            lines.append("**Zweite Antwort (vollständig, zum Vergleich):**")
            if secondary.theses_confirmed:
                lines.append("- Bestätigte Thesen (2. Aufruf):")
                for thesis in secondary.theses_confirmed:
                    lines.append(f"  - {thesis}")
            if secondary.theses_falsified_or_overdue:
                lines.append("- Widerlegte/überfällige Thesen (2. Aufruf):")
                for thesis in secondary.theses_falsified_or_overdue:
                    lines.append(f"  - {thesis}")
            lines.append(f"- Pattern-Matching-Bedenken (2. Aufruf): {secondary.pattern_matching_concerns or '-'}")
            lines.append(f"- Portfolio-Haltung (2. Aufruf): {secondary.portfolio_stance_assessment or '-'}")
            lines.append(f"- Zusammenfassung (2. Aufruf): {secondary.reflection_commentary}")
        lines.append("")

    if forced_actions:
        lines.append("## ⚠️ Automatische Stop-Loss-Schliessungen (dokumentationspflichtig)")
        for action in forced_actions:
            lines.append(f"- **{action.symbol}**: {action.documentation}")
        lines.append("")

    lines.append("## Offene Positionen")
    if open_positions:
        lines.append("| Symbol | Typ | Seite | Menge | Ø Einstand | Stop-Loss |")
        lines.append("|---|---|---|---|---|---|")
        for p in open_positions:
            stop = f"{p['stop_loss_price']:.2f}" if p["stop_loss_price"] else "-"
            lines.append(
                f"| {p['symbol']} | {p['instrument_type']} | {p['side']} | {p['quantity']:.4f} | "
                f"{p['avg_entry_price']:.2f} | {stop} |"
            )
    else:
        lines.append("_Keine offenen Positionen._")
    lines.append("")

    lines.append("## Trades in diesem Lauf")
    lines.append(
        "_Vol-Skalierung: Positionsgrösse buy/short-Orders vor der Guardrail-Prüfung mit "
        "Universums-Ø-Volatilität / Symbol-Volatilität skaliert (rollierende 20-Tage-annualisierte "
        "Volatilität, Faktor geclippt auf das in risk_config.yaml konfigurierte Band, Default "
        "0.5x-1.5x). Konviktion: zusätzlicher, bewusst SCHWACHER Multiplikator (hoch ×1.15, "
        "mittel ×1.0, niedrig ×0.8) auf Basis von Claudes optionaler Selbsteinschätzung je Order "
        "- schwach gehalten, weil die Konfidenz-Selbsteinschätzung von Sprachmodellen bekanntermassen "
        "schlecht kalibriert ist (siehe src/position_sizing.py). Beide Faktoren sind eine Verfeinerung "
        "INNERHALB der Kap.-6.8-Limiten, kein zusätzliches Veto. \"-\" = nicht anwendbar (sell/cover, "
        "oder bei Vol-Skalierung: keine Volatilitätsdaten für das Symbol)._"
    )
    if executed_results:
        lines.append("| Symbol | Seite | Status | Vol-Skalierung | Konviktion | Begründung |")
        lines.append("|---|---|---|---|---|---|")
        for r in executed_results:
            status = "✅ ausgeführt" if r.approved else "❌ abgelehnt"
            reason = "; ".join(r.reasons) if r.reasons else r.order.rationale
            if r.volatility_scaling is not None:
                vs = r.volatility_scaling
                scaling_display = (
                    f"{vs.scaling_factor:.2f}x (Vol {vs.annualized_volatility:.1%} vs. "
                    f"Ø {vs.universe_avg_volatility:.1%})"
                )
            else:
                scaling_display = "-"
            if r.conviction_scaling_factor is not None:
                conviction_label = r.order.conviction.value if r.order.conviction else "unbekannt"
                conviction_display = f"{conviction_label} ({r.conviction_scaling_factor:.2f}x)"
            else:
                conviction_display = "-"
            lines.append(
                f"| {r.order.symbol} | {r.order.side.value} | {status} | {scaling_display} | "
                f"{conviction_display} | {reason} |"
            )
    else:
        lines.append("_Keine Order-Vorschläge in diesem Lauf._")
    lines.append("")

    lines.append("## Kommentar des Modells")
    lines.append(portfolio_commentary or "_Kein Kommentar._")
    lines.append("")

    content = "\n".join(lines)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    report_file = reports_path / f"report_{now.strftime('%Y-%m-%d_%H%M%S')}.md"
    report_file.write_text(content, encoding="utf-8")
    return report_file


def generate_stress_test_report(
    portfolio_row: sqlite3.Row,
    position_weights: list["PositionWeight"],
    period_results: list["StressPeriodResult"],
    commentary: StressTestCommentary,
    reports_dir: str,
) -> Path:
    """Eigenständiges Report-Artefakt für den manuellen historischen
    Stresstest (Thesis Kap. 11.2, src/stress_test.py) - bewusst getrennt
    vom täglichen Portfolio-Report (generate_report oben), da ein Stresstest
    einen festen historischen Zeitraum untersucht statt "Stand heute".

    Der Kontaminations-Hinweis steht ZWINGEND vor jedem Ergebnis - siehe
    stress_test.py's Modul-Docstring für die vollständige Begründung.
    """
    lines = []
    now = datetime.now()
    lines.append(f"# Historischer Stresstest - {portfolio_row['name']}")
    lines.append(f"_Erstellt am {now.strftime('%Y-%m-%d %H:%M')}_\n")

    lines.append("## ⚠️ Kontaminations-Vorbehalt (Thesis Kap. 11.2)")
    lines.append(
        "Die mechanischen Drawdown-Kennzahlen unten sind reine Kursdaten-Berechnungen "
        "ohne LLM-Beteiligung und daher **nicht** von diesem Vorbehalt betroffen. Der "
        "qualitative Kommentar im Abschnitt \"Analytische Einordnung\" hingegen betrifft "
        "öffentlich extensiv dokumentierte historische Krisen - Claude kennt deren "
        "tatsächlichen Verlauf mit hoher Wahrscheinlichkeit bereits aus Trainingsdaten. "
        "Diese Kommentare testen daher Erklärungsfähigkeit im Rückblick, **nicht** echte "
        "blinde Risikoeinschätzung. Siehe Claudes eigene Selbsteinschätzung unten "
        "(\"Kontaminations-Selbsteinschätzung\")."
    )
    lines.append("")

    lines.append("## Aktuelle Positionsgewichte")
    lines.append("| Symbol | Gewicht |")
    lines.append("|---|---|")
    for pw in position_weights:
        lines.append(f"| {pw.symbol} | {pw.weight:+.1%} |")
    lines.append("")

    lines.append("## Mechanische Ergebnisse (kein Kontaminationsrisiko)")
    lines.append("| Krise | Zeitraum | Max. Drawdown | Einbezogene Symbole | Ausgeschlossen (keine Historie) |")
    lines.append("|---|---|---|---|---|")
    for r in period_results:
        drawdown = f"{r.max_drawdown_pct:.2%}" if r.max_drawdown_pct is not None else "n/a"
        lines.append(
            f"| {r.name} | {r.start.date()} bis {r.end.date()} | {drawdown} | "
            f"{', '.join(r.included_symbols) or '-'} | {', '.join(r.excluded_symbols_no_data) or '-'} |"
        )
    lines.append("")

    lines.append("## Analytische Einordnung (Claude - kontaminationsbehaftet)")
    lines.append(f"**Robustheits-Einschätzung:** {commentary.resilience_assessment}")
    lines.append(f"**Anfälligste Exposure:** {commentary.most_vulnerable_exposure}")
    lines.append(f"**Kontaminations-Selbsteinschätzung:** {commentary.contamination_caveat}")
    lines.append("")

    content = "\n".join(lines)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    report_file = reports_path / f"stress_test_{now.strftime('%Y-%m-%d_%H%M%S')}.md"
    report_file.write_text(content, encoding="utf-8")
    return report_file
