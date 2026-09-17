"""Generates a Markdown report per pipeline run."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from src.data_quality import DataQualityReport
from src.execution import ExecutedOrderResult
from src.metrics import MetricsResult
from src.risk_guardrails import ForcedStopLossAction


def generate_report(
    portfolio_row: sqlite3.Row,
    open_positions: list[sqlite3.Row],
    executed_results: list[ExecutedOrderResult],
    forced_actions: list[ForcedStopLossAction],
    portfolio_commentary: str,
    metrics: MetricsResult,
    data_quality_report: DataQualityReport,
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
    if executed_results:
        lines.append("| Symbol | Seite | Status | Begründung |")
        lines.append("|---|---|---|---|")
        for r in executed_results:
            status = "✅ ausgeführt" if r.approved else "❌ abgelehnt"
            reason = "; ".join(r.reasons) if r.reasons else r.order.rationale
            lines.append(f"| {r.order.symbol} | {r.order.side.value} | {status} | {reason} |")
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
