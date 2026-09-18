"""Historische Stresstests (Thesis Kap. 11.2) - manuelles, on-demand-Tool
(`python -m src.stress_test`), NICHT Teil des automatischen täglichen/
wöchentlichen Pipeline-Crons: rechenintensiver (mehrjährige Kurshistorien
+ ein zusätzlicher Claude-Aufruf) und methodisch ein Validierungswerkzeug
für die Thesis, kein operatives Tagessignal.

Zweigeteilt:

(A) MECHANISCH (kein Kontaminationsrisiko): Backtest der AKTUELLEN offenen
    Positionen (gewichtet nach heutigem Marktwert, Short-Positionen mit
    negativem Vorzeichen) gegen echte historische Kursdaten fester,
    dokumentierter Krisenfenster (STRESS_PERIODS unten) - reine
    Kursdaten-Arithmetik, analog zu momentum_baseline.py, das aus demselben
    Grund KEIN Kontaminationsrisiko trägt.

(B) ANALYTISCH (Kontaminationsrisiko - siehe stress_test_prompt.py): ein
    separater Claude-Aufruf kommentiert die mechanischen Ergebnisse. Claude
    kennt aus Trainingsdaten mit hoher Wahrscheinlichkeit bereits, wie sich
    diese öffentlich extensiv dokumentierten Krisen tatsächlich entwickelt
    haben - jede qualitative Einschätzung ist daher potenziell durch
    Rückschau-Wissen (Hindsight Bias) verzerrt statt echter blinder
    Risikoanalyse. Deshalb zwingend: der Prompt fordert eine explizite
    Selbsteinschätzung dieses Risikos an (Pflichtfeld im Schema), UND der
    generierte Report trägt einen unübersehbaren Warnhinweis (siehe
    reporting.generate_stress_test_report).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import pandas as pd

from src import data_fetch, db, reporting, stress_test_prompt, stress_test_schema
from src.claude_client import get_trading_decision
from src.config import AppConfig

log = logging.getLogger("stress_test")


@dataclass(frozen=True)
class StressPeriod:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


# Feste, dokumentierte Krisenfenster (2026-09-18) - bewusst hier hart
# kodiert (wie PHASE2_START in metrics.py), nicht konfigurierbar: die
# Reproduzierbarkeit der Thesis-Ergebnisse hängt an exakt diesen Zeiträumen.
STRESS_PERIODS = [
    StressPeriod("Globale Finanzkrise 2008", pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-09")),
    StressPeriod("Covid-Crash 2020", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
    StressPeriod("Zinswende-Bärenmarkt 2022", pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-13")),
    StressPeriod("Q4-Selloff 2018", pd.Timestamp("2018-10-01"), pd.Timestamp("2018-12-24")),
]


@dataclass(frozen=True)
class PositionWeight:
    symbol: str
    # Signiert: Long-Positionen positiv, Short-Positionen negativ (ein Short
    # profitiert von fallenden Kursen) - normalisiert auf Summe(|weight|) = 1
    # über alle Positionen MIT verfügbarem aktuellem Kurs.
    weight: float


@dataclass(frozen=True)
class StressPeriodResult:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    max_drawdown_pct: float | None  # None, wenn kein einziges Symbol Historie im Zeitraum hat
    included_symbols: list[str]
    excluded_symbols_no_data: list[str]


def compute_position_weights(
    position_rows: list,
    current_prices: dict[str, float],
) -> list[PositionWeight]:
    """Aggregiert offene Positionen PRO SYMBOL (falls sowohl eine Long- als
    auch eine Short-Position im selben Symbol offen ist) und normalisiert
    auf Summe(|weight|) = 1. Positionen ohne aktuellen Kurs werden
    übersprungen (können nicht gewichtet werden), nicht mit Gewicht 0
    aufgeführt - sie fehlen schlicht in der Rückgabe."""
    raw_by_symbol: dict[str, float] = {}
    for r in position_rows:
        price = current_prices.get(r["symbol"])
        if price is None:
            continue
        sign = 1.0 if r["side"] == "long" else -1.0
        raw_by_symbol[r["symbol"]] = raw_by_symbol.get(r["symbol"], 0.0) + sign * r["quantity"] * price

    total_abs = sum(abs(v) for v in raw_by_symbol.values())
    if total_abs == 0:
        return []
    return [PositionWeight(symbol=s, weight=v / total_abs) for s, v in raw_by_symbol.items()]


def compute_stress_period_result(
    period: StressPeriod,
    position_weights: list[PositionWeight],
    price_histories: dict[str, pd.Series],
) -> StressPeriodResult:
    """Rekonstruiert einen linear gewichteten Portfolio-Index über den
    Krisenzeitraum (aktuelle Gewichte, UNVERÄNDERT über den gesamten
    historischen Zeitraum gehalten - das entspricht der natürlichen
    Stresstest-Frage "was, wenn genau MEINE heutigen Positionen durch
    dieses historische Ereignis liefen", nicht einem rebalancierten Index)
    und gibt dessen Max-Drawdown zurück.

    Symbole ohne jede Kurshistorie im Zeitraum (z.B. IPO nach 2008) werden
    aus der Gewichtung ausgeschlossen (Restgewichte NICHT redistribuiert -
    das würde ein anderes Portfolio simulieren als das tatsächliche) und
    explizit dokumentiert, statt Vollständigkeit vorzutäuschen.
    """
    usable: dict[str, tuple[float, pd.Series]] = {}
    excluded: list[str] = []
    for pw in position_weights:
        series = price_histories.get(pw.symbol)
        window = series[(series.index >= period.start) & (series.index <= period.end)] if series is not None else None
        if window is None or window.empty:
            excluded.append(pw.symbol)
            continue
        usable[pw.symbol] = (pw.weight, window)

    if not usable:
        return StressPeriodResult(period.name, period.start, period.end, None, [], excluded)

    total_abs_weight = sum(abs(w) for w, _ in usable.values())
    all_dates = sorted(set().union(*(set(series.index) for _, series in usable.values())))

    index_values = []
    for d in all_dates:
        weighted_return = 0.0
        for weight, series in usable.values():
            eligible = series[series.index <= d]
            if eligible.empty:
                continue
            pct_change = eligible.iloc[-1] / series.iloc[0] - 1
            weighted_return += (weight / total_abs_weight) * pct_change
        index_values.append(100.0 * (1 + weighted_return))

    index = pd.Series(index_values, index=all_dates)
    running_max = index.cummax()
    max_drawdown = float(((index - running_max) / running_max).min())

    return StressPeriodResult(period.name, period.start, period.end, max_drawdown, sorted(usable), excluded)


def run() -> None:
    app_config = AppConfig.load()
    with db.get_connection(app_config.db_path) as conn:
        portfolio_row = db.get_portfolio(conn, app_config.portfolio_name)
        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])

    if not open_position_rows:
        log.info("Keine offenen Positionen - Stresstest übersprungen (nichts zu testen).")
        return

    symbols = sorted({r["symbol"] for r in open_position_rows})
    log.info("Lade aktuelle Kurse für %d Symbol(e)...", len(symbols))
    current_prices = data_fetch.fetch_latest_prices(symbols)

    position_weights = compute_position_weights(open_position_rows, current_prices)
    if not position_weights:
        log.info("Keine der offenen Positionen hat einen aktuellen Kurs - Stresstest übersprungen.")
        return

    earliest_start = min(p.start for p in STRESS_PERIODS)
    latest_end = max(p.end for p in STRESS_PERIODS)
    log.info(
        "Lade historische Kurse für %d Symbol(e), %s bis %s...",
        len(symbols), earliest_start.date(), latest_end.date(),
    )
    price_histories = data_fetch.fetch_price_histories_for_range(symbols, earliest_start, latest_end)

    period_results = [
        compute_stress_period_result(period, position_weights, price_histories) for period in STRESS_PERIODS
    ]

    log.info("Rufe Claude für die analytische Einordnung auf (Kontaminationsrisiko, siehe Report)...")
    user_prompt = stress_test_prompt.build_stress_test_user_prompt(portfolio_row, position_weights, period_results)
    raw_response = get_trading_decision(
        stress_test_prompt.STRESS_TEST_SYSTEM_PROMPT,
        user_prompt,
        api_key=app_config.anthropic_api_key,
        model=app_config.claude_model,
    )
    commentary = stress_test_schema.parse_stress_test_commentary_from_json(raw_response)

    report_path = reporting.generate_stress_test_report(
        portfolio_row, position_weights, period_results, commentary, app_config.reports_dir
    )
    log.info("Stresstest-Report geschrieben: %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run()
    except Exception:
        log.exception("Historischer Stresstest fehlgeschlagen.")
        sys.exit(1)
