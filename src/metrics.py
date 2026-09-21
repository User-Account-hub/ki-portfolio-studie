"""Performance metrics.

The equity curve for the report/chart is reconstructed here by replaying
`trades` chronologically (cash + position bookkeeping mirrors
execution.py's cash-flow convention) and marking open positions to market
using historical closes from yfinance at each checkpoint. This is an
approximation (intra-period price moves on already-closed positions are not
captured), acceptable for this paper-trading case study's cadence.

Checkpoint frequency (and therefore the annualization factor used in
compute_metrics) is driven by `reconstruct_nav_history`'s `freqs` parameter,
NOT hardcoded - it must match however often the pipeline actually runs (see
.github/workflows/weekly_pipeline.yml's cron schedule). Two pandas weekly
offset aliases, one per run day (default: Monday + Thursday), give 2
checkpoints/week for Phase 1; adding/removing a run day means updating
`freqs` here to match, nothing else in this module.

Regimewechsel Phase 1 -> Phase 2 (Thesis Kap. 6.3/11.2/15): ab `PHASE2_START`
(2026-09-10) laeuft die Studie mit taeglichem Handel (Mo-Fr) statt 2x/Woche.
`reconstruct_nav_history` erzeugt deshalb fuer Zeitpunkte vor `PHASE2_START`
weiterhin Mo/Do-Checkpoints (`freqs`) und ab `PHASE2_START` werktaegliche
Checkpoints. `compute_metrics` haelt Vol/Sharpe strikt getrennt: eine Rendite
zaehlt nur dann als Phase-2-Rendite (Annualisierung mit
`PHASE2_PERIODS_PER_YEAR` = 252), wenn auch ihr INTERVALL-START bereits auf
oder nach `PHASE2_START` liegt - sonst wuerde eine ~3.5-Tage-Phase-1-Rendite
mit einer 1-Tages-Phase-2-Rendite in derselben Standardabweichung vermischt
und mit dem falschen Faktor annualisiert. Solange nicht genug reine
Phase-2-Renditen vorliegen (< 2), faellt die Berechnung auf die alte
Phase-1-Annualisierung (`NavHistory.periods_per_year`) ueber die komplette
Historie zurueck. Vol/Sharpe sind die einzigen Kennzahlen, die von der
Checkpoint-Dichte/-Einteilung abhaengen. total_return, max_drawdown und
benchmark_total_return/alpha_pct sind bewusst auf feste, laufunabhaengige
Referenzwerte geankert statt auf `nav[0]`/`dates[0]` (den ersten Checkpoint
DIESES Laufs) - siehe die jeweiligen Bugfix-Kommentare 2026-09-12 bei
`NavHistory.initial_nav`, `NavHistory.historical_peak_nav` und in
`compute_metrics` fuer die Details, warum `nav[0]`/`dates[0]` dafuer die
falsche Referenz waren.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from src.db import get_nav_at_or_after, get_peak_nav
from src.momentum_baseline import reconstruct_momentum_baseline
from src.risk_guardrails import OpenPosition, compute_nav
from src.segment_basket import SEGMENT_BASKET_SYMBOLS, reconstruct_segment_basket

CASH_SIGN = {"buy": -1, "sell": 1, "short": 1, "cover": -1}
OPEN_SIDES = {"buy": "long", "short": "short"}
CLOSE_SIDES = {"sell": "long", "cover": "short"}

# Regimewechsel Phase 1 (2x/Woche) -> Phase 2 (taeglich, Mo-Fr), siehe Thesis
# Kap. 6.3/11.2/15. Wie `freqs` unten kein Config-Wert, sondern bewusst hier
# hart kodiert, direkt neben dem Code, der ihn auswertet.
PHASE2_START = pd.Timestamp("2026-09-10")
PHASE2_PERIODS_PER_YEAR = 252.0

# Kap. 6.3 Portfolio-Reset (2026-09-21, siehe RESET_2026-09-21.md / Commit
# 673064c): Pilotphase (05.09.-20.09.) endet hier, die "offizielle" Studie
# beginnt neu bei NAV=1'000'000 (0 offene Positionen). reconstruct_nav_history
# rekonstruiert die NAV-Zeitreihe ab hier ausschliesslich aus Trades ab
# diesem Zeitpunkt (siehe pipeline.py-Aufrufstelle, db.get_trades_since) und
# ankert initial_cash/initial_nav auf den tatsaechlichen Reset-NAV
# (db.get_nav_at_or_after), nicht mehr auf das statische, nur einmal bei
# Portfolio-Erstellung gesetzte `initial_cash_balance`. Pilotphase-Trades/
# -Decisions bleiben vollstaendig in der DB erhalten (Kap. 6.3), fliessen
# aber nicht mehr in die offizielle NAV-/Kennzahlen-Rekonstruktion ein.
# Analog zu PHASE2_START: bewusst hart kodiert, ein einmaliges historisches
# Ereignis, kein Config-Wert.
OFFICIAL_STUDY_START = pd.Timestamp("2026-09-21 13:32:06")

# 2026-09-17: realistische Risk-free-Rate-Annahme fuer die Sharpe-Ratio
# (vorher fix 0.0 - unterstellte damit, dass "risikofrei" 0% Rendite
# abwirft, was den Sharpe systematisch nach oben verzerrt hatte). Anker: US
# 3-Monats-T-Bill, ~3.94-4.06% je nach Quelle am 2026-09-15/16 (13-Wochen-
# T-Bill-Rendite bzw. Sekundaermarktrendite), auf 4.0% gerundet. Punkt-
# schaetzung, kein Live-Feed - sollte periodisch von Hand aktualisiert
# werden, da sich der Zins ueber die Studienlaufzeit bewegt. Der praktikable
# Weg dafuer ist AppConfig.risk_free_rate_annual (Env-Var
# RISK_FREE_RATE_ANNUAL, siehe config.py/README) - die Pipeline uebergibt
# diesen Wert explizit an compute_metrics; die Konstante hier ist nur der
# Fallback fuer direkte/Test-Aufrufe ohne expliziten Wert.
DEFAULT_RISK_FREE_RATE_ANNUAL = 0.04

# Kap. 6.9 Erweiterung (2026-09-21): zweiter, sektorspezifischer Vergleichs-
# index ZUSAETZLICH zum konfigurierbaren Haupt-Benchmark (`benchmark_symbol`,
# aktuell SPY, siehe reconstruct_nav_history) - bewusst fest kodiert statt
# ueber Config/Env aenderbar (analog zu den hardcodierten FOMC-/CPI-Terminen
# in event_calendar.py oder der festen News-Feed-Liste in news_feed.py): es
# geht hier nicht um einen frei waehlbaren Vergleichsmassstab, sondern
# spezifisch um den in Kap. 6.9 der Thesis geforderten sektorbreiten
# Tech-Index. Das Anlage-Universum (Kap. 6.7) ist stark AI-/Halbleiter-lastig
# - ein reiner S&P-500-Vergleich (SPY) allein unterrepraesentiert das; QQQ
# ergaenzt SPY, ersetzt es nicht.
SECONDARY_BENCHMARK_SYMBOL = "QQQ"


@dataclass(frozen=True)
class NavHistory:
    dates: list[pd.Timestamp]
    nav: list[float]
    benchmark_normalized: list[float]
    # Zweiter, fest kodierter Vergleichsindex (SECONDARY_BENCHMARK_SYMBOL =
    # "QQQ", Kap. 6.9 Erweiterung 2026-09-21) - ZUSAETZLICH zu
    # benchmark_normalized, ersetzt es nicht. Dieselbe Normalisierungs-/
    # Ankerlogik wie benchmark_normalized (siehe _normalize_symbol).
    qqq_normalized: list[float]
    # Regelbasierte Momentum-Baseline (Thesis Kap. 6.9, nicht-KI-Vergleichsarm):
    # Top-Quintil 12-Wochen-Performance aus dem Aktien-Universum, gleichgewichtet,
    # monatlich rebalanciert - aus Kursdaten rekonstruiert wie benchmark_normalized,
    # kein separat gehandeltes Portfolio. Siehe src/momentum_baseline.py.
    baseline_normalized: list[float]
    # Thematischer Segment-ETF-Korb (Thesis Kap. 6.9 Erweiterung, 2026-09-21,
    # SEGMENT_BASKET_SYMBOLS = SMH/URA/ICLN, siehe src/segment_basket.py) -
    # rein informativer Vergleichspunkt wie baseline_normalized, analog dazu
    # berechnet (gleichgewichtet, monatlich rebalanciert), nur OHNE
    # Auswahl/Rangliste (immer alle drei fest kodierten Symbole).
    segment_basket_normalized: list[float]
    # Checkpoints/Jahr, abgeleitet aus den `freqs` von reconstruct_nav_history -
    # treibt die Annualisierung in compute_metrics (Volatilitaet, Sharpe).
    periods_per_year: float
    # Bugfix 2026-09-12: das echte Startkapital der "offiziellen Studie" -
    # seit dem Kap.-6.3-Reset (2026-09-21) der dort persistierte
    # nav_history-Wert (db.get_nav_at_or_after, Fallback
    # portfolio_row["initial_cash_balance"]), NICHT dasselbe wie `nav[0]`.
    # `nav[0]` ist nur der erste *Checkpoint* der in diesem Lauf rekonstruierten
    # Zeitreihe - je nach Checkpoint-Dichte (siehe Phase 1/2-Regimewechsel)
    # kann das ein beliebiger spaeterer Zeitpunkt nach dem ersten Trade sein,
    # nicht der Studienbeginn. compute_metrics' total_return muss deshalb
    # gegen `initial_nav` rechnen, nicht gegen `nav[0]` - sonst misst
    # "Gesamtrendite" je nach Lauf mal die Rendite seit Studienbeginn, mal nur
    # die Rendite seit dem letzten Checkpoint (was `last_period_return_pct`
    # ohnehin schon abdeckt).
    initial_nav: float
    # Bugfix 2026-09-12 (Teil 2, analog zu initial_nav): der WIRKLICH bekannte
    # historische NAV-Hoechststand (siehe db.get_peak_nav/
    # risk_guardrails.check_circuit_breaker, Kap. 6.8), NICHT nur das Maximum
    # ueber die lokalen Checkpoints dieses Laufs. compute_metrics' max_drawdown
    # muss `running_max` hierauf floor-en - sonst bleibt ein Drawdown, der VOR
    # dem ersten lokalen Checkpoint bereits stattfand (z.B. zwischen
    # Studienbeginn und dem ersten Checkpoint), unsichtbar.
    historical_peak_nav: float


@dataclass(frozen=True)
class MetricsResult:
    current_nav: float
    total_return_pct: float
    last_period_return_pct: float | None
    annualized_volatility_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float
    benchmark_total_return_pct: float
    alpha_pct: float
    baseline_total_return_pct: float
    baseline_alpha_pct: float
    # Information Ratio = alpha_pct / tracking_error_pct (2026-09-20, siehe
    # compute_metrics fuer die Herleitung von tracking_error_pct und die
    # Annualisierungs-Logik). None, wenn kein Tracking Error berechenbar ist
    # (zu wenig Perioden oder Portfolio bewegt sich exakt wie die Benchmark -
    # std() == 0 -> Division durch Null wird vermieden statt +-inf zu liefern).
    information_ratio: float | None
    # Zweiter, sektorspezifischer Vergleichsindex (Kap. 6.9 Erweiterung,
    # 2026-09-21, SECONDARY_BENCHMARK_SYMBOL = "QQQ") - ZUSAETZLICH zu
    # benchmark_total_return_pct/alpha_pct oben, ersetzt sie nicht. Dieselbe
    # initial_nav-Ankerung wie beim Haupt-Benchmark (siehe compute_metrics).
    qqq_total_return_pct: float
    alpha_vs_qqq_pct: float
    # Thematischer Segment-ETF-Korb (Kap. 6.9 Erweiterung, 2026-09-21,
    # SEGMENT_BASKET_SYMBOLS = SMH/URA/ICLN, siehe src/segment_basket.py) -
    # architektonisch identisch zu baseline_total_return_pct/baseline_alpha_pct
    # (dieselbe initial_nav-Ankerung), nur gegen den Segment-Korb statt der
    # Momentum-Baseline.
    segment_basket_total_return_pct: float
    alpha_vs_segment_basket_pct: float


def _replay_ledger(trades: list[sqlite3.Row], initial_cash: float) -> list[dict]:
    cash = initial_cash
    positions: dict[tuple[str, str], dict] = {}
    snapshots = []

    for t in trades:
        side, qty, price, symbol = t["side"], t["quantity"], t["price"], t["symbol"]
        # 2026-09-17: die feste Spread/Slippage-Pauschale (execution.py,
        # risk_config.yaml) wird beim echten Fill vom cash_balance abgezogen -
        # muss hier mitreplayed werden, sonst driftet diese rekonstruierte
        # Zeitreihe von der tatsaechlichen (live gefuehrten) cash_balance weg.
        # `"transaction_cost" in t.keys()`-Fallback deckt Trades von vor
        # dieser Spalte ab (DEFAULT 0 in der DB, aber defensiv auch hier).
        transaction_cost = t["transaction_cost"] if "transaction_cost" in t.keys() else 0.0
        cash += CASH_SIGN[side] * qty * price - transaction_cost

        if side in OPEN_SIDES:
            key = (symbol, OPEN_SIDES[side])
            pos = positions.get(key)
            if pos is None:
                positions[key] = {
                    "instrument_type": t["instrument_type"],
                    "side": key[1],
                    "quantity": qty,
                    "avg_entry_price": price,
                }
            else:
                new_qty = pos["quantity"] + qty
                pos["avg_entry_price"] = (pos["quantity"] * pos["avg_entry_price"] + qty * price) / new_qty
                pos["quantity"] = new_qty
        else:
            key = (symbol, CLOSE_SIDES[side])
            pos = positions.get(key)
            if pos is not None:
                pos["quantity"] -= qty
                if pos["quantity"] <= 1e-9:
                    del positions[key]

        snapshots.append(
            {
                "timestamp": pd.Timestamp(t["executed_at"]),
                "cash": cash,
                "positions": [OpenPosition(symbol=k[0], **v) for k, v in positions.items()],
            }
        )
    return snapshots


def _price_lookup_symbols(snapshots: list[dict], watchlist_underlyings: dict[str, str]) -> list[str]:
    symbols = set()
    for snap in snapshots:
        for p in snap["positions"]:
            symbols.add(watchlist_underlyings.get(p.symbol, p.symbol))
    return sorted(symbols)


MOMENTUM_BASELINE_LOOKBACK_WEEKS = 12


def _normalize_symbol_to_initial_cash(
    price_on, symbol: str, start: pd.Timestamp, dates: list[pd.Timestamp], initial_cash: float
) -> list[float]:
    """Normalisiert `symbol`s Kursverlauf auf `initial_cash` am echten
    Studienbeginn (`start`) - dieselbe Anker-/Fallback-Logik, die vorher nur
    für den Haupt-Benchmark inline in reconstruct_nav_history stand (siehe
    dortigen Bugfix-Kommentar 2026-09-12 Teil 3: IMMER gegen `initial_nav`
    ankern, nie gegen den ersten lokalen Checkpoint) - seit der QQQ-
    Erweiterung (Kap. 6.9, 2026-09-21) für zwei Symbole gebraucht, daher
    hier als gemeinsame Funktion extrahiert statt ein zweites Mal zu
    duplizieren. Fehlt der Kurs (Symbol nicht auflösbar/kein Datenpunkt),
    fällt der jeweilige Tag auf eine flache Linie bei `initial_cash` zurück,
    statt NaN zu produzieren."""
    start_price = price_on(symbol, start)
    normalized = []
    for d in dates:
        price = price_on(symbol, d)
        if price is None or start_price is None:
            normalized.append(initial_cash)
        else:
            normalized.append(initial_cash * (price / start_price))
    return normalized


def reconstruct_nav_history(
    conn: sqlite3.Connection,
    portfolio_row: sqlite3.Row,
    trades: list[sqlite3.Row],
    watchlist_underlyings: dict[str, str],
    benchmark_symbol: str,
    momentum_universe_symbols: list[str] | None = None,
    freqs: tuple[str, ...] = ("W-MON", "W-THU"),
    phase2_start: pd.Timestamp = PHASE2_START,
    official_start: pd.Timestamp = OFFICIAL_STUDY_START,
) -> NavHistory:
    """`freqs` are pandas weekly offset aliases, one per weekday the pipeline
    ran during Phase 1 (Monday + Thursday, matching the pre-2026-09-10 cron
    schedule - see module docstring). periods_per_year is derived as
    52 * len(freqs), since each alias contributes one checkpoint per week -
    this remains the Phase-1-only annualization factor stored on NavHistory;
    compute_metrics switches to PHASE2_PERIODS_PER_YEAR once enough
    Phase-2-only returns exist.

    Checkpoints before `phase2_start` follow `freqs` (Phase 1, 2x/week);
    checkpoints from `phase2_start` onward are every business day (Phase 2,
    daily), reflecting the Kap.-6.3 regime change to daily trading.

    `momentum_universe_symbols` (Thesis Kap. 6.9) is the equity universe the
    regelbasierte Momentum-Baseline ranks/rebalances over - independent of
    the AI portfolio's actual trades, computed purely from price data (see
    src/momentum_baseline.py).

    `official_start` (Kap. 6.3 Reset, siehe OFFICIAL_STUDY_START-Kommentar):
    `trades` muss bereits vom Aufrufer auf `executed_at >= official_start`
    gefiltert sein (db.get_trades_since) - diese Funktion filtert nicht
    selbst nach, sondern verankert lediglich `initial_cash`/`start` darauf.
    Der echte, geankerte Startwert kommt aus db.get_nav_at_or_after (der beim
    Reset persistierte nav_history-Eintrag); nur falls dort noch nichts
    vorliegt (z.B. in Tests ohne nav_history-Daten), faellt das auf das
    statische `initial_cash_balance` zurueck."""
    since = official_start.strftime("%Y-%m-%d %H:%M:%S")
    initial_cash = (
        get_nav_at_or_after(conn, portfolio_row["id"], since) or portfolio_row["initial_cash_balance"]
    )
    periods_per_year = 52 * len(freqs)
    momentum_universe_symbols = momentum_universe_symbols or []
    # Der Circuit-Breaker (risk_guardrails.py, Kap. 6.8) pflegt bereits den
    # wahren historischen NAV-Hoechststand ueber alle Laeufe hinweg (bewusst
    # INKLUSIVE Pilotphase, siehe RESET_2026-09-21.md-Hinweis zum
    # Circuit-Breaker) - den wiederverwenden wir fuer max_drawdown, statt ihn
    # separat zu bestimmen.
    historical_peak_nav = max(initial_cash, get_peak_nav(conn, portfolio_row["id"]) or initial_cash)

    if not trades:
        return NavHistory(
            dates=[pd.Timestamp.today()],
            nav=[initial_cash],
            benchmark_normalized=[initial_cash],
            qqq_normalized=[initial_cash],
            baseline_normalized=[initial_cash],
            segment_basket_normalized=[initial_cash],
            periods_per_year=periods_per_year,
            initial_nav=initial_cash,
            historical_peak_nav=historical_peak_nav,
        )

    snapshots = _replay_ledger(trades, initial_cash)
    # Bugfix 2026-09-21 (Kap.-6.3-Reset): vorher war `start` der Zeitpunkt
    # des ERSTEN TRADES in `trades` - lag der Reset zeitlich vor dem ersten
    # danach ausgefuehrten Trade (z.B. weil Orders zunaechst wegen
    # Risk-Guardrails abgelehnt wurden), verschob das faelschlich den
    # Studienbeginn fuer Checkpoints/Benchmark-Ankerung auf diesen spaeteren
    # Zeitpunkt statt auf den echten Reset-Zeitpunkt. `official_start` ist
    # der tatsaechliche, feststehende Studienbeginn (siehe
    # OFFICIAL_STUDY_START); `trades` ist laut Docstring bereits darauf
    # gefiltert, jeder Snapshot liegt also ohnehin auf/nach `official_start`.
    start = official_start.normalize()
    end = pd.Timestamp.today().normalize()

    phase1_end = min(end, phase2_start)
    phase1_checkpoints = {ts for freq in freqs for ts in pd.date_range(start=start, end=phase1_end, freq=freq)}
    if end > phase2_start:
        phase2_checkpoints = set(
            pd.date_range(start=max(start, phase2_start + pd.Timedelta(days=1)), end=end, freq="B")
        )
    else:
        phase2_checkpoints = set()
    checkpoints = pd.DatetimeIndex(sorted(phase1_checkpoints | phase2_checkpoints))
    if len(checkpoints) == 0 or checkpoints[-1] < end:
        checkpoints = checkpoints.append(pd.DatetimeIndex([end]))

    price_symbols = sorted(
        set(_price_lookup_symbols(snapshots, watchlist_underlyings))
        | {benchmark_symbol, SECONDARY_BENCHMARK_SYMBOL}
        | set(SEGMENT_BASKET_SYMBOLS)
        | set(momentum_universe_symbols)
    )
    # Der Momentum-Baseline-Rebalance am allerersten Termin (`start`) braucht
    # bereits `MOMENTUM_BASELINE_LOOKBACK_WEEKS` Kurshistorie VOR `start`, um
    # die erste Rangliste ohne Lookahead zu bilden - der Download-Beginn muss
    # entsprechend weiter zurueckreichen als die bisherigen 7 Tage Puffer.
    download_start = start - pd.Timedelta(weeks=MOMENTUM_BASELINE_LOOKBACK_WEEKS, days=7)
    history = yf.download(
        tickers=price_symbols,
        start=download_start,
        end=end + pd.Timedelta(days=1),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    def price_on(symbol: str, when: pd.Timestamp) -> float | None:
        try:
            series = history[symbol]["Close"] if len(price_symbols) > 1 else history["Close"]
            series = series.dropna()
            eligible = series[series.index <= when]
            return float(eligible.iloc[-1]) if not eligible.empty else None
        except (KeyError, IndexError):
            return None

    nav_values, dates = [], []
    for checkpoint in checkpoints:
        applicable = [s for s in snapshots if s["timestamp"] <= checkpoint]
        snap = applicable[-1] if applicable else {"cash": initial_cash, "positions": []}
        current_prices = {}
        for p in snap["positions"]:
            lookup_symbol = watchlist_underlyings.get(p.symbol, p.symbol)
            price = price_on(lookup_symbol, checkpoint)
            if price is not None:
                current_prices[p.symbol] = price
        nav = compute_nav(snap["cash"], snap["positions"], current_prices)
        nav_values.append(nav)
        dates.append(checkpoint)

    # Bugfix 2026-09-12 (Teil 3): Anker war `dates[0]` - der erste Checkpoint
    # DIESES Laufs, nicht der echte Studienbeginn. Zusammen mit compute_metrics'
    # `benchmark_total_return` (das ebenfalls gegen `initial_nav` statt gegen
    # `benchmark[0]` rechnet) sorgt der falsche Anker sonst dafuer, dass sich
    # das Vergleichsfenster fuer die Benchmark von Lauf zu Lauf verschiebt -
    # und seit dem total_return-Fix (Teil 1) waeren total_return und
    # benchmark_total_return sonst inkonsistent geankert, was alpha_pct verzerrt.
    benchmark_normalized = _normalize_symbol_to_initial_cash(price_on, benchmark_symbol, start, dates, initial_cash)
    # Kap. 6.9 Erweiterung (2026-09-21): zweiter, sektorspezifischer
    # Vergleichsindex (SECONDARY_BENCHMARK_SYMBOL = "QQQ") - dieselbe
    # Anker-/Fallback-Logik wie oben, ZUSAETZLICH zu benchmark_normalized.
    qqq_normalized = _normalize_symbol_to_initial_cash(
        price_on, SECONDARY_BENCHMARK_SYMBOL, start, dates, initial_cash
    )

    # Kap. 6.9: regelbasierte Momentum-Baseline, rein aus Kursdaten
    # rekonstruiert (kein separat gehandeltes Portfolio) - no-op (flache Linie
    # bei initial_cash) falls kein Universum uebergeben wurde.
    baseline_normalized = reconstruct_momentum_baseline(
        price_on=price_on,
        universe_symbols=momentum_universe_symbols,
        dates=dates,
        start=start,
        initial_cash=initial_cash,
        lookback_weeks=MOMENTUM_BASELINE_LOOKBACK_WEEKS,
    ) if momentum_universe_symbols else [initial_cash] * len(dates)

    # Kap. 6.9 Erweiterung (2026-09-21): thematischer Segment-ETF-Korb
    # (SMH/URA/ICLN, siehe src/segment_basket.py) - analog zur Momentum-
    # Baseline oben berechnet, aber IMMER aktiv (feste Symbole, kein
    # optionales Universum wie bei der Momentum-Baseline).
    segment_basket_normalized = reconstruct_segment_basket(
        price_on=price_on,
        dates=dates,
        start=start,
        initial_cash=initial_cash,
    )

    return NavHistory(
        dates=dates,
        nav=nav_values,
        benchmark_normalized=benchmark_normalized,
        qqq_normalized=qqq_normalized,
        baseline_normalized=baseline_normalized,
        segment_basket_normalized=segment_basket_normalized,
        periods_per_year=periods_per_year,
        initial_nav=initial_cash,
        historical_peak_nav=historical_peak_nav,
    )


def compute_metrics(
    nav_history: NavHistory,
    risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
    phase2_start: pd.Timestamp = PHASE2_START,
) -> MetricsResult:
    nav = pd.Series(nav_history.nav, index=nav_history.dates)
    benchmark = pd.Series(nav_history.benchmark_normalized, index=nav_history.dates)
    returns = nav.pct_change().dropna()

    # Information Ratio (2026-09-20): aktive PERIODEN-Rendite (Portfolio
    # minus Benchmark je Checkpoint-Intervall, NICHT die kumulierte
    # Gesamtrendite) - Grundlage fuer den Tracking Error weiter unten. `nav`
    # und `benchmark` teilen exakt denselben `nav_history.dates`-Index, daher
    # direkt per Index alignierbar; `benchmark_normalized` enthaelt nie NaN
    # (reconstruct_nav_history faellt bei fehlendem Kurs auf initial_cash
    # zurueck), sodass hier keine gesonderte Fehlwert-Behandlung noetig ist.
    benchmark_returns = benchmark.pct_change()
    active_returns = returns - benchmark_returns.loc[returns.index]

    # Bugfix 2026-09-12: vorher `nav.iloc[-1] / nav.iloc[0] - 1` - das
    # verwechselte "erster Checkpoint dieses Laufs" mit "Studienbeginn".
    # Fiel bisher nicht auf, weil `nav.iloc[0]` in fast allen bisherigen
    # Laeufen zufaellig gleich `initial_nav` war (nur 1 Checkpoint pro Lauf).
    # Seit dem Phase-1/2-Regimewechsel (mehr Checkpoints/Lauf) driftet
    # `nav.iloc[0]` vom echten Startkapital weg, wodurch "Gesamtrendite" nur
    # noch die Rendite seit dem vorletzten Checkpoint zeigte - identisch zu
    # `last_period_return_pct` statt zur kumulierten Rendite seit Studienstart.
    total_return = nav.iloc[-1] / nav_history.initial_nav - 1
    last_period_return = float(returns.iloc[-1]) if not returns.empty else None

    # Eine Rendite zaehlt nur als Phase-2-Rendite, wenn auch ihr Intervall-
    # Start >= phase2_start liegt - sonst waere es ein Phase-1->Phase-2-
    # Uebergangsintervall mit fremder Frequenz. `interval_start_dates` sind
    # positional (nicht per Label) an `returns` ausgerichtet, da beide gleich
    # lang und gleich sortiert aus derselben `nav`-Serie stammen.
    interval_start_dates = nav.index[:-1]
    is_phase2_return = interval_start_dates >= phase2_start
    phase2_returns = returns[is_phase2_return]

    if len(phase2_returns) >= 2 and phase2_returns.std() > 0:
        periods_per_year = PHASE2_PERIODS_PER_YEAR
        vol_returns = phase2_returns
        # Denselben Phase-1/2-Split wie fuer Vol/Sharpe anwenden (siehe
        # Modul-Docstring) - Tracking Error ist annualisierungslogisch
        # dieselbe Art Kennzahl (annualisierte Standardabweichung einer
        # Renditereihe), muss also konsistent mit derselben Checkpoint-
        # Einteilung berechnet werden, nicht mit einer zweiten, unabhaengigen.
        tracking_error_returns = active_returns[is_phase2_return]
    else:
        periods_per_year = nav_history.periods_per_year
        vol_returns = returns
        tracking_error_returns = active_returns

    if len(vol_returns) >= 2 and vol_returns.std() > 0:
        rf_per_period = risk_free_rate_annual / periods_per_year
        annualized_vol = float(vol_returns.std() * np.sqrt(periods_per_year))
        sharpe = float((vol_returns.mean() - rf_per_period) / vol_returns.std() * np.sqrt(periods_per_year))
    else:
        annualized_vol = None
        sharpe = None

    if len(tracking_error_returns) >= 2 and tracking_error_returns.std() > 0:
        tracking_error = float(tracking_error_returns.std() * np.sqrt(periods_per_year))
    else:
        tracking_error = None

    # Bugfix 2026-09-12 (Teil 2): `nav.cummax()` allein kennt nur die lokalen
    # Checkpoints DIESES Laufs - ein Peak, der davor lag (z.B. das
    # Startkapital selbst oder ein Zwischenhoch kurz nach den ersten Trades),
    # faellt sonst unter den Tisch und ein bereits eingetretener Drawdown
    # zeigt sich faelschlich als 0%. `historical_peak_nav` (siehe NavHistory)
    # ist ein Floor: cummax bleibt massgeblich, sobald die lokale Serie den
    # bekannten historischen Peak selbst uebertrifft.
    running_max = nav.cummax().clip(lower=nav_history.historical_peak_nav)
    drawdown = (nav - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Bugfix 2026-09-12 (Teil 3): vorher `benchmark.iloc[-1] / benchmark.iloc[0] - 1`
    # - `benchmark.iloc[0]` ist per Konstruktion IMMER `initial_nav` am jeweils
    # gewaehlten Anker-Datum (siehe reconstruct_nav_history), unabhaengig davon
    # welches Datum das ist. Dadurch kuerzte sich der (falsche) Anker `dates[0]`
    # aus dieser Ratio komplett heraus, und selbst nachdem der Anker in
    # reconstruct_nav_history auf den echten Studienbeginn (`start`) korrigiert
    # wurde, haette diese Formel weiterhin nur die Rendite seit `dates[0]`
    # gemessen. Divisor muss `initial_nav` (fixer Dollarbetrag) sein, nicht
    # `benchmark.iloc[0]` - erst das macht total_return und
    # benchmark_total_return konsistent vergleichbar (und damit alpha_pct
    # aussagekraeftig).
    benchmark_total_return = float(benchmark.iloc[-1] / nav_history.initial_nav - 1)

    # Kap. 6.9 Erweiterung (2026-09-21): qqq_total_return_pct/alpha_vs_qqq_pct
    # sind architektonisch identisch zu benchmark_total_return_pct/alpha_pct
    # oben (dieselbe initial_nav-Ankerung) - zweiter, sektorspezifischer
    # Vergleichsindex ZUSAETZLICH zum Haupt-Benchmark, nicht als Ersatz.
    qqq = pd.Series(nav_history.qqq_normalized, index=nav_history.dates)
    qqq_total_return = float(qqq.iloc[-1] / nav_history.initial_nav - 1)

    # Kap. 6.9: baseline_total_return/baseline_alpha_pct sind architektonisch
    # identisch zu benchmark_total_return/alpha_pct - selbe initial_nav-
    # Ankerung (Details siehe Bugfix-Kommentar oben), nur gegen die
    # Momentum-Baseline-Zeitreihe statt der Index-Benchmark.
    baseline = pd.Series(nav_history.baseline_normalized, index=nav_history.dates)
    baseline_total_return = float(baseline.iloc[-1] / nav_history.initial_nav - 1)

    # Kap. 6.9 Erweiterung (2026-09-21): segment_basket_total_return_pct/
    # alpha_vs_segment_basket_pct sind architektonisch identisch zu
    # baseline_total_return_pct/baseline_alpha_pct oben (dieselbe
    # initial_nav-Ankerung), nur gegen den thematischen Segment-ETF-Korb
    # (SMH/URA/ICLN) statt der Momentum-Baseline.
    segment_basket = pd.Series(nav_history.segment_basket_normalized, index=nav_history.dates)
    segment_basket_total_return = float(segment_basket.iloc[-1] / nav_history.initial_nav - 1)

    alpha_pct = float(total_return - benchmark_total_return)
    # Information Ratio = Alpha / Tracking Error. Nutzt bewusst denselben
    # alpha_pct wie die Report-Zeile "Alpha vs. Benchmark" (kumulierte
    # Gesamtrendite-Differenz seit initial_nav), NICHT eine separat
    # annualisierte aktive Rendite - direkt neben genau dieser Zeile
    # ausgegeben (siehe reporting.py), daher dieselbe Bezugsgroesse. None,
    # wenn kein Tracking Error berechenbar ist (siehe oben).
    information_ratio = float(alpha_pct / tracking_error) if tracking_error else None

    return MetricsResult(
        current_nav=float(nav.iloc[-1]),
        total_return_pct=float(total_return),
        last_period_return_pct=last_period_return,
        annualized_volatility_pct=annualized_vol,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_drawdown,
        benchmark_total_return_pct=benchmark_total_return,
        alpha_pct=alpha_pct,
        baseline_total_return_pct=baseline_total_return,
        baseline_alpha_pct=float(total_return - baseline_total_return),
        information_ratio=information_ratio,
        qqq_total_return_pct=qqq_total_return,
        alpha_vs_qqq_pct=float(total_return - qqq_total_return),
        segment_basket_total_return_pct=segment_basket_total_return,
        alpha_vs_segment_basket_pct=float(total_return - segment_basket_total_return),
    )
