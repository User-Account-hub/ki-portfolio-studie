"""Main entrypoint for one pipeline run (triggered twice weekly by CI - Monday
and Thursday, see .github/workflows/weekly_pipeline.yml; src/metrics.py's
reconstruct_nav_history must be kept in sync with this cadence via its
`freqs` parameter).

Steps:
  0a. EINMALIGE Uebergangs-Sicherheitspruefung (siehe
      _pilot_phase_positions_still_open weiter unten) - Pilotphase -> offizielle
      Studie, nur bis zum erfolgreichen Portfolio-Reset relevant.
  0b. Idempotenz-Sperre (siehe _already_decided_today weiter unten, Kap. 12.7) -
      dauerhaft, verhindert eine doppelte Handelsentscheidung am selben Tag.
  1. Load config, connect to DB.
  2. Fetch market data for the watchlist + all open positions.
  3. Mandatory sweep: force-close any short position breaching its stop-loss,
     regardless of what follows (documented, compliance-relevant).
  4. Build the prompt, call Claude, parse & validate the JSON response.
  5. Run every proposed order through the risk guardrails and execute the
     approved ones (real Alpaca paper orders for equity/ETF, simulated
     bookkeeping for structured products).
  6. Reconstruct the NAV history, compute metrics, write a Markdown report.

Usage:
    python -m src.pipeline
"""
from __future__ import annotations

import datetime
import logging
import os
import sys

import pandas as pd

from src import (
    boundary_conditions,
    broker_alpaca,
    correlation,
    data_fetch,
    data_quality,
    db,
    deep_reflection_prompt,
    deep_reflection_schema,
    event_calendar,
    execution,
    fundamentals,
    market_phase,
    metrics,
    news_feed,
    position_sizing,
    reporting,
)
from src.claude_client import get_trading_decision
from src.config import AppConfig, RiskConfig, Watchlist
from src.order_schema import STRUCTURED_INSTRUMENT_TYPES, OrderParsingError, parse_orders_from_json
from src.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.risk_guardrails import compute_nav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def _run_data_quality_checks(
    app_config: AppConfig,
    tradable_symbols: list[str],
    structured_product_symbols: set[str],
    yfinance_prices: dict[str, float],
) -> data_quality.DataQualityReport:
    """Datenqualitäts-Check (2026-09-17): (1) Kursvergleich yfinance vs.
    Alpaca (Stichprobe), (2) Lücken-/Ausreisser-Erkennung in den yfinance-
    Kurshistorien - siehe src/data_quality.py für die eigentliche Logik.

    Bewusst so gebaut, dass WEDER ein gefundenes Datenqualitätsproblem NOCH
    ein Fehler dieses Checks selbst (z.B. Alpaca-Marktdaten-Endpunkt down)
    den Pipeline-Lauf abbricht - ein Datenqualitätsproblem ist ein
    Beobachtungssignal für Log/Report, kein Grund, einen ansonsten gültigen
    Lauf zu verwerfen.
    """
    sample_symbols = data_quality.select_price_comparison_sample(
        [s for s in tradable_symbols if s in yfinance_prices]
    )
    alpaca_prices = {}
    try:
        data_client = broker_alpaca.get_market_data_client(
            app_config.alpaca_api_key, app_config.alpaca_secret_key
        )
        alpaca_prices = broker_alpaca.get_latest_trade_prices(data_client, sample_symbols)
    except Exception:
        log.exception("Alpaca-Kursvergleich (Datenqualität) fehlgeschlagen - wird übersprungen.")

    price_histories = {}
    try:
        price_histories = data_fetch.fetch_price_histories(
            tradable_symbols, known_unresolvable_symbols=structured_product_symbols
        )
    except Exception:
        log.exception("Kurshistorien-Abruf (Datenqualität) fehlgeschlagen - wird übersprungen.")

    return data_quality.build_report(yfinance_prices, alpaca_prices, price_histories)


def _compute_volatility_scaling(
    snapshots: dict[str, data_fetch.MarketSnapshot],
    risk_config: RiskConfig,
) -> dict[str, position_sizing.VolatilityScaling]:
    """Volatilitätsadjustierte Positionsgrössen-Skalierung (siehe
    src/position_sizing.py) - nutzt die pro Symbol bereits in `snapshots`
    berechnete annualisierte 20-Tage-Volatilität, kein zusätzlicher
    Datenabruf nötig. Symbole ohne berechenbare Volatilität (zu kurze
    Historie) fehlen in `volatilities` und erhalten dadurch konsequent
    keinen Skalierungsfaktor (siehe compute_scaling_factors' Docstring)."""
    volatilities = {
        s: snap.volatility_20d_annualized
        for s, snap in snapshots.items()
        if snap.volatility_20d_annualized is not None
    }
    return position_sizing.compute_scaling_factors(
        volatilities,
        min_factor=risk_config.volatility_scaling_min_factor,
        max_factor=risk_config.volatility_scaling_max_factor,
    )


def _compute_market_phases(
    snapshots: dict[str, data_fetch.MarketSnapshot],
) -> dict[str, market_phase.MarketPhaseClassification]:
    """Regelbasierte Bull/Bear/Seitwärts-Klassifikation je Symbol (siehe
    src/market_phase.py) - wie _compute_volatility_scaling oben rein aus den
    bereits in `snapshots` vorhandenen Werten (sma20/sma50/Volatilität),
    kein zusätzlicher Datenabruf. Dient als unabhängiger, mechanischer
    Gegencheck zu Claudes optionaler cycle_position-Angabe je Order (Kap. 3,
    SYSTEM_PROMPT-Anforderung 1) - rein dokumentarisch, siehe execution.py/
    reporting.py, kein Guardrail-Veto."""
    return market_phase.classify_universe_market_phases(snapshots)


# ============================================================================
# EINMALIGE UEBERGANGS-SICHERHEITSPRUEFUNG (2026-09-21, Pilotphase -> offizielle
# Studie, Kap. 6.3) - siehe RESET_2026-09-21.md fuer den zugehoerigen,
# vollstaendig geplanten, aber noch nicht abgeschlossenen Portfolio-Reset
# (alle Pilotphase-Positionen bei Alpaca UND lokal schliessen, Cash auf
# initial_cash_balance zuruecksetzen). Ohne diesen Check wuerde ein Lauf, der
# startet BEVOR der Reset tatsaechlich durchgefuehrt wurde, auf dem
# verschmutzten Pilotphase-Zustand (offene Positionen, abweichender Cash-
# Stand) als vermeintlichem "Tag 1" der offiziellen Studie aufsetzen.
#
# Erkennung: jede noch offene Position mit `opened_at` VOR dem offiziellen
# Studienstart (deep_reflection_prompt.OFFICIAL_STUDY_START) ist zwingend ein
# Pilotphase-Ueberbleibsel - sie kann nur aus der Zeit vor dem Reset stammen.
# Sobald der Reset erfolgreich durchgefuehrt wurde (alle diese Positionen auf
# status='closed' gesetzt), liefert `_pilot_phase_positions_still_open` fuer
# IMMER eine leere Liste, unabhaengig davon, an welchem Tag der Reset
# nachgeholt wird - der Check wird dann automatisch und dauerhaft wirkungslos.
#
# GEDACHT NUR FUER DIESEN UEBERGANG: Sobald der Reset bestaetigt durchgefuehrt
# wurde, kann dieser gesamte Block (diese Funktion UND ihr Aufruf in run())
# wieder entfernt werden - das ist kein dauerhafter Guardrail, sondern eine
# einmalige Absicherung gegen genau dieses eine Uebergangsfenster.
# ============================================================================


def _pilot_phase_positions_still_open(
    open_position_rows,
    study_start: pd.Timestamp = deep_reflection_prompt.OFFICIAL_STUDY_START,
) -> list:
    """Liefert alle Zeilen aus `open_position_rows`, deren `opened_at` vor
    `study_start` liegt (siehe Block-Kommentar oben fuer die Begruendung).
    Leere Liste = kein Reset-Blocker (entweder gar keine offenen Positionen,
    oder ausschliesslich welche, die nach dem Studienstart eroeffnet wurden -
    also legitime Trades der offiziellen Studie, keine Pilotphase-Reste)."""
    return [r for r in open_position_rows if pd.Timestamp(r["opened_at"]) < study_start]


# ============================================================================
# IDEMPOTENZ-SPERRE (2026-09-21, Kap. 12.7 "Datenintegritaet") - verhindert
# eine doppelte Handelsentscheidung/-ausfuehrung am selben Kalendertag, falls
# die Pipeline versehentlich zweimal am selben Tag ausgeloest wird (z.B. ein
# manueller workflow_dispatch zusaetzlich zum planmaessigen Cron-Lauf). Anders
# als die weiter oben stehende Uebergangs-Sicherheitspruefung ist das KEIN
# einmaliger, wieder zu entfernender Mechanismus, sondern ein dauerhafter
# Guardrail - Kap. 12.7 verlangt idempotente Laeufe generell, nicht nur fuer
# den Studienstart-Uebergang.
# ============================================================================


def _is_force_rerun_requested(force: bool | None) -> bool:
    """`force` (Funktionsparameter, siehe run()) hat Vorrang vor der
    Umgebungsvariable FORCE_RERUN - erlaubt sowohl eine programmatische
    Ausnahme (Tests, ein zukuenftiges CLI-Flag) als auch eine rein
    Environment-basierte (z.B. ueber einen GitHub-Actions-workflow_dispatch-
    Input, siehe weekly_pipeline.yml). Akzeptiert die ueblichen "truthy"
    String-Schreibweisen ("1"/"true"/"yes", gross-/kleinschreibungsunabhaengig) -
    alles andere (inkl. fehlender oder leerer Wert) gilt als nicht gesetzt."""
    if force is not None:
        return force
    return os.getenv("FORCE_RERUN", "").strip().lower() in ("1", "true", "yes")


def _already_decided_today(conn, portfolio_id: int, claude_model: str):
    """Liefert den heutigen "echten" Handelsentscheidungs-Eintrag (siehe
    db.get_successful_decision_today's Docstring fuer die genaue Abgrenzung
    zu anderen Decision-Arten) oder None, falls heute noch keine
    abgeschlossene Handelsentscheidung stattgefunden hat."""
    return db.get_successful_decision_today(conn, portfolio_id, claude_model)


# Puffer über die eigentlich benötigten 60 Handelstage hinaus, damit
# Wochenenden/Feiertage genug Handelstage für ein volles 60-Tage-
# Renditefenster übrig lassen (60 Handelstage ~ 84-90 Kalendertage).
CORRELATION_FETCH_LOOKBACK_DAYS = 120


def _compute_correlation_matrix(
    universe_symbols: list[str],
    structured_product_symbols: set[str],
):
    """Rollierende 60-Tage-Korrelationsmatrix (Tagesrenditen) über das
    gesamte Anlage-Universum - siehe src/correlation.py für die eigentliche
    Berechnung und die Einordnung als Beobachtungsgrösse ohne Veto-Wirkung.

    Bewusst ein eigener, separater yfinance-Download (dasselbe akzeptierte
    Duplikations-Muster wie bei data_quality.py/metrics.py - siehe dortige
    Kommentare) statt eine der bestehenden Kurshistorien-Abfragen
    wiederzuverwenden. Kein Abbruch bei einem Fehler: liefert dann eine
    leere DataFrame zurück, die den Warn-/Cluster-Check zu einem No-Op
    macht (siehe correlation.py's Leerlauf-Verhalten).
    """
    try:
        price_histories = data_fetch.fetch_price_histories(
            universe_symbols,
            lookback_days=CORRELATION_FETCH_LOOKBACK_DAYS,
            known_unresolvable_symbols=structured_product_symbols,
        )
        return correlation.compute_correlation_matrix(price_histories)
    except Exception:
        log.exception("Korrelationsmatrix-Berechnung fehlgeschlagen - wird übersprungen.")
        return pd.DataFrame()


def _check_upcoming_events(
    universe_symbols: list[str],
) -> tuple[list[event_calendar.EarningsWarning], list[event_calendar.MacroEvent]]:
    """Event-Kalender-Hinweis: bevorstehende Quartalsberichte (pro Symbol)
    und hardcodierte FOMC-/CPI-Termine (portfolioweit) - siehe
    src/event_calendar.py. Rein informativ (Prompt-Kontext für Claude),
    kein Guardrail. Kein Abbruch bei einem Fehler: liefert dann leere
    Listen zurück (kein Hinweis diesen Lauf, statt den Lauf zu gefährden)."""
    today = datetime.date.today()
    try:
        earnings_warnings = event_calendar.check_upcoming_earnings(universe_symbols, today)
    except Exception:
        log.exception("Earnings-Kalender-Abfrage fehlgeschlagen - wird übersprungen.")
        earnings_warnings = []
    try:
        macro_events = event_calendar.get_upcoming_macro_events(today)
    except Exception:
        log.exception("Makro-Termin-Prüfung (FOMC/CPI) fehlgeschlagen - wird übersprungen.")
        macro_events = []
    return earnings_warnings, macro_events


def _fetch_news_context() -> str:
    """Kap. 12.2 (siehe src/news_feed.py für die vollständige Begründung,
    die Feed-Liste und deren Fest-ab-Studienstart-Charakter) - fester,
    täglich neu aggregierter Nachrichtentextblock aus den fünf definierten
    Feeds. Rein informativ (Prompt-Kontext für Claude), kein Guardrail. Kein
    Abbruch bei einem Fehler dieser Aggregation selbst: liefert dann einen
    leeren String zurück (kein Kontext diesen Lauf, statt den Lauf zu
    gefährden) - einzelne fehlgeschlagene Feeds werden bereits innerhalb von
    news_feed.fetch_all_feeds abgefangen, dieser Try/Except deckt nur einen
    unerwarteten Fehler in der Aggregation selbst ab."""
    try:
        feed_results = news_feed.fetch_all_feeds()
        return news_feed.build_news_text_block(feed_results)
    except Exception:
        log.exception("News-Feed-Aggregation (Kap. 12.2) fehlgeschlagen - wird übersprungen.")
        return ""


def _fetch_universe_fundamentals(universe_symbols: list[str]) -> dict[str, fundamentals.FundamentalSnapshot]:
    """Weicher Qualitäts-Score: grobe Fundamentaldaten je Titel, wo
    verfügbar - siehe src/fundamentals.py. Rein informativ (Prompt-Kontext
    für Claude), AUSDRÜCKLICH kein Filter/Guardrail. Kein Abbruch bei
    einem Fehler: liefert dann ein leeres Dict zurück (kein Kontext diesen
    Lauf, statt den Lauf zu gefährden)."""
    try:
        return fundamentals.fetch_universe_fundamentals(universe_symbols)
    except Exception:
        log.exception("Fundamentaldaten-Abfrage fehlgeschlagen - wird übersprungen.")
        return {}


def _check_boundary_conditions(
    conn,
    portfolio_id: int,
    current_prices: dict[str, float],
) -> tuple[list[boundary_conditions.BoundaryConditionCheck], list[boundary_conditions.BoundaryConditionCheck]]:
    """Kap. 7: mechanische Prüfung aller offenen Randbedingungen gegen die
    bereits geladenen `current_prices` - rein dokumentarisch, siehe
    src/boundary_conditions.py für die Prüflogik. Kein Abbruch bei einem
    Fehler dieses Checks selbst (analog zu _run_data_quality_checks)."""
    try:
        open_rows = db.get_open_boundary_conditions(conn, portfolio_id)
        checks = [
            boundary_conditions.BoundaryConditionCheck(
                id=r["id"],
                position_id=r["position_id"],
                symbol=r["symbol"],
                description=r["description"],
                check_type=r["check_type"],
                threshold_price=r["threshold_price"],
            )
            for r in open_rows
        ]
        triggered, still_open = boundary_conditions.evaluate_boundary_conditions(checks, current_prices)
        if triggered:
            db.mark_boundary_conditions_triggered(conn, [c.id for c in triggered])
        return triggered, still_open
    except Exception:
        log.exception("Randbedingungs-Prüfung (Kap. 7) fehlgeschlagen - wird übersprungen.")
        return [], []


def _nav_value_at_or_before(dates: list[pd.Timestamp], values: list[float], target: pd.Timestamp) -> float | None:
    """Letzter bekannter Wert aus `values` (nav/benchmark_normalized/
    baseline_normalized aus einer NavHistory) an oder vor `target` - genutzt,
    um den Stand zu Beginn einer Tiefenreflexions-Periode aus der ohnehin
    bereits fuer den Report berechneten NavHistory zu extrahieren, ohne eine
    weitere yfinance-Historie abzufragen."""
    eligible = [v for d, v in zip(dates, values) if d <= target]
    return eligible[-1] if eligible else None


def _maybe_run_deep_reflection(
    conn,
    app_config: AppConfig,
    portfolio_row,
    nav_history,
    metrics_result,
    today: pd.Timestamp | None = None,
):
    """Kap. 6.12.3: monatliche Tiefenreflexion zusätzlich zum täglichen
    Ablauf, ab Woche 5 der offiziellen Studie (rollierende 4-Wochen-Perioden
    ab deep_reflection_prompt.OFFICIAL_STUDY_START - siehe dort). Rein
    analytisch (keine Orders, kein Risk-Guardrail-Pfad); das Ergebnis wird
    als eigener 'deep_reflection'-Decision-Eintrag gespeichert und fliesst
    ab dem nächsten Lauf über db.get_latest_reflection/build_user_prompt in
    die tägliche Entscheidung ein.

    `today` ist ein Test-Seam (Default None -> echtes Tagesdatum) - erlaubt
    Tests, einen konkreten, in einer fälligen Periode liegenden Tag zu
    injizieren, ohne die Systemzeit zu mocken.

    Selbstkonsistenz-Prüfung (2026-09-19, siehe deep_reflection_schema.
    check_self_consistency): AUSSCHLIESSLICH hier, NICHT im täglichen
    Handelsablauf - Kostengründe (die Tiefenreflexion läuft nur alle 4
    Wochen, ein zusätzlicher Claude-Aufruf fällt dort kaum ins Gewicht;
    im täglichen Ablauf, der bei jedem Lauf ausgeführt wird, würde er die
    API-Kosten verdoppeln). Claude wird für dieselbe Periode zweimal mit
    IDENTISCHEM Prompt aufgerufen; weichen die Kernaussagen (bestätigte/
    widerlegte Thesen) ab, gibt es KEINE automatische Konfliktlösung - beide
    Antworten werden persistiert/dokumentiert (siehe unten und reporting.py),
    die erste dient unverändert als Grundlage für den nächsten Lauf.

    Weder ein Ausbleiben (noch nicht fällig) noch ein Fehler dieses Aufrufs
    selbst (Claude-API-Fehler, kein valides JSON bei EINEM der beiden
    Aufrufe) darf den für diesen Lauf bereits abgeschlossenen Handelsteil
    oder den Report gefährden - bei einem Fehler wird NICHTS in der DB
    vermerkt, sodass derselbe fällige Zeitraum beim nächsten Lauf
    automatisch erneut versucht wird (siehe
    deep_reflection_prompt.due_reflection_period's Docstring).
    """
    reflection_count = db.count_deep_reflections(conn, portfolio_row["id"])
    today = today if today is not None else pd.Timestamp(datetime.date.today())
    due_period = deep_reflection_prompt.due_reflection_period(reflection_count, today)
    if due_period is None:
        return None
    period_start, period_end = due_period
    log.info(
        "Monatliche Tiefenreflexion fällig (Periode %d, %s bis %s)...",
        reflection_count + 1, period_start.date(), period_end.date(),
    )
    try:
        since = period_start.strftime("%Y-%m-%d %H:%M:%S")  # siehe db.get_decisions_since-Docstring
        decisions_in_period = db.get_decisions_since(conn, portfolio_row["id"], since)
        trades_in_period = db.get_trades_since(conn, portfolio_row["id"], since)

        nav_at_start = _nav_value_at_or_before(nav_history.dates, nav_history.nav, period_start) or nav_history.initial_nav
        benchmark_at_start = (
            _nav_value_at_or_before(nav_history.dates, nav_history.benchmark_normalized, period_start)
            or nav_history.initial_nav
        )
        baseline_at_start = (
            _nav_value_at_or_before(nav_history.dates, nav_history.baseline_normalized, period_start)
            or nav_history.initial_nav
        )

        user_prompt = deep_reflection_prompt.build_deep_reflection_user_prompt(
            portfolio_row,
            decisions_in_period,
            trades_in_period,
            period_start,
            period_end,
            period_portfolio_return_pct=metrics_result.current_nav / nav_at_start - 1,
            period_benchmark_return_pct=nav_history.benchmark_normalized[-1] / benchmark_at_start - 1,
            period_baseline_return_pct=nav_history.baseline_normalized[-1] / baseline_at_start - 1,
        )
        # Selbstkonsistenz-Pruefung (siehe Docstring oben): zwei unabhaengige
        # Aufrufe mit IDENTISCHEM System-/User-Prompt fuer dieselbe Periode.
        raw_response = get_trading_decision(
            deep_reflection_prompt.DEEP_REFLECTION_SYSTEM_PROMPT,
            user_prompt,
            api_key=app_config.anthropic_api_key,
            model=app_config.claude_model,
        )
        raw_response_2 = get_trading_decision(
            deep_reflection_prompt.DEEP_REFLECTION_SYSTEM_PROMPT,
            user_prompt,
            api_key=app_config.anthropic_api_key,
            model=app_config.claude_model,
        )
        reflection = deep_reflection_schema.parse_reflection_from_json(raw_response)
        reflection_2 = deep_reflection_schema.parse_reflection_from_json(raw_response_2)
        consistency = deep_reflection_schema.check_self_consistency(reflection, reflection_2)
        if not consistency.consistent:
            log.warning(
                "Tiefenreflexion: Selbstkonsistenz-Prüfung zeigt Abweichung zwischen den beiden "
                "Aufrufen (%s) - keine automatische Konfliktlösung, beide Antworten werden "
                "dokumentiert.",
                "; ".join(consistency.mismatch_details),
            )
    except Exception:
        log.exception(
            "Monatliche Tiefenreflexion fehlgeschlagen - Report wird trotzdem erstellt, derselbe "
            "Zeitraum wird beim nächsten fälligen Lauf automatisch erneut versucht."
        )
        return None

    db.insert_decision(
        conn,
        portfolio_id=portfolio_row["id"],
        model="deep_reflection",
        prompt=user_prompt,
        raw_response=raw_response,
        proposed_orders=None,
        # Zweite Antwort + Konsistenz-Ergebnis landen hier (statt in eigenen
        # Spalten) - risk_check_result ist bereits das generische JSON-Feld
        # fuer Neben-Ergebnisse ausserhalb des eigentlichen Guardrail-Pfads
        # (siehe z.B. die {"info": ...}/{"error": ...}-Verwendung anderswo in
        # dieser Datei); keine Schema-Migration fuer diesen Zusatzbefund noetig.
        risk_check_result=[{
            "self_consistency_check": {
                "consistent": consistency.consistent,
                "mismatch_details": consistency.mismatch_details,
                "second_raw_response": raw_response_2,
            }
        }],
        rationale=reflection.reflection_commentary,
        forced_action=False,
        approved=True,
        executed=False,
    )
    log.info("Tiefenreflexion abgeschlossen und gespeichert.")
    return deep_reflection_schema.DeepReflectionRunResult(
        primary=reflection, secondary=reflection_2, consistency=consistency
    )


def run(force: bool | None = None) -> None:
    """`force` überschreibt die Idempotenz-Sperre (siehe Block-Kommentar bei
    _is_force_rerun_requested oben) für eine bewusste manuelle Wiederholung
    am selben Tag - None (Default) lässt die Entscheidung an die
    Umgebungsvariable FORCE_RERUN, True erzwingt einen Rerun unabhängig
    davon, False erzwingt NIE einen Rerun (auch wenn FORCE_RERUN gesetzt
    wäre - z.B. für Tests, die die Sperre gezielt prüfen wollen)."""
    app_config = AppConfig.load()
    risk_config = RiskConfig.from_yaml(app_config.risk_config_path)
    watchlist = Watchlist.from_yaml(app_config.watchlist_path)
    watchlist_underlyings = {
        s.symbol: s.underlying_symbol for s in watchlist.symbols if s.underlying_symbol
    }
    structured_product_symbols = {
        s.symbol for s in watchlist.symbols if s.instrument_type in STRUCTURED_INSTRUMENT_TYPES
    }
    symbol_metadata = {s.symbol: s for s in watchlist.symbols}
    # Kap. 6.9: Universum der regelbasierten Momentum-Baseline - reguläre
    # Aktien des Anhang-A-Universums, ohne die gehebelten ETF-Proxies
    # (NVDL/TSDD) und ohne strukturierte Produkte (siehe momentum_baseline.py).
    momentum_universe_symbols = [s.symbol for s in watchlist.symbols if s.instrument_type == "equity"]

    broker_client = broker_alpaca.get_trading_client(app_config.alpaca_api_key, app_config.alpaca_secret_key)

    with db.get_connection(app_config.db_path) as conn:
        db.ensure_nav_history_table(conn)  # idempotente Migration, siehe db.py-Docstring
        db.ensure_trade_transaction_cost_column(conn)  # idempotente Migration, siehe db.py-Docstring
        db.ensure_boundary_conditions_table(conn)  # idempotente Migration, siehe db.py-Docstring
        portfolio_row = db.get_portfolio(conn, app_config.portfolio_name)
        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])

        # EINMALIGE Uebergangs-Sicherheitspruefung, siehe Block-Kommentar bei
        # _pilot_phase_positions_still_open oben - bewusst VOR jedem
        # Marktdaten-Abruf/Claude-Aufruf, damit ein blockierter Lauf wirklich
        # sauber abbricht statt nur den Handelsteil zu ueberspringen.
        pending_reset_positions = _pilot_phase_positions_still_open(open_position_rows)
        if pending_reset_positions:
            message = (
                f"Reset noch nicht durchgeführt, Lauf übersprungen: {len(pending_reset_positions)} "
                f"offene Position(en) aus der Pilotphase (eröffnet vor "
                f"{deep_reflection_prompt.OFFICIAL_STUDY_START.date()}) gefunden - siehe "
                "RESET_2026-09-21.md für den geplanten Reset."
            )
            log.error(message)
            db.insert_decision(
                conn,
                portfolio_id=portfolio_row["id"],
                model="pipeline_guard",
                prompt="(kein Prompt - Uebergangs-Sicherheitspruefung, Claude wurde nicht aufgerufen)",
                raw_response=None,
                proposed_orders=None,
                risk_check_result=[{"info": message}],
                rationale=message,
                forced_action=False,
                approved=False,
                executed=False,
            )
            return

        # Idempotenz-Sperre (siehe Block-Kommentar bei _is_force_rerun_requested
        # oben) - ebenfalls VOR jedem Marktdaten-Abruf/Claude-Aufruf, damit ein
        # blockierter Lauf wirklich nichts weiter tut (kein zweiter Claude-
        # Aufruf, kein zusaetzlicher Trade), statt nur den Handelsteil zu
        # ueberspringen.
        existing_decision = _already_decided_today(conn, portfolio_row["id"], app_config.claude_model)
        if existing_decision is not None and not _is_force_rerun_requested(force):
            message = (
                f"Für heute liegt bereits eine abgeschlossene Handelsentscheidung vor "
                f"(decisions.id={existing_decision['id']}, {existing_decision['created_at']}) - Lauf "
                "übersprungen, um eine doppelte Ausführung am selben Tag zu vermeiden. Für eine bewusste "
                "manuelle Wiederholung FORCE_RERUN=true setzen oder force=True übergeben."
            )
            log.warning(message)
            db.insert_decision(
                conn,
                portfolio_id=portfolio_row["id"],
                model="pipeline_guard",
                prompt="(kein Prompt - Idempotenz-Sperre, Claude wurde nicht aufgerufen)",
                raw_response=None,
                proposed_orders=None,
                risk_check_result=[{"info": message}],
                rationale=message,
                forced_action=False,
                approved=False,
                executed=False,
            )
            return

        price_lookup_symbols = sorted(
            set(watchlist.all_symbols())
            | {r["symbol"] for r in open_position_rows}
            | {watchlist_underlyings.get(r["symbol"], r["symbol"]) for r in open_position_rows}
            | set(watchlist_underlyings.values())
        )
        log.info("Lade Marktdaten für %d Symbole via yfinance...", len(price_lookup_symbols))
        snapshots = data_fetch.fetch_market_snapshots(
            price_lookup_symbols, known_unresolvable_symbols=structured_product_symbols
        )
        current_prices = {s: snap.last_price for s, snap in snapshots.items()}
        # Kap. 6.13 Liquiditätslimit (2026-09-21, siehe risk_guardrails.
        # check_liquidity_limit) - nutzt das ohnehin bereits pro Symbol
        # abgerufene volume-Feld aus `snapshots`, kein zusätzlicher Datenabruf.
        average_daily_volumes = {
            s: snap.volume for s, snap in snapshots.items() if snap.volume is not None
        }

        log.info("Berechne volatilitätsadjustierte Positionsgrössen-Skalierung (rollierende 20-Tage-Vol)...")
        volatility_scaling = _compute_volatility_scaling(snapshots, risk_config)

        log.info("Berechne regelbasierte Markt-Phasen-Klassifikation (Bull/Bear/Seitwärts) je Symbol...")
        market_phases = _compute_market_phases(snapshots)

        log.info("Führe Datenqualitäts-Checks durch (yfinance vs. Alpaca, Lücken/Ausreisser)...")
        tradable_symbols = [s for s in price_lookup_symbols if s not in structured_product_symbols]
        dq_report = _run_data_quality_checks(
            app_config, tradable_symbols, structured_product_symbols, current_prices
        )
        for line in dq_report.log_lines():
            log.warning("Datenqualität: %s", line)

        log.info("Berechne rollierende 60-Tage-Korrelationsmatrix für das Universum...")
        correlation_matrix = _compute_correlation_matrix(watchlist.all_symbols(), structured_product_symbols)

        log.info("Prüfe Event-Kalender (Earnings, FOMC/CPI)...")
        earnings_warnings, macro_events = _check_upcoming_events(watchlist.all_symbols())
        for w in earnings_warnings:
            log.info("Event-Hinweis: %s berichtet in %d Handelstag(en) - erhöhtes Ereignisrisiko.", w.symbol, w.trading_days_until)
        for m in macro_events:
            log.info("Event-Hinweis: %s in %d Handelstag(en) - erhöhtes Ereignisrisiko fürs Portfolio.", m.name, m.trading_days_until)

        log.info("Lade grobe Fundamentaldaten für das Universum (weicher Kontext, kein Filter)...")
        universe_fundamentals = _fetch_universe_fundamentals(watchlist.all_symbols())

        log.info("Aggregiere News-Feed-Textblock (Kap. 12.2, fest ab Studienstart)...")
        news_text_block = _fetch_news_context()

        log.info("Prüfe Randbedingungen (Kap. 7) offener Positionen...")
        triggered_boundary_conditions, still_open_boundary_conditions = _check_boundary_conditions(
            conn, portfolio_row["id"], current_prices
        )
        for c in triggered_boundary_conditions:
            log.warning("Randbedingung ausgelöst für %s: %s", c.symbol, c.description)

        start_of_run_nav = compute_nav(
            portfolio_row["cash_balance"],
            db.open_positions_as_risk_objects(open_position_rows),
            current_prices,
        )
        log.info("NAV zu Lauf-Beginn: %.2f", start_of_run_nav)

        # Persistiert den Lauf-Start-NAV in nav_history und liest danach den
        # echten historischen Höchststand (MAX über alle bisherigen Läufe)
        # zurück - Grundlage für den Kap.-6.8-Circuit-Breaker. Reihenfolge
        # wichtig: erst schreiben, dann lesen, damit ein neuer Höchststand in
        # genau diesem Lauf sofort mitzählt.
        db.record_nav(conn, portfolio_row["id"], start_of_run_nav)
        peak_nav = db.get_peak_nav(conn, portfolio_row["id"])
        log.info("NAV-Höchststand (historisch): %.2f", peak_nav)

        log.info("Prüfe offene Short-Positionen auf Stop-Loss-Trigger...")
        forced_actions = execution.run_short_stop_loss_sweep(
            conn, portfolio_row, current_prices, risk_config, broker_client
        )
        if forced_actions:
            log.warning("%d Short-Position(en) zwangsweise geschlossen (Stop-Loss).", len(forced_actions))
        portfolio_row = db.get_portfolio(conn, app_config.portfolio_name)  # cash_balance kann sich geändert haben

        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])

        executed_results = []
        portfolio_commentary = ""

        if not broker_alpaca.is_trading_day(broker_client):
            # Börse heute komplett geschlossen (Wochenende/Feiertag) - Market-
            # Orders könnten ohnehin nicht füllen (siehe cron-Timing-Fix), also
            # wird erst gar kein Claude-Aufruf gemacht (spart API-Kosten) und
            # keine Order versucht. Trotzdem als Decision dokumentiert, damit
            # jeder Lauf nachvollziehbar bleibt.
            log.info("Börse heute geschlossen (Wochenende/Feiertag) - kein Handelsversuch, Claude wird nicht aufgerufen.")
            portfolio_commentary = "Börse heute geschlossen, kein Handelsversuch."
            db.insert_decision(
                conn,
                portfolio_id=portfolio_row["id"],
                model=app_config.claude_model,
                prompt="(kein Prompt - Börse heute geschlossen, Claude wurde nicht aufgerufen)",
                raw_response=None,
                proposed_orders=None,
                risk_check_result=[{"info": portfolio_commentary}],
                rationale=portfolio_commentary,
                forced_action=False,
                approved=False,
                executed=False,
            )
        else:
            latest_reflection = db.get_latest_reflection(conn, portfolio_row["id"])
            prompt = build_user_prompt(
                portfolio_row, open_position_rows, watchlist, snapshots, risk_config,
                latest_reflection=latest_reflection,
                triggered_boundary_conditions=triggered_boundary_conditions,
                still_open_boundary_conditions=still_open_boundary_conditions,
                earnings_warnings=earnings_warnings,
                macro_events=macro_events,
                fundamentals=universe_fundamentals,
                news_text_block=news_text_block,
            )

            log.info("Rufe Claude (%s) für Handelsentscheidung auf...", app_config.claude_model)
            raw_response = get_trading_decision(
                SYSTEM_PROMPT, prompt, api_key=app_config.anthropic_api_key, model=app_config.claude_model
            )

            try:
                decision_output = parse_orders_from_json(raw_response)
                portfolio_commentary = decision_output.portfolio_commentary
                log.info("Claude schlägt %d Order(s) vor.", len(decision_output.orders))
                executed_results = execution.execute_proposed_orders(
                    conn,
                    portfolio_row,
                    decision_output.orders,
                    model=app_config.claude_model,
                    prompt=prompt,
                    raw_response=raw_response,
                    risk_config=risk_config,
                    current_prices=current_prices,
                    start_of_run_nav=start_of_run_nav,
                    broker_client=broker_client,
                    symbol_metadata=symbol_metadata,
                    peak_nav=peak_nav,
                    correlation_matrix=correlation_matrix,
                    volatility_scaling=volatility_scaling,
                    market_phases=market_phases,
                    average_daily_volumes=average_daily_volumes,
                )
            except OrderParsingError as exc:
                log.error("Konnte Claude-Antwort nicht parsen: %s", exc)
                db.insert_decision(
                    conn,
                    portfolio_id=portfolio_row["id"],
                    model=app_config.claude_model,
                    prompt=prompt,
                    raw_response=raw_response,
                    proposed_orders=None,
                    risk_check_result=[{"error": str(exc)}],
                    rationale=f"Parsing fehlgeschlagen: {exc}",
                    forced_action=False,
                    approved=False,
                    executed=False,
                )
                portfolio_commentary = f"(Antwort konnte nicht geparst werden: {exc})"

        portfolio_row = db.get_portfolio(conn, app_config.portfolio_name)
        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])
        all_trades = db.get_all_trades(conn, portfolio_row["id"])

        log.info("Berechne Metriken...")
        nav_history = metrics.reconstruct_nav_history(
            conn,
            portfolio_row,
            all_trades,
            watchlist_underlyings,
            portfolio_row["benchmark_symbol"],
            momentum_universe_symbols=momentum_universe_symbols,
        )
        metrics_result = metrics.compute_metrics(
            nav_history, risk_free_rate_annual=app_config.risk_free_rate_annual
        )

        deep_reflection_result = _maybe_run_deep_reflection(
            conn, app_config, portfolio_row, nav_history, metrics_result
        )

        try:
            open_position_symbols = sorted({r["symbol"] for r in open_position_rows})
            correlation_cluster_count = correlation.compute_correlation_clusters(
                open_position_symbols, correlation_matrix
            )
        except Exception:
            log.exception("Korrelations-Cluster-Berechnung fehlgeschlagen - wird im Report als 'n/a' vermerkt.")
            correlation_cluster_count = None

        report_path = reporting.generate_report(
            portfolio_row,
            open_position_rows,
            executed_results,
            forced_actions,
            portfolio_commentary,
            metrics_result,
            dq_report,
            deep_reflection_result,
            triggered_boundary_conditions,
            still_open_boundary_conditions,
            correlation_cluster_count,
            earnings_warnings,
            macro_events,
            app_config.reports_dir,
        )
        log.info("Report geschrieben: %s", report_path)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log.exception("Pipeline-Lauf fehlgeschlagen.")
        sys.exit(1)
