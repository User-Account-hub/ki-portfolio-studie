"""Main entrypoint for one pipeline run (triggered twice weekly by CI - Monday
and Thursday, see .github/workflows/weekly_pipeline.yml; src/metrics.py's
reconstruct_nav_history must be kept in sync with this cadence via its
`freqs` parameter).

Steps:
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
import sys

import pandas as pd

from src import broker_alpaca, data_fetch, data_quality, db, deep_reflection_prompt, deep_reflection_schema, execution, metrics, reporting
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
):
    """Kap. 6.12.3: monatliche Tiefenreflexion zusätzlich zum täglichen
    Ablauf, ab Woche 5 der offiziellen Studie (rollierende 4-Wochen-Perioden
    ab deep_reflection_prompt.OFFICIAL_STUDY_START - siehe dort). Rein
    analytisch (keine Orders, kein Risk-Guardrail-Pfad); das Ergebnis wird
    als eigener 'deep_reflection'-Decision-Eintrag gespeichert und fliesst
    ab dem nächsten Lauf über db.get_latest_reflection/build_user_prompt in
    die tägliche Entscheidung ein.

    Weder ein Ausbleiben (noch nicht fällig) noch ein Fehler dieses Aufrufs
    selbst (Claude-API-Fehler, kein valides JSON) darf den für diesen Lauf
    bereits abgeschlossenen Handelsteil oder den Report gefährden - bei
    einem Fehler wird NICHTS in der DB vermerkt, sodass derselbe fällige
    Zeitraum beim nächsten Lauf automatisch erneut versucht wird (siehe
    deep_reflection_prompt.due_reflection_period's Docstring).
    """
    reflection_count = db.count_deep_reflections(conn, portfolio_row["id"])
    today = pd.Timestamp(datetime.date.today())
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
        raw_response = get_trading_decision(
            deep_reflection_prompt.DEEP_REFLECTION_SYSTEM_PROMPT,
            user_prompt,
            api_key=app_config.anthropic_api_key,
            model=app_config.claude_model,
        )
        reflection = deep_reflection_schema.parse_reflection_from_json(raw_response)
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
        risk_check_result=None,
        rationale=reflection.reflection_commentary,
        forced_action=False,
        approved=True,
        executed=False,
    )
    log.info("Tiefenreflexion abgeschlossen und gespeichert.")
    return reflection


def run() -> None:
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
        portfolio_row = db.get_portfolio(conn, app_config.portfolio_name)
        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])

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

        log.info("Führe Datenqualitäts-Checks durch (yfinance vs. Alpaca, Lücken/Ausreisser)...")
        tradable_symbols = [s for s in price_lookup_symbols if s not in structured_product_symbols]
        dq_report = _run_data_quality_checks(
            app_config, tradable_symbols, structured_product_symbols, current_prices
        )
        for line in dq_report.log_lines():
            log.warning("Datenqualität: %s", line)

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

        report_path = reporting.generate_report(
            portfolio_row,
            open_position_rows,
            executed_results,
            forced_actions,
            portfolio_commentary,
            metrics_result,
            dq_report,
            deep_reflection_result,
            app_config.reports_dir,
        )
        log.info("Report geschrieben: %s", report_path)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log.exception("Pipeline-Lauf fehlgeschlagen.")
        sys.exit(1)
