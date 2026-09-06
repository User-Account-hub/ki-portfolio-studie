"""Main entrypoint for one pipeline run (intended to be triggered weekly by CI).

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

import logging
import sys

from src import broker_alpaca, data_fetch, db, execution, metrics, reporting
from src.claude_client import get_trading_decision
from src.config import AppConfig, RiskConfig, Watchlist
from src.order_schema import STRUCTURED_INSTRUMENT_TYPES, OrderParsingError, parse_orders_from_json
from src.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.risk_guardrails import compute_nav

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


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

    broker_client = broker_alpaca.get_trading_client(app_config.alpaca_api_key, app_config.alpaca_secret_key)

    with db.get_connection(app_config.db_path) as conn:
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

        start_of_run_nav = compute_nav(
            portfolio_row["cash_balance"],
            db.open_positions_as_risk_objects(open_position_rows),
            current_prices,
        )
        log.info("NAV zu Lauf-Beginn: %.2f", start_of_run_nav)
        # Vereinfachter NAV-Höchststand für den Kap.-6.8-Circuit-Breaker: ohne
        # dedizierte NAV-Historie (siehe metrics.py-Docstring) wird das
        # Maximum aus Startkapital und aktuellem Lauf-Start-NAV verwendet,
        # statt für jeden Lauf zusätzlich die volle Historie zu rekonstruieren.
        peak_nav = max(app_config.initial_cash_balance, start_of_run_nav)

        log.info("Prüfe offene Short-Positionen auf Stop-Loss-Trigger...")
        forced_actions = execution.run_short_stop_loss_sweep(
            conn, portfolio_row, current_prices, risk_config, broker_client
        )
        if forced_actions:
            log.warning("%d Short-Position(en) zwangsweise geschlossen (Stop-Loss).", len(forced_actions))
        portfolio_row = db.get_portfolio(conn, app_config.portfolio_name)  # cash_balance kann sich geändert haben

        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])
        prompt = build_user_prompt(portfolio_row, open_position_rows, watchlist, snapshots, risk_config)

        log.info("Rufe Claude (%s) für Handelsentscheidung auf...", app_config.claude_model)
        raw_response = get_trading_decision(
            SYSTEM_PROMPT, prompt, api_key=app_config.anthropic_api_key, model=app_config.claude_model
        )

        executed_results = []
        portfolio_commentary = ""
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
            conn, portfolio_row, all_trades, watchlist_underlyings, portfolio_row["benchmark_symbol"]
        )
        metrics_result = metrics.compute_metrics(nav_history)

        report_path = reporting.generate_report(
            portfolio_row,
            open_position_rows,
            executed_results,
            forced_actions,
            portfolio_commentary,
            metrics_result,
            app_config.reports_dir,
        )
        log.info("Report geschrieben: %s", report_path)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log.exception("Pipeline-Lauf fehlgeschlagen.")
        sys.exit(1)
