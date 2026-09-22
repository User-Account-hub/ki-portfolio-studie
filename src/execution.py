"""Orchestrates: pre-trade risk checks -> broker/simulated execution -> DB updates.

Cash-flow convention (documented once here, applied consistently):
  buy   -> cash -= notional   (long position opened/increased)
  sell  -> cash += notional   (long position reduced/closed)
  short -> cash += notional   (short sale proceeds; short position opened/increased)
  cover -> cash -= notional   (buy-back cost; short position reduced/closed)
This matches real cash economics and keeps NAV (see risk_guardrails.compute_nav)
unchanged at the instant a trade is opened - only price movement thereafter
produces P&L.

Transaction cost (2026-09-17, risk_config.yaml's transaction_cost_pct_of_notional):
a fixed spread/slippage allowance, `notional * transaction_cost_pct_of_notional`,
is additionally deducted from cash on EVERY fill regardless of direction (buy,
sell, short, and cover all erode cash a little via this cost) - see
_apply_fill_to_db/execute_forced_stop_loss_actions. Stored per-trade on
trades.transaction_cost so metrics.py's ledger replay stays consistent with the
live cash_balance instead of silently drifting from it.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from src import broker_alpaca, correlation, db, market_phase, position_sizing
from src.config import RiskConfig
from src.order_schema import STRUCTURED_INSTRUMENT_TYPES, TRADABLE_INSTRUMENT_TYPES, OrderSide, ProposedOrder
from src.risk_guardrails import (
    ForcedStopLossAction,
    OpenPosition,
    PortfolioContext,
    compute_nav,
    evaluate_order,
    evaluate_short_positions_for_stop_loss,
)

CASH_DIRECTION = {
    OrderSide.BUY: -1,
    OrderSide.SELL: +1,
    OrderSide.SHORT: +1,
    OrderSide.COVER: -1,
}

# Maps an order side onto the position side it opens/reduces.
POSITION_SIDE = {
    OrderSide.BUY: "long",
    OrderSide.SELL: "long",
    OrderSide.SHORT: "short",
    OrderSide.COVER: "short",
}

INCREASES_POSITION = {OrderSide.BUY, OrderSide.SHORT}

log = logging.getLogger("pipeline")


@dataclass
class ExecutedOrderResult:
    order: ProposedOrder
    approved: bool
    reasons: list[str]
    trade_id: int | None = None
    fill_price: float | None = None
    # Korrelations-Beobachtung (rein dokumentarisch, kein Guardrail - siehe
    # src/correlation.py) - nur fuer genehmigte BUY-Orders befuellt.
    correlation_warnings: list[correlation.CorrelationWarning] = field(default_factory=list)
    # Volatilitaetsadjustierte Positionsgroessen-Skalierung (siehe
    # src/position_sizing.py) - None, falls fuer diese Order keine Skalierung
    # angewendet wurde (sell/cover, oder keine Vola-Daten fuer das Symbol).
    # `order` oben traegt bereits die skalierte Groesse, falls vorhanden.
    volatility_scaling: position_sizing.VolatilityScaling | None = None
    # Konviktions-Multiplikator (siehe src/position_sizing.py) - None, falls
    # nicht anwendbar (sell/cover); 1.0, falls anwendbar aber Claude keine
    # Konviktion angegeben hat oder "medium" gewaehlt wurde.
    conviction_scaling_factor: float | None = None
    # Markt-Phasen-Abgleich (rein dokumentarisch, kein Veto - siehe
    # src/market_phase.py) - None, wenn kein Widerspruch gefunden wurde
    # (oder kein Vergleich moeglich war: Claude ohne cycle_position-Angabe,
    # oder keine Regel-Klassifikation fuers Symbol verfuegbar).
    market_phase_contradiction: market_phase.MarketPhaseContradiction | None = None
    # Bugfix 2026-09-21 (17-Punkte-Audit Fund #3, real eingetreten - siehe
    # execute_proposed_orders' try/except unten): True, wenn diese Order
    # NICHT wegen eines Guardrail-Vetos abgelehnt wurde, sondern weil ihre
    # Ausfuehrung selbst eine unerwartete Exception geworfen hat (z.B. eine
    # vom Broker abgelehnte fraktionierte Short-Order). Unterscheidet einen
    # echten Fehler/Datenintegritaets-Fall im Report von einer normalen,
    # planmaessigen Guardrail-Ablehnung (approved=False, execution_error=
    # False) - siehe reporting.py.
    execution_error: bool = False


def resolve_price(symbol: str, underlying_symbol: str | None, current_prices: dict[str, float]) -> float:
    if symbol in current_prices:
        return current_prices[symbol]
    if underlying_symbol and underlying_symbol in current_prices:
        # Dokumentierte Vereinfachung: ohne eigenes Kurs-Feed für das strukturierte
        # Produkt wird der Kurs des Basiswerts als Proxy verwendet.
        return current_prices[underlying_symbol]
    raise ValueError(f"Kein Kurs verfügbar für '{symbol}' (auch nicht über Basiswert '{underlying_symbol}').")


def build_context(
    portfolio_row: sqlite3.Row,
    open_position_rows: list[sqlite3.Row],
    trades_today_by_symbol: dict[str, int],
    current_prices: dict[str, float],
    start_of_run_nav: float | None,
    symbol_metadata: dict | None = None,
    peak_nav: float | None = None,
) -> PortfolioContext:
    positions = db.open_positions_as_risk_objects(open_position_rows, symbol_metadata)
    nav = compute_nav(portfolio_row["cash_balance"], positions, current_prices)
    return PortfolioContext(
        nav=nav,
        cash=portfolio_row["cash_balance"],
        positions=positions,
        trades_today_by_symbol=trades_today_by_symbol,
        start_of_run_nav=start_of_run_nav,
        peak_nav=peak_nav,
    )


def _route_fill(
    order_side: OrderSide,
    symbol: str,
    instrument_type: str,
    quantity: float,
    order_type: str,
    limit_price: float | None,
    price: float,
    broker_client,
) -> tuple[broker_alpaca.FillResult, str]:
    """Returns (fill_result, source) - source is 'alpaca' or 'manual_simulation'."""
    if instrument_type in TRADABLE_INSTRUMENT_TYPES:
        fill = broker_alpaca.submit_equity_order(
            broker_client,
            symbol=symbol,
            quantity=quantity,
            side=order_side.value,
            order_type=order_type,
            limit_price=limit_price,
        )
        return fill, "alpaca"
    assert instrument_type in STRUCTURED_INSTRUMENT_TYPES
    fill = broker_alpaca.simulate_structured_product_fill(quantity, price)
    return fill, "manual_simulation"


def _apply_fill_to_db(
    conn: sqlite3.Connection,
    portfolio_id: int,
    decision_id: int | None,
    order: ProposedOrder,
    order_side: OrderSide,
    fill: broker_alpaca.FillResult,
    source: str,
    transaction_cost_pct: float,
) -> int:
    position_side = POSITION_SIDE[order_side]
    quantity = fill.filled_qty
    notional = quantity * fill.filled_price
    # Feste Spread/Slippage-Pauschale (2026-09-17, risk_config.yaml) - trifft
    # das Cash unabhaengig von der Handelsrichtung, siehe Modul-Docstring.
    transaction_cost = notional * transaction_cost_pct

    if order_side in INCREASES_POSITION:
        position_id = db.upsert_open_position(
            conn,
            portfolio_id=portfolio_id,
            symbol=order.symbol,
            instrument_type=order.instrument_type.value,
            underlying_symbol=order.underlying_symbol,
            side=position_side,
            delta_quantity=quantity,
            fill_price=fill.filled_price,
            stop_loss_price=order.stop_loss_price,
        )
        # Kap. 7: von Claude genannte Randbedingungen fuer diese These
        # speichern (optional/best-effort - order.boundary_conditions ist
        # meist leer, kein Fehler). decision_id bleibt bewusst None (die
        # Decision existiert an dieser Stelle im Ablauf noch nicht, siehe
        # execute_proposed_orders) - position_id reicht fuer die Zuordnung.
        for bc in order.boundary_conditions:
            db.insert_boundary_condition(
                conn,
                portfolio_id=portfolio_id,
                position_id=position_id,
                symbol=order.symbol,
                description=bc.description,
                check_type=bc.check_type.value,
                threshold_price=bc.threshold_price,
            )
    else:
        row = conn.execute(
            "SELECT id FROM positions WHERE portfolio_id = ? AND symbol = ? AND side = ? AND status = 'open'",
            (portfolio_id, order.symbol, position_side),
        ).fetchone()
        if row is None:
            raise ValueError(f"Keine offene {position_side}-Position für {order.symbol} zum Reduzieren gefunden.")
        position_id = row["id"]
        db.reduce_or_close_position(
            conn, position_id, delta_quantity=quantity, fill_price=fill.filled_price, closure_reason="claude_decision"
        )

    db.update_cash_balance(conn, portfolio_id, CASH_DIRECTION[order_side] * notional - transaction_cost)

    return db.insert_trade(
        conn,
        portfolio_id=portfolio_id,
        position_id=position_id,
        decision_id=decision_id,
        symbol=order.symbol,
        instrument_type=order.instrument_type.value,
        side=order_side.value,
        quantity=quantity,
        price=fill.filled_price,
        order_type=order.order_type,
        broker_order_id=fill.broker_order_id or None,
        source=source,
        status=fill.status,
        transaction_cost=transaction_cost,
    )


def execute_forced_stop_loss_actions(
    conn: sqlite3.Connection,
    portfolio_row: sqlite3.Row,
    forced_actions: list[ForcedStopLossAction],
    broker_client,
    transaction_cost_pct: float,
) -> list[int]:
    """Executes mandatory short-covers regardless of the daily-loss-stop gate.

    Each action is logged as its own forced_action decision row for
    documentation-compliance (short stop-loss triggers must be traceable).
    """
    trade_ids = []
    for action in forced_actions:
        decision_id = db.insert_decision(
            conn,
            portfolio_id=portfolio_row["id"],
            model="risk_guardrail",
            prompt="(automatischer Short-Stop-Loss-Sweep, kein Claude-Aufruf)",
            raw_response=None,
            proposed_orders=[{"symbol": action.symbol, "side": "cover", "quantity": action.quantity}],
            risk_check_result=[{"approved": True, "reasons": [action.documentation]}],
            rationale=action.documentation,
            forced_action=True,
            approved=True,
            executed=True,
        )
        fill, source = _route_fill(
            OrderSide.COVER,
            action.symbol,
            action.instrument_type,
            action.quantity,
            "market",
            None,
            action.current_price,
            broker_client,
        )

        if fill.filled_qty <= 0:
            # Gleiche Absicherung wie in execute_proposed_orders: ein vom Broker
            # nicht gefüllter Cover würde sonst am CHECK(quantity > 0) crashen.
            # Die Position bleibt offen und wird beim nächsten Lauf erneut geprüft;
            # die fehlgeschlagene Zwangsschliessung wird nur geloggt, nicht stillschweigend
            # verworfen (Dokumentationspflicht bleibt über decision_id-Eintrag erhalten).
            log.error(
                "Pflicht-Stop-Loss-Cover für %s nicht ausgeführt (Broker meldete Menge %s, Status %s) - "
                "Position bleibt offen, wird im nächsten Lauf erneut geprüft.",
                action.symbol,
                fill.filled_qty,
                fill.status,
            )
            continue

        # closure_notes trägt die Dokumentationspflicht direkt auf der Position mit.
        position_side = POSITION_SIDE[OrderSide.COVER]
        row = conn.execute(
            "SELECT id FROM positions WHERE portfolio_id = ? AND symbol = ? AND side = ? AND status = 'open'",
            (portfolio_row["id"], action.symbol, position_side),
        ).fetchone()
        if row is not None:
            db.reduce_or_close_position(
                conn,
                row["id"],
                delta_quantity=fill.filled_qty,
                fill_price=fill.filled_price,
                closure_reason="short_stop_loss_forced",
                closure_notes=action.documentation,
            )
            # Feste Spread/Slippage-Pauschale gilt auch fuer Pflicht-Stop-Loss-
            # Covers - der Markt macht dabei keinen Unterschied zu freiwilligen Trades.
            forced_notional = fill.filled_qty * fill.filled_price
            forced_transaction_cost = forced_notional * transaction_cost_pct
            db.update_cash_balance(
                conn,
                portfolio_row["id"],
                CASH_DIRECTION[OrderSide.COVER] * forced_notional - forced_transaction_cost,
            )
            trade_id = db.insert_trade(
                conn,
                portfolio_id=portfolio_row["id"],
                position_id=row["id"],
                decision_id=decision_id,
                symbol=action.symbol,
                instrument_type=action.instrument_type,
                side="cover",
                quantity=fill.filled_qty,
                price=fill.filled_price,
                order_type="market",
                broker_order_id=fill.broker_order_id or None,
                source=source,
                status=fill.status,
                transaction_cost=forced_transaction_cost,
            )
            trade_ids.append(trade_id)
    return trade_ids


def execute_proposed_orders(
    conn: sqlite3.Connection,
    portfolio_row: sqlite3.Row,
    orders: list[ProposedOrder],
    model: str,
    prompt: str,
    raw_response: str,
    risk_config: RiskConfig,
    current_prices: dict[str, float],
    start_of_run_nav: float,
    broker_client,
    symbol_metadata: dict | None = None,
    peak_nav: float | None = None,
    correlation_matrix: pd.DataFrame | None = None,
    volatility_scaling: dict[str, position_sizing.VolatilityScaling] | None = None,
    market_phases: dict[str, market_phase.MarketPhaseClassification] | None = None,
    average_daily_volumes: dict[str, float] | None = None,
) -> list[ExecutedOrderResult]:
    results: list[ExecutedOrderResult] = []
    risk_check_log = []
    symbol_metadata = symbol_metadata or {}

    trades_today = db.count_trades_today_by_symbol(conn, portfolio_row["id"])

    for order in orders:
        try:
            price = resolve_price(order.symbol, order.underlying_symbol, current_prices)
        except ValueError as exc:
            results.append(ExecutedOrderResult(order=order, approved=False, reasons=[str(exc)]))
            risk_check_log.append({"symbol": order.symbol, "approved": False, "reasons": [str(exc)]})
            continue

        # Vor-Initialisiert (2026-09-21, Fund #3 unten), damit `scaling`/
        # `conviction_factor`/`contradiction` im except-Block auch dann
        # definiert sind, wenn eine Exception vor ihrer eigentlichen
        # Zuweisung weiter unten auftritt.
        scaling: position_sizing.VolatilityScaling | None = None
        conviction_factor: float | None = None
        contradiction: market_phase.MarketPhaseContradiction | None = None

        try:
            # Positionsgroessen-Skalierung (siehe src/position_sizing.py) - nur
            # fuer positionsaufbauende Seiten (buy/short) und VOR der Guardrail-
            # Pruefung unten, sodass die Kap.-6.8-Limiten anschliessend auf die
            # bereits skalierte Groesse angewendet werden (Verfeinerung innerhalb
            # der Limiten, kein zusaetzliches Veto und kein Aushebeln der
            # Limiten). Zwei unabhaengige, MULTIPLIKATIV kombinierte Faktoren:
            # (1) Volatilitaet - ein gemessenes Marktsignal, Band 0.5x-1.5x.
            # (2) Konviktion - Claudes eigene, optionale Selbsteinschaetzung
            #     (order.conviction); bewusst ein deutlich schwaecheres Band
            #     (siehe position_sizing.CONVICTION_SCALING_FACTORS' Kommentar
            #     fuer die ausfuehrliche Begruendung: unkalibrierte LLM-
            #     Selbsteinschaetzung soll die Positionsgroesse nur leicht
            #     nudgen, nicht substanziell treiben).
            if order.side in (OrderSide.BUY, OrderSide.SHORT):
                if volatility_scaling:
                    scaling = volatility_scaling.get(order.symbol)
                conviction_factor = position_sizing.conviction_scaling_factor(order.conviction)

            combined_factor = (scaling.scaling_factor if scaling is not None else 1.0) * (conviction_factor or 1.0)
            if combined_factor != 1.0:
                scaled_quantity, scaled_notional = position_sizing.scale_order_size(
                    order.quantity, order.notional, combined_factor
                )
                order = order.model_copy(update={"quantity": scaled_quantity, "notional": scaled_notional})

            open_position_rows = db.get_open_positions(conn, portfolio_row["id"])
            ctx = build_context(
                portfolio_row, open_position_rows, trades_today, current_prices, start_of_run_nav,
                symbol_metadata=symbol_metadata, peak_nav=peak_nav,
            )

            order_meta = symbol_metadata.get(order.symbol)
            check = evaluate_order(
                order, ctx, price, current_prices, risk_config,
                order_segment=order_meta.segment if order_meta is not None else None,
                order_cap_tier=order_meta.cap_tier if order_meta is not None else None,
                universe_symbols=set(symbol_metadata) or None,
                order_leveraged=getattr(order_meta, "leveraged", False) if order_meta is not None else False,
                average_daily_volume=(average_daily_volumes or {}).get(order.symbol),
            )
            risk_check_log.append({"symbol": order.symbol, "approved": check.approved, "reasons": check.reasons})

            # Markt-Phasen-Abgleich (rein dokumentarisch, kein Guardrail - siehe
            # src/market_phase.py): Claudes optionale cycle_position-Angabe gegen
            # die regelbasierte SMA/Vola-Klassifikation. Unabhaengig von
            # `check.approved` berechnet (auch eine abgelehnte Order dokumentiert
            # ihren Widerspruch), analog zu Vol-Skalierung/Konviktion oben.
            phase_classification = (market_phases or {}).get(order.symbol)
            if phase_classification is not None:
                contradiction = market_phase.check_cycle_position_against_market_phase(
                    order.symbol, order.cycle_position, phase_classification.phase
                )
                if contradiction is not None:
                    log.warning(
                        "Markt-Phasen-Widerspruch: %s - kein Veto, nur dokumentiert.", contradiction.detail
                    )

            if not check.approved:
                results.append(
                    ExecutedOrderResult(
                        order=order, approved=False, reasons=check.reasons,
                        volatility_scaling=scaling, conviction_scaling_factor=conviction_factor,
                        market_phase_contradiction=contradiction,
                    )
                )
                continue

            order_side = OrderSide(order.side)

            # Korrelations-Beobachtung (rein dokumentarisch, kein Guardrail) -
            # vor der Ausfuehrung, nur fuer BUY-Orders. `open_position_rows` ist
            # hier bereits der Stand NACH allen vorherigen Orders dieses Laufs
            # (siehe Neu-Abfrage oben), erfasst also auch bereits in diesem
            # Lauf eroeffnete Positionen.
            correlation_warnings: list[correlation.CorrelationWarning] = []
            if order_side == OrderSide.BUY and correlation_matrix is not None and not correlation_matrix.empty:
                existing_symbols = [r["symbol"] for r in open_position_rows]
                correlation_warnings = correlation.check_correlation_to_existing_positions(
                    order.symbol, existing_symbols, correlation_matrix
                )
                for w in correlation_warnings:
                    log.warning(
                        "Korrelations-Beobachtung: %s korreliert mit bestehender Position %s (%.2f) - "
                        "kein Veto, nur dokumentiert.",
                        w.candidate_symbol, w.existing_symbol, w.correlation,
                    )

            # v14 (2026-09-22, 17-Punkte-Audit Fund #2, HIGH - Kehrseite des
            # 8.9.-Vorfalls, siehe INCIDENT_2026-09-08.md): fuer die
            # Ausfuehrungsweg-Entscheidung (echt bei Alpaca vs. rein
            # simuliert) ist das in der Watchlist hinterlegte instrument_type
            # massgeblich, NICHT Claudes eigene Angabe im JSON - vorher
            # entschied allein order.instrument_type.value darueber, ohne
            # Abgleich gegen die Watchlist. Eine falsche/abweichende Angabe
            # (versehentlich oder nicht) fuer ein real gelistetes Symbol
            # haette DB-Zustand und echtes Alpaca-Konto dauerhaft und
            # unbemerkt auseinanderlaufen lassen. Fehlt der Symbol-Eintrag in
            # der Watchlist (sollte durch die LOW-3-Universums-Pruefung in
            # evaluate_order oben ohnehin nie erreicht werden), bleibt
            # Claudes Angabe der Fallback.
            watchlist_instrument_type = order_meta.instrument_type if order_meta is not None else None
            routing_instrument_type = watchlist_instrument_type or order.instrument_type.value
            if watchlist_instrument_type is not None and watchlist_instrument_type != order.instrument_type.value:
                log.warning(
                    "instrument_type-Abweichung fuer %s: Claude gab '%s' an, Watchlist sagt '%s' - "
                    "Watchlist-Wert entscheidet ueber den Ausfuehrungsweg (echt/simuliert), Abweichung "
                    "dokumentiert statt stillschweigend uebernommen (v14).",
                    order.symbol, order.instrument_type.value, watchlist_instrument_type,
                )

            fill, source = _route_fill(
                order_side,
                order.symbol,
                routing_instrument_type,
                order.quantity if order.quantity is not None else order.notional / price,
                order.order_type,
                order.limit_price,
                price,
                broker_client,
            )

            if fill.filled_qty <= 0:
                # Broker hat (noch) keine oder null Stück gemeldet - z.B. eine vom
                # Broker abgelehnte/stornierte Order oder ein Notional-Betrag, der
                # zum Antwortzeitpunkt noch nicht gefüllt war. `trades.quantity`
                # hat ein CHECK(quantity > 0); ein Insert würde die Pipeline zum
                # Absturz bringen. Stattdessen: Order überspringen und dokumentieren,
                # keine Cash-/Positions-/Trade-Mutation für diese Order.
                reason = (
                    f"Broker meldet gefüllte Menge {fill.filled_qty} (Status: {fill.status}) für "
                    f"{order.symbol} - Order wird übersprungen, kein Trade gebucht."
                )
                results.append(
                    ExecutedOrderResult(order=order, approved=False, reasons=[reason], market_phase_contradiction=contradiction)
                )
                risk_check_log.append({"symbol": order.symbol, "approved": False, "reasons": [reason]})
                continue

            # Neu einlesen der Portfolio-Zeile für aktuellen cash_balance vor jedem Trade.
            portfolio_row = db.get_portfolio(conn, portfolio_row["name"])
            results.append(
                ExecutedOrderResult(
                    order=order, approved=True, reasons=[], fill_price=fill.filled_price,
                    correlation_warnings=correlation_warnings,
                    volatility_scaling=scaling,
                    conviction_scaling_factor=conviction_factor,
                    market_phase_contradiction=contradiction,
                )
            )

            trades_today[order.symbol] = trades_today.get(order.symbol, 0) + 1

            # Trade + Positions-Update erfolgt sofort, Decision-Datensatz gebündelt danach.
            results[-1].trade_id = _apply_fill_to_db(
                conn, portfolio_row["id"], None, order, order_side, fill, source,
                transaction_cost_pct=risk_config.transaction_cost_pct_of_notional,
            )
        except Exception as exc:
            # Bugfix 2026-09-21 (17-Punkte-Audit Fund #3, real eingetreten im
            # Lauf um 2026-09-21 17:36 UTC: eine fraktionierte Short-Order
            # wurde vom Broker mit "fractional orders cannot be sold short"
            # abgelehnt - die Exception lief vorher ungefangen bis zum
            # Pipeline-Absturz durch. Vier zuvor in DERSELBEN Order-Liste
            # bereits erfolgreich ausgefuehrte Buy-Orders blieben dabei OHNE
            # decisions-Eintrag zurueck (ihre DB-Mutation war zu dem
            # Zeitpunkt schon committet, siehe _apply_fill_to_db oben) - das
            # untergrub die Idempotenz-Sperre (pipeline._already_decided_
            # today), da diese exakt so einen decisions-Eintrag braucht.
            # Dieses try/except faengt JEDE unerwartete Exception waehrend
            # der Verarbeitung EINER Order ab, statt die gesamte restliche
            # Order-Liste abstuerzen zu lassen: bereits committete Trades
            # vorheriger Orders bleiben unberuehrt, die Schleife verarbeitet
            # den Rest der Liste weiter, UND db.insert_decision unten wird
            # garantiert IMMER erreicht (haelt die Idempotenz-Sperre
            # zuverlaessig).
            log.exception(
                "Unerwarteter Fehler bei der Ausführung von Order %s (%s) - "
                "Order übersprungen, restliche Order-Liste wird fortgesetzt.",
                order.symbol, order.side,
            )
            reason = f"Unerwarteter Fehler bei Order-Ausführung ({type(exc).__name__}): {exc}"
            results.append(
                ExecutedOrderResult(
                    order=order, approved=False, reasons=[reason], execution_error=True,
                    volatility_scaling=scaling, conviction_scaling_factor=conviction_factor,
                    market_phase_contradiction=contradiction,
                )
            )
            risk_check_log.append({"symbol": order.symbol, "approved": False, "reasons": [reason]})

    decision_id = db.insert_decision(
        conn,
        portfolio_id=portfolio_row["id"],
        model=model,
        prompt=prompt,
        raw_response=raw_response,
        proposed_orders=[o.model_dump(mode="json") for o in orders],
        risk_check_result=risk_check_log,
        rationale=None,
        forced_action=False,
        approved=any(r.approved for r in results),
        executed=any(r.trade_id is not None for r in results),
    )

    # Trades nachträglich mit der Decision verknüpfen (siehe
    # db.link_trades_to_decision fuer die Begruendung dieser Reihenfolge).
    db.link_trades_to_decision(
        conn, [r.trade_id for r in results if r.trade_id is not None], decision_id
    )

    return results


def run_short_stop_loss_sweep(
    conn: sqlite3.Connection,
    portfolio_row: sqlite3.Row,
    current_prices: dict[str, float],
    risk_config: RiskConfig,
    broker_client,
) -> list[ForcedStopLossAction]:
    open_position_rows = db.get_open_positions(conn, portfolio_row["id"])
    positions = db.open_positions_as_risk_objects(open_position_rows)
    forced = evaluate_short_positions_for_stop_loss(positions, current_prices, risk_config.short_stop_loss_pct)
    if forced:
        execute_forced_stop_loss_actions(
            conn, portfolio_row, forced, broker_client, risk_config.transaction_cost_pct_of_notional
        )
    return forced
