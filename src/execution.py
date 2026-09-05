"""Orchestrates: pre-trade risk checks -> broker/simulated execution -> DB updates.

Cash-flow convention (documented once here, applied consistently):
  buy   -> cash -= notional   (long position opened/increased)
  sell  -> cash += notional   (long position reduced/closed)
  short -> cash += notional   (short sale proceeds; short position opened/increased)
  cover -> cash -= notional   (buy-back cost; short position reduced/closed)
This matches real cash economics and keeps NAV (see risk_guardrails.compute_nav)
unchanged at the instant a trade is opened - only price movement thereafter
produces P&L.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src import broker_alpaca, db
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


@dataclass
class ExecutedOrderResult:
    order: ProposedOrder
    approved: bool
    reasons: list[str]
    trade_id: int | None = None
    fill_price: float | None = None


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
) -> PortfolioContext:
    positions = db.open_positions_as_risk_objects(open_position_rows)
    nav = compute_nav(portfolio_row["cash_balance"], positions, current_prices)
    return PortfolioContext(
        nav=nav,
        cash=portfolio_row["cash_balance"],
        positions=positions,
        trades_today_by_symbol=trades_today_by_symbol,
        start_of_run_nav=start_of_run_nav,
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
) -> int:
    position_side = POSITION_SIDE[order_side]
    quantity = fill.filled_qty
    notional = quantity * fill.filled_price

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

    db.update_cash_balance(conn, portfolio_id, CASH_DIRECTION[order_side] * notional)

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
    )


def execute_forced_stop_loss_actions(
    conn: sqlite3.Connection,
    portfolio_row: sqlite3.Row,
    forced_actions: list[ForcedStopLossAction],
    broker_client,
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
        order = ProposedOrder(
            symbol=action.symbol,
            instrument_type=action.instrument_type,
            side="cover",
            quantity=action.quantity,
            rationale=action.documentation,
        )
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
            db.update_cash_balance(conn, portfolio_row["id"], CASH_DIRECTION[OrderSide.COVER] * fill.filled_qty * fill.filled_price)
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
) -> list[ExecutedOrderResult]:
    results: list[ExecutedOrderResult] = []
    risk_check_log = []

    trades_today = db.count_trades_today_by_symbol(conn, portfolio_row["id"])

    for order in orders:
        try:
            price = resolve_price(order.symbol, order.underlying_symbol, current_prices)
        except ValueError as exc:
            results.append(ExecutedOrderResult(order=order, approved=False, reasons=[str(exc)]))
            risk_check_log.append({"symbol": order.symbol, "approved": False, "reasons": [str(exc)]})
            continue

        open_position_rows = db.get_open_positions(conn, portfolio_row["id"])
        ctx = build_context(portfolio_row, open_position_rows, trades_today, current_prices, start_of_run_nav)

        check = evaluate_order(order, ctx, price, current_prices, risk_config)
        risk_check_log.append({"symbol": order.symbol, "approved": check.approved, "reasons": check.reasons})

        if not check.approved:
            results.append(ExecutedOrderResult(order=order, approved=False, reasons=check.reasons))
            continue

        order_side = OrderSide(order.side)
        fill, source = _route_fill(
            order_side,
            order.symbol,
            order.instrument_type.value,
            order.quantity if order.quantity is not None else order.notional / price,
            order.order_type,
            order.limit_price,
            price,
            broker_client,
        )
        # Neu einlesen der Portfolio-Zeile für aktuellen cash_balance vor jedem Trade.
        portfolio_row = db.get_portfolio(conn, portfolio_row["name"])
        results.append(ExecutedOrderResult(order=order, approved=True, reasons=[], fill_price=fill.filled_price))

        trades_today[order.symbol] = trades_today.get(order.symbol, 0) + 1

        # Trade + Positions-Update erfolgt sofort, Decision-Datensatz gebündelt danach.
        results[-1].trade_id = _apply_fill_to_db(conn, portfolio_row["id"], None, order, order_side, fill, source)

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

    # Trades nachträglich mit der Decision verknüpfen.
    for r in results:
        if r.trade_id is not None:
            conn.execute("UPDATE trades SET decision_id = ? WHERE id = ?", (decision_id, r.trade_id))
    conn.commit()

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
        execute_forced_stop_loss_actions(conn, portfolio_row, forced, broker_client)
    return forced
