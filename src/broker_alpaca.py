"""Alpaca Paper Trading execution for tradable instruments (equity/etf).

Structured products (leverage_certificate, mini_future, warrant) are NOT
tradable on Alpaca - they are Swiss/German retail derivatives, Alpaca only
covers US equities/ETFs/options/crypto. Orders for those instrument types
must go through `simulate_structured_product_fill` instead of this module's
`submit_equity_order` (see execution.py for the routing decision).

Shorting equities on Alpaca requires a margin-enabled paper account (that's
structurally how short selling works - you borrow the shares). This is
distinct from "margin trading" as restricted by risk_config.yaml, which
concerns leveraged BUYING POWER beyond 1x cash; that restriction is enforced
in risk_guardrails.check_no_margin before any order reaches this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest


@dataclass(frozen=True)
class FillResult:
    broker_order_id: str
    filled_price: float
    filled_qty: float
    status: str  # 'filled' | 'pending'


def get_trading_client(api_key: str, secret_key: str) -> TradingClient:
    return TradingClient(api_key, secret_key, paper=True)


def get_account_snapshot(client: TradingClient) -> dict:
    account = client.get_account()
    return {
        "cash": float(account.cash),
        "equity": float(account.equity),
        "portfolio_value": float(account.portfolio_value),
    }


def submit_equity_order(
    client: TradingClient,
    symbol: str,
    quantity: float,
    side: str,  # 'buy' | 'sell' | 'short' | 'cover'
    order_type: str = "market",
    limit_price: float | None = None,
) -> FillResult:
    """Submits an order for an equity/ETF instrument.

    'short' and 'cover' are mapped onto Alpaca's plain buy/sell semantics:
    shorting is a sell order against no (or a negative) position, covering
    is a buy order that reduces/closes that negative position.
    """
    alpaca_side = AlpacaOrderSide.SELL if side in ("sell", "short") else AlpacaOrderSide.BUY

    if order_type == "market":
        request = MarketOrderRequest(
            symbol=symbol, qty=quantity, side=alpaca_side, time_in_force=TimeInForce.DAY
        )
    else:
        if limit_price is None:
            raise ValueError("limit_price ist bei order_type='limit' erforderlich.")
        request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )

    order = client.submit_order(request)
    filled_price = float(order.filled_avg_price) if order.filled_avg_price else (limit_price or 0.0)
    filled_qty = float(order.filled_qty) if order.filled_qty else quantity
    status = "filled" if order.status.value == "filled" else "pending"
    return FillResult(
        broker_order_id=str(order.id),
        filled_price=filled_price,
        filled_qty=filled_qty,
        status=status,
    )


def simulate_structured_product_fill(quantity: float, reference_price: float) -> FillResult:
    """Books a structured-product order without routing it to a broker.

    reference_price should be the last known price for the product itself
    if available, otherwise a documented proxy (e.g. underlying price) -
    see execution.py for how the price is resolved. This is a bookkeeping
    simulation only; no real order is placed anywhere.
    """
    return FillResult(
        broker_order_id="",
        filled_price=reference_price,
        filled_qty=quantity,
        status="filled",
    )
