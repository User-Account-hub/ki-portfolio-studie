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

import time
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import GetCalendarRequest, LimitOrderRequest, MarketOrderRequest

# Order states Alpaca will never transition out of on its own - no point
# polling further once one of these is reached.
_TERMINAL_UNFILLED_STATUSES = {
    "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended", "replaced",
}


@dataclass(frozen=True)
class FillResult:
    broker_order_id: str
    filled_price: float
    filled_qty: float
    status: str  # 'filled' | 'pending'


def get_trading_client(api_key: str, secret_key: str) -> TradingClient:
    return TradingClient(api_key, secret_key, paper=True)


def is_trading_day(client: TradingClient) -> bool:
    """True if the US equity market has a regular session today, False on a
    full closure (weekend or market holiday).

    Uses the calendar endpoint rather than clock.is_open, since is_open is
    also False outside today's session hours on an otherwise completely
    normal trading day (e.g. before 9:30 ET) - the calendar only returns an
    entry for days the market opens at all, so an empty result for today
    means the whole day is closed. "Today" is taken from Alpaca's own clock
    timestamp (US market time), not the caller's local/server date, to
    avoid any timezone ambiguity.
    """
    today = client.get_clock().timestamp.date()
    calendar = client.get_calendar(GetCalendarRequest(start=today, end=today))
    return len(calendar) > 0


def get_account_snapshot(client: TradingClient) -> dict:
    account = client.get_account()
    return {
        "cash": float(account.cash),
        "equity": float(account.equity),
        "portfolio_value": float(account.portfolio_value),
    }


def _poll_until_settled(client: TradingClient, order, poll_interval_seconds: float, max_attempts: int):
    """Alpaca fills orders asynchronously - submit_order() commonly returns
    before the fill is reflected on the returned order object, even for a
    plain market order during trading hours. Without this, the normal case
    (fills within well under a second) would be indistinguishable from a
    genuinely unfilled/rejected order.

    Polls get_order_by_id up to `max_attempts` times, `poll_interval_seconds`
    apart, stopping early once the order reaches 'filled' or any terminal
    unfilled status (rejected/canceled/...). If it's still open when we give
    up waiting, it gets actively canceled (see _cancel_if_still_open) -
    execution.py treats a still-zero filled_qty as a documented skip rather
    than a crash, and that's only safe if the order is actually dead on the
    broker side too; otherwise it can silently fill later (e.g. at next
    market open) with no matching trade in the local ledger.
    """
    for _ in range(max_attempts):
        if order.status.value == "filled" or order.status.value in _TERMINAL_UNFILLED_STATUSES:
            return order
        time.sleep(poll_interval_seconds)
        order = client.get_order_by_id(order.id)
    return _cancel_if_still_open(client, order)


def _cancel_if_still_open(client: TradingClient, order):
    """Called once polling gives up while the order is still live on the
    broker (typically: market closed, so it's sitting as accepted/new).
    Cancels it so it can't fill later outside the pipeline's knowledge - a
    real incident this fix addresses: 19 orders left unfilled by earlier
    pipeline runs (and therefore skipped/undocumented-as-executed locally)
    were found still open as DAY orders a day later, primed to fill at the
    next market open with zero record of it in db/portfolio.db.

    Best-effort: if the cancel or the status refetch fails, the original
    (still-open) order is returned rather than raising - a real order
    submission having already succeeded should never crash the pipeline
    over a follow-up cleanup call.
    """
    if order.status.value in _TERMINAL_UNFILLED_STATUSES:
        return order
    try:
        client.cancel_order_by_id(order.id)
        return client.get_order_by_id(order.id)
    except Exception:
        return order


def submit_equity_order(
    client: TradingClient,
    symbol: str,
    quantity: float,
    side: str,  # 'buy' | 'sell' | 'short' | 'cover'
    order_type: str = "market",
    limit_price: float | None = None,
    poll_interval_seconds: float = 0.5,
    max_poll_attempts: int = 10,
) -> FillResult:
    """Submits an order for an equity/ETF instrument.

    'short' and 'cover' are mapped onto Alpaca's plain buy/sell semantics:
    shorting is a sell order against no (or a negative) position, covering
    is a buy order that reduces/closes that negative position.

    Polls briefly for the actual fill after submitting (see
    _poll_until_settled, default: up to 10 x 0.5s = 5s) - a market order
    during trading hours typically settles within one or two polls; an
    order that's still unfilled after the timeout (e.g. market closed, or a
    genuine rejection) is reported back with filled_qty=0 exactly as
    before, for the caller to skip and document.
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
    order = _poll_until_settled(client, order, poll_interval_seconds, max_poll_attempts)

    filled_price = float(order.filled_avg_price) if order.filled_avg_price is not None else (limit_price or 0.0)
    filled_qty = float(order.filled_qty) if order.filled_qty is not None else 0.0
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
