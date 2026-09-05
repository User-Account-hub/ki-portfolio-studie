"""Pre-trade risk guardrails.

This module is intentionally pure / I/O-free (no DB, no network) so it can
be unit-tested in isolation and reasoned about independently of the rest of
the pipeline. It is the single place where trading limits are enforced -
the prompt sent to Claude mentions the same limits, but execution never
relies on the model honouring them (defense in depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.order_schema import STRUCTURED_INSTRUMENT_TYPES, OrderSide, ProposedOrder


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    instrument_type: str
    side: str  # 'long' | 'short'
    quantity: float
    avg_entry_price: float


@dataclass(frozen=True)
class PortfolioContext:
    """Snapshot of portfolio state needed to evaluate one proposed order."""

    nav: float
    cash: float
    positions: list[OpenPosition] = field(default_factory=list)
    trades_today_by_symbol: dict[str, int] = field(default_factory=dict)
    start_of_run_nav: float | None = None  # NAV vor dem aktuellen Pipeline-Lauf

    def position_for(self, symbol: str) -> OpenPosition | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def structured_products_notional(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for p in self.positions:
            if p.instrument_type in STRUCTURED_INSTRUMENT_TYPES:
                price = current_prices.get(p.symbol, p.avg_entry_price)
                total += p.quantity * price
        return total


@dataclass(frozen=True)
class RiskCheckResult:
    approved: bool
    reasons: list[str] = field(default_factory=list)

    @staticmethod
    def ok() -> "RiskCheckResult":
        return RiskCheckResult(approved=True, reasons=[])

    @staticmethod
    def reject(reason: str) -> "RiskCheckResult":
        return RiskCheckResult(approved=False, reasons=[reason])

    def merged_with(self, other: "RiskCheckResult") -> "RiskCheckResult":
        if self.approved and other.approved:
            return RiskCheckResult.ok()
        return RiskCheckResult(approved=False, reasons=[*self.reasons, *other.reasons])


@dataclass(frozen=True)
class ForcedStopLossAction:
    """A mandatory position close triggered by the short stop-loss guardrail.

    Must be executed regardless of the daily-loss-stop gate and regardless
    of Claude's proposal, and must be documented (closure_notes on the
    position + a forced_action decision row) per compliance requirement.
    """

    symbol: str
    instrument_type: str
    quantity: float
    entry_price: float
    current_price: float
    loss_pct: float
    documentation: str


def compute_nav(cash: float, positions: list[OpenPosition], current_prices: dict[str, float]) -> float:
    """Mark-to-market NAV = cash + long market value - short market value.

    Consistent with the cash-flow convention used in execution.py (opening a
    short credits cash with the sale proceeds, covering debits it), so NAV
    does not jump at trade time - only P&L from price movement affects it.
    """
    value = cash
    for p in positions:
        price = current_prices.get(p.symbol, p.avg_entry_price)
        market_value = p.quantity * price
        value += market_value if p.side == "long" else -market_value
    return value


def order_notional(order: ProposedOrder, price: float) -> float:
    """Resolves an order's notional value given quantity-or-notional input."""
    if order.notional is not None:
        return order.notional
    assert order.quantity is not None
    return order.quantity * price


def check_max_trades_per_symbol(order: ProposedOrder, ctx: PortfolioContext, max_per_day: int) -> RiskCheckResult:
    count = ctx.trades_today_by_symbol.get(order.symbol, 0)
    if count >= max_per_day:
        return RiskCheckResult.reject(
            f"Max. {max_per_day} Trade(s)/Tag für {order.symbol} bereits erreicht (heute: {count})."
        )
    return RiskCheckResult.ok()


def check_no_margin(order: ProposedOrder, ctx: PortfolioContext, price: float, allow_margin: bool) -> RiskCheckResult:
    """Buys and shorts must be fully covered by available cash (no leveraged buying power)."""
    if allow_margin:
        return RiskCheckResult.ok()
    if order.side in (OrderSide.BUY, OrderSide.SHORT):
        notional = order_notional(order, price)
        if notional > ctx.cash:
            return RiskCheckResult.reject(
                f"Kein Margin-Trading erlaubt: Notional {notional:.2f} übersteigt "
                f"verfügbares Cash {ctx.cash:.2f}."
            )
    return RiskCheckResult.ok()


def check_trade_notional(
    order: ProposedOrder, ctx: PortfolioContext, price: float, max_pct_of_cash: float
) -> RiskCheckResult:
    notional = order_notional(order, price)
    limit = ctx.cash * max_pct_of_cash
    if notional > limit:
        return RiskCheckResult.reject(
            f"Trade-Notional {notional:.2f} übersteigt Limit von "
            f"{max_pct_of_cash:.0%} des Cash ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_position_size(
    order: ProposedOrder, ctx: PortfolioContext, price: float, max_pct_of_portfolio: float
) -> RiskCheckResult:
    """Ensures the resulting position (existing +/- this order) stays within the NAV limit."""
    existing = ctx.position_for(order.symbol)
    existing_qty = existing.quantity if existing else 0.0
    delta_qty = order.quantity if order.quantity is not None else order_notional(order, price) / price

    if order.side in (OrderSide.BUY, OrderSide.SHORT):
        resulting_qty = existing_qty + delta_qty
    else:  # SELL / COVER reduce exposure, never breach the limit
        return RiskCheckResult.ok()

    resulting_notional = resulting_qty * price
    limit = ctx.nav * max_pct_of_portfolio
    if resulting_notional > limit:
        return RiskCheckResult.reject(
            f"Resultierende Position in {order.symbol} ({resulting_notional:.2f}) "
            f"übersteigt Limit von {max_pct_of_portfolio:.0%} des NAV ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_structured_products_cap(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    current_prices: dict[str, float],
    max_pct_of_nav: float,
) -> RiskCheckResult:
    if order.instrument_type.value not in STRUCTURED_INSTRUMENT_TYPES:
        return RiskCheckResult.ok()
    if order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()

    current_total = ctx.structured_products_notional(current_prices)
    added = order_notional(order, price)
    limit = ctx.nav * max_pct_of_nav
    if current_total + added > limit:
        return RiskCheckResult.reject(
            f"Strukturierte Produkte gesamt ({current_total + added:.2f}) würden Limit von "
            f"{max_pct_of_nav:.0%} des NAV ({limit:.2f}) überschreiten."
        )
    return RiskCheckResult.ok()


def check_daily_loss_stop(ctx: PortfolioContext, daily_loss_stop_pct: float) -> RiskCheckResult:
    """Portfolio-wide gate: blocks ALL new/increasing orders once the loss
    threshold since the start of the current run is breached. Does not
    apply to forced stop-loss closes (those are evaluated separately and
    executed regardless)."""
    if ctx.start_of_run_nav is None or ctx.start_of_run_nav == 0:
        return RiskCheckResult.ok()
    change_pct = (ctx.nav - ctx.start_of_run_nav) / ctx.start_of_run_nav
    if change_pct <= daily_loss_stop_pct:
        return RiskCheckResult.reject(
            f"Verlust-Stop ausgelöst: {change_pct:.2%} seit Lauf-Start "
            f"(Limit {daily_loss_stop_pct:.2%}). Keine neuen Trades in diesem Lauf."
        )
    return RiskCheckResult.ok()


def evaluate_order(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    current_prices: dict[str, float],
    config,
) -> RiskCheckResult:
    """Runs all applicable guardrail checks for a single proposed order.

    `config` is a src.config.RiskConfig (typed loosely here to keep this
    module importable without a hard dependency on YAML loading in tests).
    """
    if order.side == OrderSide.SHORT and not config.allow_short:
        return RiskCheckResult.reject("Short-Positionen sind laut Risk-Config nicht erlaubt.")

    if order.instrument_type.value in STRUCTURED_INSTRUMENT_TYPES and not config.allow_structured_products:
        return RiskCheckResult.reject("Strukturierte Produkte sind laut Risk-Config nicht erlaubt.")

    checks = [
        check_daily_loss_stop(ctx, config.daily_loss_stop_pct)
        if order.side in (OrderSide.BUY, OrderSide.SHORT)
        else RiskCheckResult.ok(),
        check_max_trades_per_symbol(order, ctx, config.max_trades_per_symbol_per_day),
        check_no_margin(order, ctx, price, config.allow_margin),
        check_trade_notional(order, ctx, price, config.max_trade_notional_pct_of_cash),
        check_position_size(order, ctx, price, config.max_position_size_pct_of_portfolio),
        check_structured_products_cap(
            order, ctx, price, current_prices, config.structured_products_max_notional_pct_of_nav
        ),
    ]
    result = RiskCheckResult.ok()
    for c in checks:
        result = result.merged_with(c)
    return result


def evaluate_short_positions_for_stop_loss(
    positions: list[OpenPosition],
    current_prices: dict[str, float],
    short_stop_loss_pct: float,
) -> list[ForcedStopLossAction]:
    """Mandatory sweep run before every pipeline decision cycle.

    For every open short position, force a close if the price has moved
    against the position by more than abs(short_stop_loss_pct). This is
    independent of the daily-loss-stop gate and independent of what Claude
    proposes - it is a hard, documentation-required safety mechanism.
    """
    threshold = abs(short_stop_loss_pct)
    forced: list[ForcedStopLossAction] = []
    for p in positions:
        if p.side != "short":
            continue
        price = current_prices.get(p.symbol)
        if price is None:
            continue
        loss_pct = (price - p.avg_entry_price) / p.avg_entry_price  # positive = loss on a short
        if loss_pct >= threshold:
            forced.append(
                ForcedStopLossAction(
                    symbol=p.symbol,
                    instrument_type=p.instrument_type,
                    quantity=p.quantity,
                    entry_price=p.avg_entry_price,
                    current_price=price,
                    loss_pct=loss_pct,
                    documentation=(
                        f"Automatischer Short-Stop-Loss ausgelöst für {p.symbol}: "
                        f"Kurs {price:.2f} liegt {loss_pct:.2%} über Einstandskurs "
                        f"{p.avg_entry_price:.2f} (Schwelle {threshold:.2%})."
                    ),
                )
            )
    return forced
