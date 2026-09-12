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
    # Thesis-Anhang-A-Taxonomie, vom Caller aus der Watchlist angereichert
    # (siehe db.open_positions_as_risk_objects) - None fuer strukturierte
    # Produkte und alles ausserhalb des Anhang-A-Universums.
    segment: str | None = None
    cap_tier: str | None = None
    # Wirtschaftlich gehebelt, aber nicht als strukturiertes Produkt getaggt
    # (z.B. gehebelte ETFs NVDL/TSDD) - vom Caller aus der Watchlist angereichert
    # (siehe db.open_positions_as_risk_objects). Bringt diese Positionen in den
    # Geltungsbereich der Hebel-spezifischen Guardrails.
    leveraged: bool = False


@dataclass(frozen=True)
class PortfolioContext:
    """Snapshot of portfolio state needed to evaluate one proposed order."""

    nav: float
    cash: float
    positions: list[OpenPosition] = field(default_factory=list)
    trades_today_by_symbol: dict[str, int] = field(default_factory=dict)
    start_of_run_nav: float | None = None  # NAV vor dem aktuellen Pipeline-Lauf
    # Echter historischer NAV-Hoechststand (MAX ueber alle bisherigen Laeufe
    # aus der nav_history-Tabelle, siehe db.get_peak_nav), fuer
    # check_circuit_breaker. None nur, wenn ein Portfolio noch keinen
    # nav_history-Eintrag hat (z.B. in Tests ohne DB).
    peak_nav: float | None = None

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

    def leveraged_notional(self, current_prices: dict[str, float]) -> float:
        """Total notional of all leverage-controlled positions (structured
        products + watchlist-flagged leveraged instruments like NVDL/TSDD)."""
        total = 0.0
        for p in self.positions:
            if _is_leverage_controlled(p.instrument_type, p.leveraged):
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


def _is_leverage_controlled(instrument_type: str, leveraged: bool) -> bool:
    """True for instruments the leverage-specific guardrails (structured-products
    cap + drawdown circuit-breaker) must cover: the tagged structured products
    AND anything flagged `leveraged` in the watchlist (e.g. the 2x ETFs NVDL/TSDD,
    whose instrument_type is "etf" but which carry real economic leverage)."""
    return instrument_type in STRUCTURED_INSTRUMENT_TYPES or leveraged


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
    order: ProposedOrder, ctx: PortfolioContext, price: float, max_pct_of_nav: float
) -> RiskCheckResult:
    """Kap. 6.8: max. Trade-Notional als Anteil des NAV zu Laufbeginn
    (ctx.start_of_run_nav), NICHT des aktuellen Cash. Cash schrumpft mit
    jedem ausgeführten Buy innerhalb desselben Laufs - würde man dagegen
    prüfen, würde sich das Limit von Order zu Order verschärfen, obwohl die
    Konfiguration eine feste Grösse vorsieht. Fällt auf ctx.nav zurück, falls
    start_of_run_nav nicht gesetzt ist (z.B. in Tests ohne vollen Kontext).
    """
    notional = order_notional(order, price)
    reference_nav = ctx.start_of_run_nav if ctx.start_of_run_nav is not None else ctx.nav
    limit = reference_nav * max_pct_of_nav
    if notional > limit:
        return RiskCheckResult.reject(
            f"Trade-Notional {notional:.2f} übersteigt Limit von "
            f"{max_pct_of_nav:.0%} des NAV zu Laufbeginn ({limit:.2f})."
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
    order_leveraged: bool = False,
) -> RiskCheckResult:
    """Aggregate cap on leveraged exposure. Covers tagged structured products
    AND watchlist-flagged leveraged instruments (e.g. the 2x ETFs NVDL/TSDD) -
    keyed off economic leverage via `_is_leverage_controlled`, not the
    instrument_type label alone, so a 2x ETF cannot sidestep the cap by being
    typed "etf"."""
    if not _is_leverage_controlled(order.instrument_type.value, order_leveraged):
        return RiskCheckResult.ok()
    if order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()

    current_total = ctx.leveraged_notional(current_prices)
    added = order_notional(order, price)
    limit = ctx.nav * max_pct_of_nav
    if current_total + added > limit:
        return RiskCheckResult.reject(
            f"Gehebelte/strukturierte Produkte gesamt ({current_total + added:.2f}) würden Limit von "
            f"{max_pct_of_nav:.0%} des NAV ({limit:.2f}) überschreiten."
        )
    return RiskCheckResult.ok()


def _is_micro_cap(cap_tier: str | None) -> bool:
    """Matches "Micro-Cap" and mixed tiers like "Micro/Small-Cap" from Anhang A."""
    return bool(cap_tier) and "micro" in cap_tier.lower()


def check_segment_weight(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    current_prices: dict[str, float],
    order_segment: str | None,
    max_pct_of_nav: float,
) -> RiskCheckResult:
    """Kap. 6.8: ein einzelnes Anhang-A-Segment darf nach Ausführung des
    Trades nicht mehr als `max_pct_of_nav` des NAV ausmachen. Orders ohne
    bekanntes Segment (z.B. strukturierte Produkte) werden nicht geprüft."""
    if order_segment is None or order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()
    existing = sum(
        p.quantity * current_prices.get(p.symbol, p.avg_entry_price)
        for p in ctx.positions
        if p.segment == order_segment
    )
    added = order_notional(order, price)
    limit = ctx.nav * max_pct_of_nav
    if existing + added > limit:
        return RiskCheckResult.reject(
            f"Segment '{order_segment}' läge nach dieser Order ({existing + added:.2f}) "
            f"über dem Limit von {max_pct_of_nav:.0%} des NAV ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_correlated_segment_exposure(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    current_prices: dict[str, float],
    order_segment: str | None,
    correlated_segments: set[str],
    max_pct_of_nav: float,
) -> RiskCheckResult:
    """Kap. 6.8: kombinierte Exposure der korrelierten Segmente (Default:
    Krypto-Mining + Digital Assets & Krypto-Oekosystem, siehe risk_config.yaml)
    darf `max_pct_of_nav` des NAV nicht überschreiten."""
    if not correlated_segments or order_segment not in correlated_segments:
        return RiskCheckResult.ok()
    if order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()
    existing = sum(
        p.quantity * current_prices.get(p.symbol, p.avg_entry_price)
        for p in ctx.positions
        if p.segment in correlated_segments
    )
    added = order_notional(order, price)
    limit = ctx.nav * max_pct_of_nav
    if existing + added > limit:
        return RiskCheckResult.reject(
            f"Korrelierte Segmente {sorted(correlated_segments)} lägen nach dieser Order "
            f"({existing + added:.2f}) über dem Limit von {max_pct_of_nav:.0%} des NAV ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_micro_cap_exposure(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    current_prices: dict[str, float],
    order_cap_tier: str | None,
    max_pct_of_nav: float,
) -> RiskCheckResult:
    """Kap. 6.8: Micro-Cap-Sublimit über alle Micro-Cap-Titel (CapTier laut
    Anhang A) hinweg, unabhängig vom Segment."""
    if order.side not in (OrderSide.BUY, OrderSide.SHORT) or not _is_micro_cap(order_cap_tier):
        return RiskCheckResult.ok()
    existing = sum(
        p.quantity * current_prices.get(p.symbol, p.avg_entry_price)
        for p in ctx.positions
        if _is_micro_cap(p.cap_tier)
    )
    added = order_notional(order, price)
    limit = ctx.nav * max_pct_of_nav
    if existing + added > limit:
        return RiskCheckResult.reject(
            f"Micro-Cap-Exposure läge nach dieser Order ({existing + added:.2f}) über dem "
            f"Sublimit von {max_pct_of_nav:.0%} des NAV ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_top3_concentration(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    current_prices: dict[str, float],
    max_pct_of_nav: float,
) -> RiskCheckResult:
    """Kap. 6.8: die drei grössten Einzelpositionen (nach Marktwert, je
    Symbol über Long/Short summiert) dürfen zusammen `max_pct_of_nav` des
    NAV nicht überschreiten."""
    if order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()
    values: dict[str, float] = {}
    for p in ctx.positions:
        values[p.symbol] = values.get(p.symbol, 0.0) + p.quantity * current_prices.get(p.symbol, p.avg_entry_price)
    delta_qty = order.quantity if order.quantity is not None else order_notional(order, price) / price
    values[order.symbol] = values.get(order.symbol, 0.0) + delta_qty * price
    top3_total = sum(sorted(values.values(), reverse=True)[:3])
    limit = ctx.nav * max_pct_of_nav
    if top3_total > limit:
        return RiskCheckResult.reject(
            f"Top-3-Konzentration läge nach dieser Order ({top3_total:.2f}) über dem Limit "
            f"von {max_pct_of_nav:.0%} des NAV ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_min_cash_quota(
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    min_pct_of_nav: float,
) -> RiskCheckResult:
    """Kap. 6.8: Mindest-Cash-Quote - Orders, die Cash verbrauchen (buy/cover),
    dürfen die Cash-Quote nicht unter `min_pct_of_nav` des NAV drücken."""
    if order.side not in (OrderSide.BUY, OrderSide.COVER):
        return RiskCheckResult.ok()
    notional = order_notional(order, price)
    resulting_cash = ctx.cash - notional
    limit = ctx.nav * min_pct_of_nav
    if resulting_cash < limit:
        return RiskCheckResult.reject(
            f"Resultierende Cash-Quote ({resulting_cash:.2f}) läge unter dem Minimum von "
            f"{min_pct_of_nav:.0%} des NAV ({limit:.2f})."
        )
    return RiskCheckResult.ok()


def check_circuit_breaker(
    order: ProposedOrder,
    ctx: PortfolioContext,
    drawdown_pct: float,
    order_leveraged: bool = False,
) -> RiskCheckResult:
    """Kap. 6.8: Portfolio-Circuit-Breaker. Sobald der NAV seit seinem
    bisherigen Höchststand (ctx.peak_nav) um mehr als abs(drawdown_pct)
    gefallen ist, werden keine neuen/aufstockenden Hebelpositionen
    (strukturierte Produkte) mehr zugelassen. Bestehende Positionen können
    weiterhin reduziert/geschlossen werden; reguläre Aktien-/ETF-Orders sind
    nicht betroffen (dafür gilt weiterhin nur der daily_loss_stop_pct). Eine
    "Hebelposition" umfasst hier strukturierte Produkte UND watchlist-markierte
    gehebelte Instrumente (z.B. NVDL/TSDD), keyed off _is_leverage_controlled."""
    if not _is_leverage_controlled(order.instrument_type.value, order_leveraged):
        return RiskCheckResult.ok()
    if order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()
    if not ctx.peak_nav:
        return RiskCheckResult.ok()
    current_drawdown = (ctx.nav - ctx.peak_nav) / ctx.peak_nav
    if current_drawdown <= drawdown_pct:
        return RiskCheckResult.reject(
            f"Circuit-Breaker aktiv: Drawdown {current_drawdown:.2%} seit Höchststand "
            f"({ctx.peak_nav:.2f}) unterschreitet Schwelle {drawdown_pct:.2%} - keine neuen "
            "Hebelpositionen erlaubt."
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
    order_segment: str | None = None,
    order_cap_tier: str | None = None,
    universe_symbols: set[str] | None = None,
    order_leveraged: bool = False,
) -> RiskCheckResult:
    """Runs all applicable guardrail checks for a single proposed order.

    `config` is a src.config.RiskConfig (typed loosely here to keep this
    module importable without a hard dependency on YAML loading in tests).

    `order_segment`/`order_cap_tier` are the order's symbol's Anhang-A
    taxonomy values (resolved by the caller from the watchlist - see
    execution.py); None for symbols outside that taxonomy (e.g. structured
    products), in which case the corresponding Kap.-6.8 checks no-op.

    `universe_symbols`, when provided, is the set of symbols the model is
    allowed to trade (the watchlist). Any order for a symbol outside it is
    rejected outright. The system prompt already instructs Claude to stay
    in-universe, but - consistent with the defense-in-depth stance of this
    module - execution must not rely on the model honouring that. When None
    (e.g. in unit tests that exercise a single check), the allowlist gate is
    skipped.
    """
    if universe_symbols and order.symbol not in universe_symbols:
        return RiskCheckResult.reject(
            f"Symbol '{order.symbol}' ist nicht im Anlage-Universum - Order abgelehnt."
        )

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
        check_trade_notional(order, ctx, price, config.max_trade_notional_pct_of_nav),
        check_position_size(order, ctx, price, config.max_position_size_pct_of_portfolio),
        check_structured_products_cap(
            order, ctx, price, current_prices, config.structured_products_max_notional_pct_of_nav,
            order_leveraged=order_leveraged,
        ),
        check_segment_weight(order, ctx, price, current_prices, order_segment, config.max_segment_weight_pct_of_nav),
        check_correlated_segment_exposure(
            order,
            ctx,
            price,
            current_prices,
            order_segment,
            set(config.correlated_crypto_mining_segments),
            config.max_correlated_crypto_mining_pct_of_nav,
        ),
        check_micro_cap_exposure(order, ctx, price, current_prices, order_cap_tier, config.max_micro_cap_pct_of_nav),
        check_top3_concentration(order, ctx, price, current_prices, config.max_top3_concentration_pct_of_nav),
        check_min_cash_quota(order, ctx, price, config.min_cash_pct_of_nav),
        check_circuit_breaker(order, ctx, config.circuit_breaker_drawdown_pct, order_leveraged=order_leveraged),
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
