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
    # Individueller, bei Order-Aufgabe von Claude genannter (oder von
    # risk_guardrails/execution auf Basis von short_stop_loss_pct gesetzter)
    # Stop-Loss-Kurs, aus positions.stop_loss_price (siehe
    # db.open_positions_as_risk_objects). None, wenn fuer diese Position nie
    # einer gesetzt wurde - dann greift in
    # evaluate_short_positions_for_stop_loss der globale Fallback.
    stop_loss_price: float | None = None


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
    order: ProposedOrder,
    ctx: PortfolioContext,
    price: float,
    max_pct_of_portfolio: float,
    drawdown_size_factor: float = 1.0,
) -> RiskCheckResult:
    """Ensures the resulting position (existing +/- this order) stays within
    the NAV limit.

    `drawdown_size_factor` (Kap. 6.8 Erweiterung, siehe
    drawdown_position_size_factor) reduziert `max_pct_of_portfolio` weiter -
    Default 1.0 (keine Reduktion) hält bestehende Aufrufer/Tests ohne
    Drawdown-Kontext unverändert kompatibel.
    """
    existing = ctx.position_for(order.symbol)
    existing_qty = existing.quantity if existing else 0.0
    delta_qty = order.quantity if order.quantity is not None else order_notional(order, price) / price

    if order.side in (OrderSide.BUY, OrderSide.SHORT):
        resulting_qty = existing_qty + delta_qty
    else:  # SELL / COVER reduce exposure, never breach the limit
        return RiskCheckResult.ok()

    resulting_notional = resulting_qty * price
    limit = ctx.nav * max_pct_of_portfolio * drawdown_size_factor
    if resulting_notional > limit:
        drawdown_note = (
            f" (Positionslimit drawdown-bedingt auf {drawdown_size_factor:.0%} reduziert)"
            if drawdown_size_factor < 1.0
            else ""
        )
        return RiskCheckResult.reject(
            f"Resultierende Position in {order.symbol} ({resulting_notional:.2f}) "
            f"übersteigt Limit von {max_pct_of_portfolio:.0%} des NAV ({limit:.2f}){drawdown_note}."
        )
    return RiskCheckResult.ok()


def drawdown_position_size_factor(
    ctx: PortfolioContext,
    tier1_drawdown_pct: float,
    tier1_factor: float,
    tier2_drawdown_pct: float,
    tier2_factor: float,
    full_stop_drawdown_pct: float,
) -> float:
    """Kap. 6.8 Erweiterung (2026-09-19): abgestufter Drawdown-Schutz für die
    maximal erlaubte NEUE Einzelpositionsgrösse (check_position_size),
    zusätzlich zum bestehenden harten Circuit-Breaker (check_circuit_breaker,
    ausschliesslich für Hebelpositionen). Je tiefer der Drawdown seit dem
    historischen NAV-Höchststand (ctx.peak_nav - dieselbe bereits vorhandene
    historical_peak_nav-Infrastruktur wie check_circuit_breaker, siehe
    db.get_peak_nav), desto kleiner der zurückgegebene Multiplikator auf
    `max_position_size_pct_of_portfolio`:

      Drawdown <= tier1_drawdown_pct (Default -10%) -> tier1_factor (0.75)
      Drawdown <= tier2_drawdown_pct (Default -15%) -> tier2_factor (0.50)
      Drawdown <= full_stop_drawdown_pct (Default -25%, = derselbe Wert wie
        circuit_breaker_drawdown_pct) -> 0.0 (kompletter Stop neuer/
        aufstockender Positionen - "wie bisher", siehe check_circuit_breaker)
      sonst -> 1.0 (keine Reduktion)

    Der TIEFSTE erreichte Tier gewinnt (Funktion prüft von streng nach
    locker). Erwartet `full_stop_drawdown_pct <= tier2_drawdown_pct <=
    tier1_drawdown_pct <= 0`; die Caller-seitige Config (risk_config.yaml)
    ist dafür verantwortlich, keine widersprüchliche Reihenfolge zu pflegen -
    diese Funktion selbst validiert das nicht (bewusst analog zu den übrigen
    reinen risk_guardrails-Funktionen, die ihrer Config vertrauen).

    Ohne bekannten Höchststand (ctx.peak_nav None/0, z.B. in Tests ohne
    vollen Kontext) wird 1.0 (keine Reduktion) angenommen - konsistent mit
    check_circuit_breaker's Verhalten im selben Fall.

    AUSDRÜCKLICH NICHT für Hebelpositionen/strukturierte Produkte anzuwenden
    (siehe evaluate_order's Aufrufstelle, die für sie unverändert 1.0
    übergibt) - deren Verhalten bleibt exklusiv durch check_circuit_breaker
    geregelt (harter Stop einzig bei full_stop_drawdown_pct, keine
    Zwischenstufen). Würde diese Funktion auch auf sie angewendet, gälten
    für dieselben Positionen zwei unterschiedliche Drawdown-Regimes
    gleichzeitig und ihr bisheriges Verhalten würde sich ändern.
    """
    if not ctx.peak_nav:
        return 1.0
    current_drawdown = (ctx.nav - ctx.peak_nav) / ctx.peak_nav
    if current_drawdown <= full_stop_drawdown_pct:
        return 0.0
    if current_drawdown <= tier2_drawdown_pct:
        return tier2_factor
    if current_drawdown <= tier1_drawdown_pct:
        return tier1_factor
    return 1.0


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


def check_liquidity_limit(
    order: ProposedOrder,
    price: float,
    average_daily_volume: float | None,
    max_pct_of_avg_daily_volume: float,
) -> RiskCheckResult:
    """Kap. 6.13: eine Order darf nicht mehr als `max_pct_of_avg_daily_volume`
    des Tagesvolumens des Titels ausmachen - schützt vor einem Ausführungs-/
    Slippage-Risiko, das die NAV-basierten Positionsgrössen-Limiten oben
    NICHT abdecken: eine vom NAV her erlaubte Positionsgrösse (z.B. 10% eines
    grossen Portfolios) kann das tatsächliche Handelsvolumen eines dünn
    gehandelten Micro-/Small-Cap-Titels trotzdem weit übersteigen.

    `average_daily_volume` ist MarketSnapshot.volume (data_fetch.py) - das
    zuletzt bekannte EINZELTAGES-Volumen. Das ist KEIN echter mehrtägiger
    gleitender Durchschnitt (yfinance liefert hier keinen zusätzlichen Abruf
    dafür) - eine dokumentierte Vereinfachung, analog zu den anderen
    Näherungen in diesem Projekt (siehe metrics.py-Modul-Docstring), für
    dieses grobe Ausführungsrisiko-Signal aber ausreichend.

    Nur für positionsaufbauende Seiten (buy/short) relevant - Sell/Cover
    reduzieren Exposure und werden wie bei den anderen Guardrails oben nicht
    beschränkt. Fehlt das Volumen (None oder <= 0, z.B. Datenausfall oder ein
    frisch gelisteter Titel), wird NICHT blockiert - ein Datenausfall soll
    nicht fälschlich als Liquiditätsproblem gewertet werden (dieselbe
    Fallback-Haltung wie z.B. bei current_prices.get in compute_nav)."""
    if order.side not in (OrderSide.BUY, OrderSide.SHORT):
        return RiskCheckResult.ok()
    if average_daily_volume is None or average_daily_volume <= 0:
        return RiskCheckResult.ok()
    quantity = order.quantity if order.quantity is not None else order_notional(order, price) / price
    limit = average_daily_volume * max_pct_of_avg_daily_volume
    if quantity > limit:
        return RiskCheckResult.reject(
            f"Order-Menge für {order.symbol} ({quantity:.2f} Stück) übersteigt das Liquiditätslimit von "
            f"{max_pct_of_avg_daily_volume:.0%} des Tagesvolumens ({average_daily_volume:.0f} Stück, "
            f"Limit {limit:.2f})."
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
    average_daily_volume: float | None = None,
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

    `average_daily_volume` (Kap. 6.13, siehe check_liquidity_limit) ist das
    zuletzt bekannte Tagesvolumen des Order-Symbols (MarketSnapshot.volume,
    vom Caller aufgeloest) - None, falls unbekannt, dann greift das
    Liquiditätslimit nicht (siehe dortige Begründung).
    """
    if universe_symbols and order.symbol not in universe_symbols:
        return RiskCheckResult.reject(
            f"Symbol '{order.symbol}' ist nicht im Anlage-Universum - Order abgelehnt."
        )

    if order.side == OrderSide.SHORT and not config.allow_short:
        return RiskCheckResult.reject("Short-Positionen sind laut Risk-Config nicht erlaubt.")

    if order.instrument_type.value in STRUCTURED_INSTRUMENT_TYPES and not config.allow_structured_products:
        return RiskCheckResult.reject("Strukturierte Produkte sind laut Risk-Config nicht erlaubt.")

    # Abgestufter Drawdown-Schutz (siehe drawdown_position_size_factor) -
    # AUSDRUECKLICH nur fuer Nicht-Hebelpositionen (1.0 = keine Reduktion
    # fuer Hebelpositionen/strukturierte Produkte, deren Drawdown-Verhalten
    # unveraendert exklusiv durch check_circuit_breaker unten geregelt wird).
    drawdown_size_factor = 1.0
    if not _is_leverage_controlled(order.instrument_type.value, order_leveraged):
        drawdown_size_factor = drawdown_position_size_factor(
            ctx,
            tier1_drawdown_pct=config.drawdown_tier1_pct,
            tier1_factor=config.drawdown_tier1_position_size_factor,
            tier2_drawdown_pct=config.drawdown_tier2_pct,
            tier2_factor=config.drawdown_tier2_position_size_factor,
            full_stop_drawdown_pct=config.circuit_breaker_drawdown_pct,
        )

    checks = [
        check_daily_loss_stop(ctx, config.daily_loss_stop_pct)
        if order.side in (OrderSide.BUY, OrderSide.SHORT)
        else RiskCheckResult.ok(),
        check_max_trades_per_symbol(order, ctx, config.max_trades_per_symbol_per_day),
        check_no_margin(order, ctx, price, config.allow_margin),
        check_trade_notional(order, ctx, price, config.max_trade_notional_pct_of_nav),
        check_position_size(
            order, ctx, price, config.max_position_size_pct_of_portfolio,
            drawdown_size_factor=drawdown_size_factor,
        ),
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
        check_liquidity_limit(order, price, average_daily_volume, config.max_order_pct_of_avg_daily_volume),
    ]
    result = RiskCheckResult.ok()
    for c in checks:
        result = result.merged_with(c)
    return result


# v13 (2026-09-22, 17-Punkte-Audit Fund #1, HIGH): BEWUSSTE AUSNAHME von der
# Grundregel, Audit-Funde ohne akute Dringlichkeit gesammelt erst zur
# naechsten monatlichen Tiefenreflexion (Kap. 6.12.3, naechster Termin
# Oktober 2026) zu adressieren, statt einzeln vorzuziehen. Begruendung fuer
# das Vorziehen: Fund #1 betrifft den bestehenden Short-Stop-Loss-Mechanismus
# SELBST (nicht die Anlagelogik) - Claude wird im Prompt aufgefordert, pro
# Short-Order einen eigenen stop_loss_price zu nennen; dieser wurde zwar in
# der DB gespeichert und im Report angezeigt, aber
# evaluate_short_positions_for_stop_loss hatte ihn nie gelesen und wendete
# stattdessen ausnahmslos den globalen risk_config.short_stop_loss_pct an -
# ein Pflicht-Sicherheitsmechanismus verhielt sich damit nachweislich anders,
# als das System (Prompt + Report) es Claude UND dem menschlichen Leser
# suggerierte. Das ist keine methodische Aenderung an sich (die Guardrail
# existierte bereits, dieselbe Sweep-Logik, derselbe Aufrufzeitpunkt) und
# kein Eingriff in die Anlagelogik, sondern das Schliessen einer Luecke
# zwischen dokumentiertem und tatsaechlichem Verhalten einer bereits
# bestehenden Leitplanke - genau die Kategorie, bei der ein Zuwarten bis
# Oktober das Risiko selbst (nicht nur seine Dokumentation) einen Monat lang
# unadressiert liesse. Fix: individueller stop_loss_price hat Vorrang, falls
# fuer die Position gesetzt; fehlt er, bleibt der bisherige globale
# Prozent-Fallback unveraendert (Rueckwaertskompatibel, siehe
# test_short_stop_loss_not_triggered_below_threshold).
def evaluate_short_positions_for_stop_loss(
    positions: list[OpenPosition],
    current_prices: dict[str, float],
    short_stop_loss_pct: float,
) -> list[ForcedStopLossAction]:
    """Mandatory sweep run before every pipeline decision cycle.

    For every open short position, force a close if the price has moved
    against the position by more than abs(short_stop_loss_pct) - UNLESS an
    individual stop_loss_price is stored for that position (typically the
    value Claude named when opening the short, see order_schema.ProposedOrder
    .stop_loss_price), in which case that price takes precedence over the
    global percentage (v13, 17-Punkte-Audit Fund #1). This is independent of
    the daily-loss-stop gate and independent of what Claude proposes - it is
    a hard, documentation-required safety mechanism.
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
        if p.stop_loss_price is not None:
            triggered = price >= p.stop_loss_price
            threshold_desc = f"individueller Stop-Loss {p.stop_loss_price:.2f}"
        else:
            triggered = loss_pct >= threshold
            threshold_desc = f"globale Schwelle {threshold:.2%}"
        if triggered:
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
                        f"{p.avg_entry_price:.2f} ({threshold_desc})."
                    ),
                )
            )
    return forced
