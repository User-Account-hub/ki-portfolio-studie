"""Builds the system/user prompt sent to Claude for a trading decision."""
from __future__ import annotations

import json

from src.config import RiskConfig, Watchlist
from src.data_fetch import MarketSnapshot

SYSTEM_PROMPT = """\
Du bist der Portfolio-Analyst einer KI-gestützten Portfolio-Fallstudie im Paper-Trading-Modus \
(kein echtes Geld). Du erhältst den aktuellen Portfolio-Zustand und Marktdaten für ein festes \
Anlage-Universum und schlägst darauf basierend Handelsentscheidungen vor.

Wichtige Rahmenbedingungen:
- Du darfst NUR Symbole aus dem gelieferten Universum vorschlagen.
- Long- und Short-Positionen auf Aktien/ETFs sind erlaubt, ebenso strukturierte Produkte \
(Hebelzertifikate, Mini-Futures, Optionsscheine) auf Titel des Universums.
- Kein Margin-Trading: Order-Notionals dürfen das verfügbare Cash nicht überschreiten.
- Die tatsächliche Durchsetzung aller Risikolimiten (Positionsgrössen, Tagesverlust-Stop, \
strukturierte-Produkte-Obergrenze, Short-Stop-Loss, max. 1 Trade/Symbol/Tag, max. \
Segmentgewichtung, korrelierte Krypto/Mining-Exposure, Micro-Cap-Sublimit, \
Top-3-Konzentration, Mindest-Cash-Quote, Drawdown-Circuit-Breaker für Hebelpositionen) \
erfolgt serverseitig nach deiner Antwort - Vorschläge ausserhalb der Limiten werden \
automatisch abgelehnt und nicht ausgeführt. Halte dich trotzdem an die unten genannten \
Limiten (inkl. "segment"/"cap_tier" je Titel im Universum), um unnötige Ablehnungen zu \
vermeiden.
- Antworte AUSSCHLIESSLICH mit einem einzigen validen JSON-Objekt, ohne Markdown-Fences, \
ohne Fliesstext davor oder danach.

JSON-Ausgabeschema:
{
  "orders": [
    {
      "symbol": "string (muss im Universum sein)",
      "instrument_type": "equity|etf|leverage_certificate|mini_future|warrant",
      "side": "buy|sell|short|cover",
      "quantity": number,            // ODER "notional", nicht beides zwingend, aber mind. eines
      "notional": number,
      "order_type": "market|limit",
      "limit_price": number,          // Pflicht wenn order_type = limit
      "underlying_symbol": "string",  // Pflicht bei strukturierten Produkten
      "stop_loss_price": number,      // empfohlen bei "short"
      "rationale": "string - kurze Begründung"
    }
  ],
  "portfolio_commentary": "string - kurze Gesamteinschätzung"
}

Wenn du aktuell keine Handlung empfiehlst, gib "orders": [] zurück und begründe dies in \
"portfolio_commentary".
"""


def build_user_prompt(
    portfolio_row,
    open_positions: list,
    watchlist: Watchlist,
    snapshots: dict[str, MarketSnapshot],
    risk_config: RiskConfig,
) -> str:
    portfolio_state = {
        "name": portfolio_row["name"],
        "currency": portfolio_row["currency"],
        "cash_balance": portfolio_row["cash_balance"],
        "benchmark_symbol": portfolio_row["benchmark_symbol"],
    }

    positions_state = [
        {
            "symbol": p["symbol"],
            "instrument_type": p["instrument_type"],
            "side": p["side"],
            "quantity": p["quantity"],
            "avg_entry_price": p["avg_entry_price"],
            "stop_loss_price": p["stop_loss_price"],
        }
        for p in open_positions
    ]

    universe = [
        {
            "symbol": s.symbol,
            "instrument_type": s.instrument_type,
            "underlying_symbol": s.underlying_symbol,
            "segment": s.segment,
            "cap_tier": s.cap_tier,
        }
        for s in watchlist.symbols
    ]

    market_data = {
        symbol: {
            "last_price": snap.last_price,
            "change_1d_pct": snap.change_1d_pct,
            "sma20": snap.sma20,
            "sma50": snap.sma50,
            "volatility_20d_annualized": snap.volatility_20d_annualized,
        }
        for symbol, snap in snapshots.items()
    }

    limits = {
        "max_position_size_pct_of_portfolio": risk_config.max_position_size_pct_of_portfolio,
        "max_trade_notional_pct_of_cash": risk_config.max_trade_notional_pct_of_cash,
        "daily_loss_stop_pct": risk_config.daily_loss_stop_pct,
        "max_trades_per_symbol_per_day": risk_config.max_trades_per_symbol_per_day,
        "structured_products_max_notional_pct_of_nav": risk_config.structured_products_max_notional_pct_of_nav,
        "short_stop_loss_pct": risk_config.short_stop_loss_pct,
        "max_segment_weight_pct_of_nav": risk_config.max_segment_weight_pct_of_nav,
        "correlated_crypto_mining_segments": risk_config.correlated_crypto_mining_segments,
        "max_correlated_crypto_mining_pct_of_nav": risk_config.max_correlated_crypto_mining_pct_of_nav,
        "max_micro_cap_pct_of_nav": risk_config.max_micro_cap_pct_of_nav,
        "max_top3_concentration_pct_of_nav": risk_config.max_top3_concentration_pct_of_nav,
        "min_cash_pct_of_nav": risk_config.min_cash_pct_of_nav,
        "circuit_breaker_drawdown_pct": risk_config.circuit_breaker_drawdown_pct,
    }

    payload = {
        "portfolio": portfolio_state,
        "open_positions": positions_state,
        "universe": universe,
        "market_data": market_data,
        "risk_limits": limits,
    }
    return (
        "Aktueller Portfolio-Zustand, Marktdaten und Risikolimiten (JSON):\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Erstelle deine Handelsentscheidung gemäss dem im System-Prompt definierten JSON-Schema."
    )
