"""Loads .env and YAML configuration into typed objects."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class RiskConfig:
    max_position_size_pct_of_portfolio: float
    max_trade_notional_pct_of_cash: float
    daily_loss_stop_pct: float
    max_trades_per_symbol_per_day: int
    allow_short: bool
    allow_structured_products: bool
    structured_products_max_notional_pct_of_nav: float
    short_stop_loss_pct: float
    allow_margin: bool
    # --- Kap. 6.8 Thesis: zusaetzliche Portfolio-Leitplanken ---
    max_segment_weight_pct_of_nav: float
    correlated_crypto_mining_segments: list[str]
    max_correlated_crypto_mining_pct_of_nav: float
    max_micro_cap_pct_of_nav: float
    max_top3_concentration_pct_of_nav: float
    min_cash_pct_of_nav: float
    circuit_breaker_drawdown_pct: float

    @classmethod
    def from_yaml(cls, path: str) -> "RiskConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(frozen=True)
class WatchlistSymbol:
    symbol: str
    instrument_type: str
    underlying_symbol: str | None = None
    # Thesis-Anhang-A-Taxonomie (nur fuer reguläre Aktien gesetzt, nicht fuer
    # strukturierte Produkte) - Grundlage fuer die Kap.-6.8-Guardrails
    # (Segmentgewicht, korrelierte Krypto/Mining-Exposure, Micro-Cap-Sublimit).
    segment: str | None = None
    cap_tier: str | None = None


@dataclass(frozen=True)
class Watchlist:
    benchmark_symbol: str
    symbols: list[WatchlistSymbol] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "Watchlist":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        symbols = [WatchlistSymbol(**s) for s in data.get("symbols", [])]
        return cls(benchmark_symbol=data["benchmark_symbol"], symbols=symbols)

    def all_symbols(self) -> list[str]:
        return [s.symbol for s in self.symbols]


@dataclass(frozen=True)
class AppConfig:
    anthropic_api_key: str
    claude_model: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    db_path: str
    portfolio_name: str
    initial_cash_balance: float
    portfolio_currency: str
    watchlist_path: str
    risk_config_path: str
    reports_dir: str

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            anthropic_api_key=_require_env("ANTHROPIC_API_KEY"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
            alpaca_api_key=_require_env("ALPACA_API_KEY"),
            alpaca_secret_key=_require_env("ALPACA_SECRET_KEY"),
            alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            db_path=os.getenv("DB_PATH", "./db/portfolio.db"),
            portfolio_name=os.getenv("PORTFOLIO_NAME", "ki-fallstudie-1"),
            initial_cash_balance=float(os.getenv("INITIAL_CASH_BALANCE", "100000")),
            portfolio_currency=os.getenv("PORTFOLIO_CURRENCY", "USD"),
            watchlist_path=os.getenv("WATCHLIST_PATH", "./config/watchlist.yaml"),
            risk_config_path=os.getenv("RISK_CONFIG_PATH", "./config/risk_config.yaml"),
            reports_dir=os.getenv("REPORTS_DIR", "./reports"),
        )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Umgebungsvariable {name} ist nicht gesetzt (siehe .env.example).")
    return value
