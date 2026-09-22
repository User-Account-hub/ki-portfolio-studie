"""Tests for src/config.py's real YAML/.env loading paths (RiskConfig.from_yaml,
AppConfig.load) - previously untested (17-point audit Fund #10). Deliberately
exercises the actual file-read/env-parsing code, not just the dataclasses
themselves.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _valid_risk_config_yaml() -> str:
    return """
max_position_size_pct_of_portfolio: 0.10
max_trade_notional_pct_of_nav: 0.05
daily_loss_stop_pct: -0.03
max_trades_per_symbol_per_day: 1
allow_short: true
allow_structured_products: true
structured_products_max_notional_pct_of_nav: 0.20
short_stop_loss_pct: -0.20
allow_margin: false
max_segment_weight_pct_of_nav: 0.30
correlated_crypto_mining_segments: ["Krypto-Mining"]
max_correlated_crypto_mining_pct_of_nav: 0.30
max_micro_cap_pct_of_nav: 0.15
max_top3_concentration_pct_of_nav: 0.35
min_cash_pct_of_nav: 0.05
circuit_breaker_drawdown_pct: -0.25
drawdown_tier1_pct: -0.10
drawdown_tier1_position_size_factor: 0.75
drawdown_tier2_pct: -0.15
drawdown_tier2_position_size_factor: 0.50
transaction_cost_pct_of_notional: 0.001
volatility_scaling_min_factor: 0.5
volatility_scaling_max_factor: 1.5
max_order_pct_of_avg_daily_volume: 0.10
"""


# --- RiskConfig.from_yaml -----------------------------------------------------


def test_risk_config_from_yaml_reads_values_from_real_file(tmp_path):
    yaml_path = tmp_path / "risk_config.yaml"
    yaml_path.write_text(_valid_risk_config_yaml(), encoding="utf-8")

    risk_config = config.RiskConfig.from_yaml(str(yaml_path))

    assert risk_config.max_position_size_pct_of_portfolio == 0.10
    assert risk_config.short_stop_loss_pct == -0.20
    assert risk_config.allow_short is True
    assert risk_config.allow_margin is False
    assert risk_config.correlated_crypto_mining_segments == ["Krypto-Mining"]


def test_risk_config_from_yaml_missing_field_raises(tmp_path):
    """Ein Pflichtfeld fehlt in der YAML-Datei - from_yaml darf das nicht
    stillschweigend mit einem Default auffuellen, sondern muss (ueber die
    frozen dataclass) hart fehlschlagen."""
    yaml_path = tmp_path / "risk_config.yaml"
    incomplete = _valid_risk_config_yaml().replace("allow_margin: false\n", "")
    yaml_path.write_text(incomplete, encoding="utf-8")

    with pytest.raises(TypeError):
        config.RiskConfig.from_yaml(str(yaml_path))


def test_risk_config_from_yaml_reads_real_repo_config():
    """Regressionsschutz: config/risk_config.yaml (die tatsaechlich in
    Produktion genutzte Datei) muss mit den RiskConfig-Feldern in Sync
    bleiben - ein Feld-Mismatch faellt hier sofort auf, statt erst beim
    naechsten echten Pipeline-Lauf."""
    risk_config = config.RiskConfig.from_yaml(str(REPO_ROOT / "config" / "risk_config.yaml"))
    assert risk_config.allow_short is True
    assert risk_config.short_stop_loss_pct < 0
    assert risk_config.correlated_crypto_mining_segments  # nicht leer


# --- AppConfig.load ------------------------------------------------------------


REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
OPTIONAL_ENV_VARS = [
    "CLAUDE_MODEL", "ALPACA_BASE_URL", "DB_PATH", "PORTFOLIO_NAME",
    "INITIAL_CASH_BALANCE", "PORTFOLIO_CURRENCY", "WATCHLIST_PATH",
    "RISK_CONFIG_PATH", "REPORTS_DIR", "RISK_FREE_RATE_ANNUAL",
]


def _isolate_env(monkeypatch):
    """Verhindert, dass ein evtl. lokal vorhandenes echtes .env (siehe
    README, gitignored) die Default-Annahmen unten verfaelscht - load_dotenv
    selbst wird neutralisiert, alle relevanten Variablen explizit gesetzt."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in REQUIRED_ENV_VARS + OPTIONAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_app_config_load_applies_documented_defaults(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "x")

    app_config = config.AppConfig.load()

    assert app_config.claude_model == "claude-sonnet-5"
    assert app_config.alpaca_base_url == "https://paper-api.alpaca.markets"
    assert app_config.db_path == "./db/portfolio.db"
    assert app_config.portfolio_name == "ki-fallstudie-1"
    assert app_config.initial_cash_balance == 100_000.0
    assert app_config.portfolio_currency == "USD"
    assert app_config.watchlist_path == "./config/watchlist.yaml"
    assert app_config.risk_config_path == "./config/risk_config.yaml"
    assert app_config.reports_dir == "./reports"
    assert app_config.risk_free_rate_annual == 0.04


def test_app_config_load_overrides_defaults_from_env(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "x")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("INITIAL_CASH_BALANCE", "250000")

    app_config = config.AppConfig.load()

    assert app_config.claude_model == "claude-opus-5"
    assert app_config.initial_cash_balance == 250_000.0


@pytest.mark.parametrize("missing", REQUIRED_ENV_VARS)
def test_app_config_load_raises_when_required_env_var_missing(monkeypatch, missing):
    _isolate_env(monkeypatch)
    for name in REQUIRED_ENV_VARS:
        if name != missing:
            monkeypatch.setenv(name, "x")

    with pytest.raises(RuntimeError, match=missing):
        config.AppConfig.load()
