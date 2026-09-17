"""Tests for src/prompt_builder.py's Kap.-6.12.3 "Nachwirkung": the latest
monthly deep-reflection result must be surfaced to the daily prompt as
`latest_deep_reflection`, robustly (None before the first reflection, or if
its raw_response can't be parsed).
"""
from __future__ import annotations

import json

from src.config import RiskConfig, Watchlist, WatchlistSymbol
from src.prompt_builder import _summarize_reflection_for_prompt, build_user_prompt


def make_risk_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_position_size_pct_of_portfolio=0.10,
        max_trade_notional_pct_of_nav=0.05,
        daily_loss_stop_pct=-0.03,
        max_trades_per_symbol_per_day=1,
        allow_short=True,
        allow_structured_products=True,
        structured_products_max_notional_pct_of_nav=0.20,
        short_stop_loss_pct=-0.20,
        allow_margin=False,
        max_segment_weight_pct_of_nav=0.30,
        correlated_crypto_mining_segments=[],
        max_correlated_crypto_mining_pct_of_nav=0.30,
        max_micro_cap_pct_of_nav=0.15,
        max_top3_concentration_pct_of_nav=0.35,
        min_cash_pct_of_nav=0.05,
        circuit_breaker_drawdown_pct=-0.25,
        transaction_cost_pct_of_notional=0.001,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


# --- _summarize_reflection_for_prompt ---------------------------------------


def test_summarize_reflection_none_before_first_reflection():
    assert _summarize_reflection_for_prompt(None) is None


def test_summarize_reflection_extracts_relevant_fields():
    row = {
        "created_at": "2026-10-19 15:00:00",
        "raw_response": json.dumps(
            {
                "theses_confirmed": ["A"],
                "theses_falsified_or_overdue": ["B ist überfällig"],
                "pattern_matching_concerns": "keine",
                "portfolio_stance_assessment": "kohärent",
                "reflection_commentary": "Alles im Rahmen.",
            }
        ),
    }
    summary = _summarize_reflection_for_prompt(row)
    assert summary["created_at"] == "2026-10-19 15:00:00"
    assert summary["theses_falsified_or_overdue"] == ["B ist überfällig"]
    assert summary["pattern_matching_concerns"] == "keine"
    assert summary["portfolio_stance_assessment"] == "kohärent"
    # reflection_commentary bewusst NICHT im Kurz-Auszug (haelt den taeglichen
    # Prompt schlank) - nur die fuer die Tagesentscheidung relevanten Felder.
    assert "reflection_commentary" not in summary


def test_summarize_reflection_returns_none_on_unparseable_raw_response():
    """Robust gegen kaputte/fehlende Daten - darf den gesamten taeglichen
    Prompt-Aufbau nicht zum Absturz bringen."""
    row = {"created_at": "2026-10-19 15:00:00", "raw_response": "kein valides JSON"}
    assert _summarize_reflection_for_prompt(row) is None


def test_summarize_reflection_returns_none_when_raw_response_missing():
    row = {"created_at": "2026-10-19 15:00:00", "raw_response": None}
    assert _summarize_reflection_for_prompt(row) is None


# --- build_user_prompt integration ------------------------------------------


def _build_minimal_prompt(latest_reflection):
    portfolio_row = {"name": "test", "currency": "USD", "cash_balance": 100_000.0, "benchmark_symbol": "SPY"}
    watchlist = Watchlist(benchmark_symbol="SPY", symbols=[WatchlistSymbol(symbol="AAPL", instrument_type="equity")])
    return build_user_prompt(
        portfolio_row, [], watchlist, {}, make_risk_config(), latest_reflection=latest_reflection
    )


def test_build_user_prompt_includes_none_when_no_reflection_yet():
    prompt = _build_minimal_prompt(latest_reflection=None)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["latest_deep_reflection"] is None


def test_build_user_prompt_includes_reflection_summary_when_present():
    row = {
        "created_at": "2026-10-19 15:00:00",
        "raw_response": json.dumps(
            {
                "theses_falsified_or_overdue": ["CCJ überfällig"],
                "pattern_matching_concerns": "keine",
                "portfolio_stance_assessment": "kohärent",
                "reflection_commentary": "...",
            }
        ),
    }
    prompt = _build_minimal_prompt(latest_reflection=row)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["latest_deep_reflection"]["theses_falsified_or_overdue"] == ["CCJ überfällig"]
