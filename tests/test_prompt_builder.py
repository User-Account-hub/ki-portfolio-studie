"""Tests for src/prompt_builder.py's Kap.-6.12.3 "Nachwirkung": the latest
monthly deep-reflection result must be surfaced to the daily prompt as
`latest_deep_reflection`, robustly (None before the first reflection, or if
its raw_response can't be parsed).
"""
from __future__ import annotations

import json

import datetime

from src.boundary_conditions import BoundaryConditionCheck
from src.config import RiskConfig, Watchlist, WatchlistSymbol
from src.event_calendar import EarningsWarning, MacroEvent
from src.fundamentals import FundamentalSnapshot
from src.prompt_builder import (
    SYSTEM_PROMPT,
    _summarize_boundary_condition,
    _summarize_earnings_warning,
    _summarize_macro_event,
    _summarize_reflection_for_prompt,
    build_user_prompt,
)


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
        volatility_scaling_min_factor=0.5,
        volatility_scaling_max_factor=1.5,
        drawdown_tier1_pct=-0.10,
        drawdown_tier1_position_size_factor=0.75,
        drawdown_tier2_pct=-0.15,
        drawdown_tier2_position_size_factor=0.50,
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


def _build_minimal_prompt(
    latest_reflection=None,
    triggered_boundary_conditions=None,
    still_open_boundary_conditions=None,
    earnings_warnings=None,
    macro_events=None,
    fundamentals=None,
    news_text_block=None,
):
    portfolio_row = {"name": "test", "currency": "USD", "cash_balance": 100_000.0, "benchmark_symbol": "SPY"}
    watchlist = Watchlist(benchmark_symbol="SPY", symbols=[WatchlistSymbol(symbol="AAPL", instrument_type="equity")])
    return build_user_prompt(
        portfolio_row, [], watchlist, {}, make_risk_config(),
        latest_reflection=latest_reflection,
        triggered_boundary_conditions=triggered_boundary_conditions,
        still_open_boundary_conditions=still_open_boundary_conditions,
        earnings_warnings=earnings_warnings,
        macro_events=macro_events,
        fundamentals=fundamentals,
        news_text_block=news_text_block,
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


# --- Randbedingungen (Kap. 7) -------------------------------------------------


def test_summarize_boundary_condition_excludes_internal_ids():
    cond = BoundaryConditionCheck(
        id=1, position_id=10, symbol="NVDA", description="Fällt unter $150",
        check_type="price_below", threshold_price=150.0,
    )
    summary = _summarize_boundary_condition(cond)
    assert summary == {
        "symbol": "NVDA", "description": "Fällt unter $150",
        "check_type": "price_below", "threshold_price": 150.0,
    }
    assert "id" not in summary
    assert "position_id" not in summary


def test_build_user_prompt_defaults_boundary_conditions_to_empty_lists():
    prompt = _build_minimal_prompt()
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["boundary_conditions"] == {"triggered_this_run": [], "still_open": []}


def test_build_user_prompt_includes_boundary_condition_status():
    triggered = [
        BoundaryConditionCheck(
            id=1, position_id=10, symbol="NVDA", description="Fällt unter $150",
            check_type="price_below", threshold_price=150.0,
        )
    ]
    still_open = [
        BoundaryConditionCheck(
            id=2, position_id=11, symbol="CCJ", description="Q3-Earnings enttäuschen",
            check_type="qualitative", threshold_price=None,
        )
    ]
    prompt = _build_minimal_prompt(triggered_boundary_conditions=triggered, still_open_boundary_conditions=still_open)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["boundary_conditions"]["triggered_this_run"][0]["symbol"] == "NVDA"
    assert payload["boundary_conditions"]["still_open"][0]["symbol"] == "CCJ"


# --- Event-Kalender-Hinweis --------------------------------------------------


def test_summarize_earnings_warning_includes_required_wording():
    warning = EarningsWarning(symbol="NVDA", earnings_date=datetime.date(2026, 9, 21), trading_days_until=2)
    summary = _summarize_earnings_warning(warning)
    assert summary["symbol"] == "NVDA"
    assert summary["earnings_date"] == "2026-09-21"
    assert summary["trading_days_until"] == 2
    assert summary["note"] == "NVDA berichtet in 2 Handelstag(en) - erhöhtes Ereignisrisiko."


def test_summarize_macro_event_includes_portfolio_wide_wording():
    event = MacroEvent(name="FOMC", event_date=datetime.date(2026, 10, 28), trading_days_until=1)
    summary = _summarize_macro_event(event)
    assert summary["name"] == "FOMC"
    assert "FOMC-Termin in 1 Handelstag(en)" in summary["note"]
    assert "gesamte Portfolio" in summary["note"]


def test_build_user_prompt_defaults_upcoming_events_to_empty_lists():
    prompt = _build_minimal_prompt()
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["upcoming_events"] == {
        "earnings_within_3_trading_days": [],
        "macro_events_within_3_trading_days": [],
    }


def test_build_user_prompt_includes_earnings_and_macro_events():
    earnings = [EarningsWarning(symbol="NVDA", earnings_date=datetime.date(2026, 9, 21), trading_days_until=1)]
    macro = [MacroEvent(name="CPI", event_date=datetime.date(2026, 10, 14), trading_days_until=2)]
    prompt = _build_minimal_prompt(earnings_warnings=earnings, macro_events=macro)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["upcoming_events"]["earnings_within_3_trading_days"][0]["symbol"] == "NVDA"
    assert payload["upcoming_events"]["macro_events_within_3_trading_days"][0]["name"] == "CPI"


# --- Weicher Qualitäts-Score (Fundamentaldaten) -----------------------------


def test_build_user_prompt_defaults_fundamentals_to_empty_dict():
    prompt = _build_minimal_prompt()
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["fundamentals"] == {}


def test_build_user_prompt_includes_fundamentals_per_symbol():
    snapshots = {
        "AAPL": FundamentalSnapshot(symbol="AAPL", revenue_growth=0.164, debt_to_equity=78.445, free_cash_flow=1.0e11),
    }
    prompt = _build_minimal_prompt(fundamentals=snapshots)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["fundamentals"]["AAPL"] == {
        "revenue_growth": 0.164,
        "debt_to_equity": 78.445,
        "free_cash_flow": 1.0e11,
    }


def test_build_user_prompt_fundamentals_keeps_missing_fields_as_none():
    snapshots = {"SOME": FundamentalSnapshot(symbol="SOME", revenue_growth=0.05, debt_to_equity=None, free_cash_flow=None)}
    prompt = _build_minimal_prompt(fundamentals=snapshots)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["fundamentals"]["SOME"]["debt_to_equity"] is None
    assert payload["fundamentals"]["SOME"]["free_cash_flow"] is None


def test_system_prompt_mentions_fundamentals_are_not_a_filter():
    """Kernaussage des Auftrags: der System-Prompt muss explizit klarstellen,
    dass Fundamentaldaten kein Ausschlusskriterium sind."""
    assert "KEIN Ausschlusskriterium" in SYSTEM_PROMPT
    assert "spekulativ" in SYSTEM_PROMPT


# --- news_context (Kap. 12.2, 2026-09-21) -------------------------------------


def test_build_user_prompt_includes_none_when_no_news_text_block():
    prompt = _build_minimal_prompt(news_text_block=None)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["news_context"] is None


def test_build_user_prompt_includes_none_when_news_text_block_is_empty_string():
    """Ein fehlgeschlagener Aggregationslauf liefert '' (siehe
    pipeline._fetch_news_context) - das muss im Prompt wie "kein Kontext"
    (null) aussehen, nicht wie ein leerer, aber vorhandener String."""
    prompt = _build_minimal_prompt(news_text_block="")
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["news_context"] is None


def test_build_user_prompt_includes_news_text_block_verbatim_when_present():
    block = "[CNBC - Tech]\n  - [2026-09-20] Beispiel-Meldung"
    prompt = _build_minimal_prompt(news_text_block=block)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["news_context"] == block


def test_system_prompt_mentions_news_context_is_not_exhaustive_or_binding():
    """Der System-Prompt muss Claude ueber das neue Feld informieren und
    klarstellen, dass es kein Ausschlusskriterium/Vollstaendigkeitsanspruch
    hat (eigene Nennung, nicht nur die bereits bestehende bei fundamentals)."""
    assert "news_context" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.count("KEIN Ausschlusskriterium") >= 2
