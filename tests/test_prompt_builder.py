"""Tests for src/prompt_builder.py's Kap.-6.12.3 "Nachwirkung": the latest
monthly deep-reflection result must be surfaced to the daily prompt as
`latest_deep_reflection`, robustly (None before the first reflection, or if
its raw_response can't be parsed).
"""
from __future__ import annotations

import json

import datetime

import pytest

from src.boundary_conditions import BoundaryConditionCheck
from src.config import RiskConfig, Watchlist, WatchlistSymbol
from src.data_fetch import MarketSnapshot
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
        max_order_pct_of_avg_daily_volume=0.10,
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
        start_of_run_nav=100_000.0,
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


# --- Liquiditätslimit (Kap. 6.13, 2026-09-21) --------------------------------


def test_build_user_prompt_includes_volume_in_market_data():
    portfolio_row = {"name": "test", "currency": "USD", "cash_balance": 100_000.0, "benchmark_symbol": "SPY"}
    watchlist = Watchlist(benchmark_symbol="SPY", symbols=[WatchlistSymbol(symbol="AAPL", instrument_type="equity")])
    snapshots = {
        "AAPL": MarketSnapshot(
            symbol="AAPL", last_price=190.0, change_1d_pct=0.01, sma20=185.0, sma50=180.0,
            volatility_20d_annualized=0.25, volume=45_123_456,
        )
    }
    prompt = build_user_prompt(portfolio_row, [], watchlist, snapshots, make_risk_config(), start_of_run_nav=100_000.0)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["market_data"]["AAPL"]["volume"] == 45_123_456


def test_build_user_prompt_market_data_volume_is_none_when_unavailable():
    portfolio_row = {"name": "test", "currency": "USD", "cash_balance": 100_000.0, "benchmark_symbol": "SPY"}
    watchlist = Watchlist(benchmark_symbol="SPY", symbols=[WatchlistSymbol(symbol="AAPL", instrument_type="equity")])
    snapshots = {
        "AAPL": MarketSnapshot(
            symbol="AAPL", last_price=190.0, change_1d_pct=0.01, sma20=185.0, sma50=180.0,
            volatility_20d_annualized=0.25, volume=None,
        )
    }
    prompt = build_user_prompt(portfolio_row, [], watchlist, snapshots, make_risk_config(), start_of_run_nav=100_000.0)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["market_data"]["AAPL"]["volume"] is None


def test_build_user_prompt_includes_liquidity_limit_in_risk_limits():
    prompt = _build_minimal_prompt()
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])
    assert payload["risk_limits"]["max_order_pct_of_avg_daily_volume"] == pytest.approx(0.10)


def test_system_prompt_mentions_liquidity_limit():
    assert "Liquiditätslimit" in SYSTEM_PROMPT


# --- Post-hoc Vol-/Konviktions-Reskalierung (v10, 2026-09-21) -----------------


def test_system_prompt_mentions_post_hoc_order_size_rescaling():
    """v10: Claude muss explizit darauf hingewiesen werden, dass seine
    vorgeschlagene Order-Groesse NACH seiner Entscheidung noch reskaliert
    wird (Volatilitaets-Band 0.5x-1.5x, siehe src/position_sizing.py) -
    Grund war, dass am 2026-09-21 drei Kauf-Orders (TSM, ASML, CCJ)
    ausschliesslich wegen dieser fuer Claude unsichtbaren Reskalierung ueber
    das Trade-Notional-Limit gehoben und abgelehnt wurden (siehe
    Code-Kommentar bei SYSTEM_PROMPT)."""
    assert "reskaliert" in SYSTEM_PROMPT
    assert "0.5x-1.5x" in SYSTEM_PROMPT
    assert "volatility_20d_annualized" in SYSTEM_PROMPT


def test_system_prompt_advises_more_conservative_sizing_for_volatile_names():
    assert "konservativer" in SYSTEM_PROMPT
    assert "überdurchschnittlicher Volatilität" in SYSTEM_PROMPT


def test_system_prompt_mentions_liquidity_limit_concrete_value_in_v10_hint():
    """v10-Ergaenzung (2026-09-21, selber Tag): Claude muss im selben Hinweis
    auch ueber das Liquiditaetslimit (Kap. 6.13) informiert werden, inkl. des
    konkreten, aus risk_config.yaml uebernommenen Prozentwerts
    (max_order_pct_of_avg_daily_volume = 0.10)."""
    assert "max_order_pct_of_avg_daily_volume" in SYSTEM_PROMPT
    assert "10%" in SYSTEM_PROMPT
    assert "GEKAPPT" in SYSTEM_PROMPT


def test_system_prompt_mentions_tiered_drawdown_position_size_reduction():
    """v10-Ergaenzung: der gestufte Drawdown-Schutz (Kap. 6.8) - -10%/-15%
    Drawdown reduziert die maximal erlaubte Positionsgroesse auf 75%/50% -
    muss Claude im selben Hinweis erklaert werden, inkl. der konkreten Werte
    aus risk_config.yaml (drawdown_tier1_pct=-0.10/-tier1_position_size_
    factor=0.75, drawdown_tier2_pct=-0.15/-tier2_position_size_factor=0.50)."""
    assert "drawdown_tier1_pct" in SYSTEM_PROMPT
    assert "drawdown_tier2_pct" in SYSTEM_PROMPT
    assert "-10%" in SYSTEM_PROMPT
    assert "-15%" in SYSTEM_PROMPT
    assert "75%" in SYSTEM_PROMPT
    assert "50%" in SYSTEM_PROMPT


# --- v11: vorberechnete max_conservative_notional_usd (2026-09-21) -----------


def test_system_prompt_mentions_max_conservative_notional_field_v11():
    """v11: Reaktion auf Testlauf 94f4465, in dem die rein qualitative
    v10-Empfehlung nicht ausreichte - Claude muss auf das neue, konkret
    vorberechnete Feld "max_conservative_notional_usd" je Titel hingewiesen
    und angewiesen werden, es als verbindliche Obergrenze zu behandeln."""
    assert "max_conservative_notional_usd" in SYSTEM_PROMPT
    assert "VERBINDLICHE OBERGRENZE" in SYSTEM_PROMPT


def test_build_user_prompt_max_conservative_notional_reflects_worst_case_volatility_scaling():
    """Zwei Symbole mit unterschiedlicher Volatilitaet muessen unterschiedliche
    max_conservative_notional_usd-Werte bekommen: das Symbol mit der
    NIEDRIGEREN Volatilitaet wird staerker hochskaliert (hoeherer
    Skalierungsfaktor) und darf deshalb UNSKALIERT weniger vorgeschlagen
    werden, damit es nach der Skalierung nicht ueber dem Limit landet."""
    portfolio_row = {"name": "test", "currency": "USD", "cash_balance": 1_000_000.0, "benchmark_symbol": "SPY"}
    watchlist = Watchlist(
        benchmark_symbol="SPY",
        symbols=[
            WatchlistSymbol(symbol="LOWVOL", instrument_type="equity"),
            WatchlistSymbol(symbol="HIGHVOL", instrument_type="equity"),
        ],
    )
    snapshots = {
        "LOWVOL": MarketSnapshot(
            symbol="LOWVOL", last_price=100.0, change_1d_pct=0.0, sma20=100.0, sma50=100.0,
            volatility_20d_annualized=0.10, volume=1_000_000,
        ),
        "HIGHVOL": MarketSnapshot(
            symbol="HIGHVOL", last_price=100.0, change_1d_pct=0.0, sma20=100.0, sma50=100.0,
            volatility_20d_annualized=0.90, volume=1_000_000,
        ),
    }
    risk_config = make_risk_config(
        max_trade_notional_pct_of_nav=0.05, volatility_scaling_min_factor=0.5, volatility_scaling_max_factor=1.5
    )
    prompt = build_user_prompt(portfolio_row, [], watchlist, snapshots, risk_config, start_of_run_nav=1_000_000.0)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])

    # avg_vol = (0.10+0.90)/2 = 0.50; LOWVOL-Faktor = 0.50/0.10 = 5.0 -> auf
    # max_factor 1.5 geclippt; HIGHVOL-Faktor = 0.50/0.90 = 0.5556 (kein
    # Clipping noetig). Trade-Notional-Limit = 0.05 * 1_000_000 = 50_000.
    trade_notional_limit = 50_000.0
    expected_lowvol = trade_notional_limit / (1.5 * 1.15)
    expected_highvol = trade_notional_limit / ((0.50 / 0.90) * 1.15)

    assert payload["market_data"]["LOWVOL"]["max_conservative_notional_usd"] == pytest.approx(expected_lowvol)
    assert payload["market_data"]["HIGHVOL"]["max_conservative_notional_usd"] == pytest.approx(expected_highvol)
    assert expected_lowvol < expected_highvol


def test_build_user_prompt_max_conservative_notional_uses_conviction_only_without_volatility_data():
    """Fehlen Volatilitaetsdaten fuer ein Symbol, muss der Vol-Faktor wie in
    execution.py auf 1.0 (KEINE Skalierung) fallen, nicht auf 0.5x-1.5x -
    max_conservative_notional_usd ist dann nur durch den worst-case
    Konviktions-Faktor (1.15) begrenzt."""
    portfolio_row = {"name": "test", "currency": "USD", "cash_balance": 1_000_000.0, "benchmark_symbol": "SPY"}
    watchlist = Watchlist(benchmark_symbol="SPY", symbols=[WatchlistSymbol(symbol="NOVOL", instrument_type="equity")])
    snapshots = {
        "NOVOL": MarketSnapshot(
            symbol="NOVOL", last_price=100.0, change_1d_pct=0.0, sma20=100.0, sma50=100.0,
            volatility_20d_annualized=None, volume=1_000_000,
        )
    }
    risk_config = make_risk_config(max_trade_notional_pct_of_nav=0.05)
    prompt = build_user_prompt(portfolio_row, [], watchlist, snapshots, risk_config, start_of_run_nav=1_000_000.0)
    payload = json.loads(prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0])

    expected = (0.05 * 1_000_000.0) / 1.15
    assert payload["market_data"]["NOVOL"]["max_conservative_notional_usd"] == pytest.approx(expected)
