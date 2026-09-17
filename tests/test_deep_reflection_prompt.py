"""Tests for src/deep_reflection_prompt.py's trigger timing (Thesis Kap.
6.12.3: monatliche Tiefenreflexion ab Woche 5, rollierende 4-Wochen-Perioden)
and user-prompt construction. Pure functions, no DB/network.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.deep_reflection_prompt import (
    DEEP_REFLECTION_PERIOD_DAYS,
    OFFICIAL_STUDY_START,
    build_deep_reflection_user_prompt,
    due_reflection_period,
    periods_elapsed_since_study_start,
)


# --- periods_elapsed_since_study_start ------------------------------------------


def test_periods_elapsed_is_zero_before_week_5():
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START) == 0
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START + pd.Timedelta(days=27)) == 0


def test_periods_elapsed_becomes_one_exactly_at_week_5():
    """Woche 5 beginnt an Tag 28 (4 volle Wochen = 28 Tage) - genau dort
    muss die erste Periode als abgeschlossen gelten."""
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START + pd.Timedelta(days=28)) == 1
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START + pd.Timedelta(days=55)) == 1


def test_periods_elapsed_increments_every_28_days():
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START + pd.Timedelta(days=56)) == 2
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START + pd.Timedelta(days=84)) == 3


def test_periods_elapsed_never_negative_before_study_start():
    assert periods_elapsed_since_study_start(OFFICIAL_STUDY_START - pd.Timedelta(days=10)) == 0


# --- due_reflection_period -------------------------------------------------------


def test_due_reflection_period_none_before_week_5():
    assert due_reflection_period(0, OFFICIAL_STUDY_START + pd.Timedelta(days=27)) is None


def test_due_reflection_period_covers_weeks_1_to_4_once_week_5_begins():
    result = due_reflection_period(0, OFFICIAL_STUDY_START + pd.Timedelta(days=28))
    assert result == (OFFICIAL_STUDY_START, OFFICIAL_STUDY_START + pd.Timedelta(days=DEEP_REFLECTION_PERIOD_DAYS))


def test_due_reflection_period_none_once_current_period_already_recorded():
    """Wurde die faellige Reflexion (existing_reflection_count=1) bereits
    erledigt, darf am selben Tag keine zweite ausgeloest werden."""
    assert due_reflection_period(1, OFFICIAL_STUDY_START + pd.Timedelta(days=30)) is None


def test_due_reflection_period_advances_to_second_period():
    result = due_reflection_period(1, OFFICIAL_STUDY_START + pd.Timedelta(days=56))
    expected_start = OFFICIAL_STUDY_START + pd.Timedelta(days=28)
    expected_end = OFFICIAL_STUDY_START + pd.Timedelta(days=56)
    assert result == (expected_start, expected_end)


def test_due_reflection_period_catches_up_only_one_period_at_a_time():
    """Auch wenn rechnerisch mehrere Perioden faellig waeren (z.B. nach
    laengerem Pipeline-Ausfall), wird pro Aufruf nur die naechste
    ausstehende zurueckgegeben - kein Rueckstau in einem Lauf."""
    # Tag 90: 3 Perioden waeren vollstaendig (0,1,2), aber noch keine erledigt.
    result = due_reflection_period(0, OFFICIAL_STUDY_START + pd.Timedelta(days=90))
    assert result == (OFFICIAL_STUDY_START, OFFICIAL_STUDY_START + pd.Timedelta(days=28))


# --- build_deep_reflection_user_prompt --------------------------------------------


def test_build_deep_reflection_user_prompt_includes_period_and_returns():
    portfolio_row = {"name": "ki-fallstudie-1"}
    decisions = [{"created_at": "2026-09-25 15:00:00", "rationale": None, "proposed_orders": '[{"symbol": "AAPL"}]'}]
    trades = [{"executed_at": "2026-09-25 15:00:05", "symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}]

    prompt = build_deep_reflection_user_prompt(
        portfolio_row,
        decisions,
        trades,
        period_start=OFFICIAL_STUDY_START,
        period_end=OFFICIAL_STUDY_START + pd.Timedelta(days=28),
        period_portfolio_return_pct=0.05,
        period_benchmark_return_pct=0.03,
        period_baseline_return_pct=0.04,
    )

    assert "2026-09-21" in prompt
    assert "2026-10-19" in prompt
    payload_json = prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0]
    payload = json.loads(payload_json)
    assert payload["period_portfolio_return_pct"] == pytest.approx(0.05)
    assert payload["period_benchmark_return_pct"] == pytest.approx(0.03)
    assert payload["period_momentum_baseline_return_pct"] == pytest.approx(0.04)
    assert payload["decisions_in_period"][0]["proposed_orders"] == [{"symbol": "AAPL"}]
    assert payload["trades_in_period"][0]["symbol"] == "AAPL"


def test_build_deep_reflection_user_prompt_handles_no_proposed_orders():
    portfolio_row = {"name": "test"}
    decisions = [{"created_at": "2026-09-25 15:00:00", "rationale": "Börse geschlossen", "proposed_orders": None}]

    prompt = build_deep_reflection_user_prompt(
        portfolio_row, decisions, [], OFFICIAL_STUDY_START,
        OFFICIAL_STUDY_START + pd.Timedelta(days=28), 0.0, 0.0, 0.0,
    )
    payload_json = prompt.split("(JSON):\n\n", 1)[1].split("\n\nErstelle")[0]
    payload = json.loads(payload_json)
    assert payload["decisions_in_period"][0]["proposed_orders"] is None
    assert payload["decisions_in_period"][0]["rationale"] == "Börse geschlossen"
