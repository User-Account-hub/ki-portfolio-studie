"""Tests for src/stress_test_prompt.py's user-prompt construction - previously
untested (17-point audit Fund #15), analog zu test_deep_reflection_prompt.py.
Pure function, no DB/network.
"""
from __future__ import annotations

import json

import pandas as pd

from src.stress_test import PositionWeight, StressPeriodResult
from src.stress_test_prompt import build_stress_test_user_prompt


def test_build_stress_test_user_prompt_includes_position_weights_and_periods():
    portfolio_row = {"name": "ki-fallstudie-1"}
    position_weights = [PositionWeight(symbol="NVDA", weight=0.6), PositionWeight(symbol="TSDD", weight=-0.4)]
    period_results = [
        StressPeriodResult(
            name="GFC 2008",
            start=pd.Timestamp("2008-09-01"),
            end=pd.Timestamp("2009-03-01"),
            max_drawdown_pct=-0.45,
            included_symbols=["NVDA"],
            excluded_symbols_no_data=["TSDD"],
        )
    ]

    prompt = build_stress_test_user_prompt(portfolio_row, position_weights, period_results)

    payload_json = prompt.split(" (JSON):\n\n", 1)[1].split("\n\nErstelle")[0]
    payload = json.loads(payload_json)

    assert payload["portfolio"] == "ki-fallstudie-1"
    assert payload["current_position_weights"] == [
        {"symbol": "NVDA", "weight_pct": 0.6},
        {"symbol": "TSDD", "weight_pct": -0.4},
    ]
    period = payload["stress_periods"][0]
    assert period["name"] == "GFC 2008"
    assert period["start"] == "2008-09-01"
    assert period["end"] == "2009-03-01"
    assert period["max_drawdown_pct"] == -0.45
    assert period["included_symbols"] == ["NVDA"]
    assert period["excluded_symbols_no_historical_data"] == ["TSDD"]


def test_build_stress_test_user_prompt_handles_empty_inputs():
    prompt = build_stress_test_user_prompt({"name": "test"}, [], [])

    payload_json = prompt.split(" (JSON):\n\n", 1)[1].split("\n\nErstelle")[0]
    payload = json.loads(payload_json)

    assert payload["current_position_weights"] == []
    assert payload["stress_periods"] == []


def test_build_stress_test_user_prompt_handles_none_max_drawdown():
    """max_drawdown_pct ist None, wenn kein Symbol im Zeitraum Historie hat
    (siehe StressPeriodResult-Docstring) - muss als JSON null landen, nicht
    crashen."""
    period_results = [
        StressPeriodResult(
            name="Q4 2018 Selloff",
            start=pd.Timestamp("2018-10-01"),
            end=pd.Timestamp("2018-12-24"),
            max_drawdown_pct=None,
            included_symbols=[],
            excluded_symbols_no_data=["IONQ"],
        )
    ]

    prompt = build_stress_test_user_prompt({"name": "test"}, [], period_results)

    payload_json = prompt.split(" (JSON):\n\n", 1)[1].split("\n\nErstelle")[0]
    payload = json.loads(payload_json)

    assert payload["stress_periods"][0]["max_drawdown_pct"] is None


def test_build_stress_test_user_prompt_ends_with_instruction_to_follow_schema():
    prompt = build_stress_test_user_prompt({"name": "test"}, [], [])
    assert prompt.strip().endswith("Erstelle deinen Kommentar gemäss dem im System-Prompt definierten JSON-Schema.")
