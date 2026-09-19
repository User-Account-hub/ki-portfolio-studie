"""Tests for src/pipeline.py's _maybe_run_deep_reflection - specifically the
2026-09-19 self-consistency check: EXCLUSIVE to the monthly deep reflection
(out of API-cost concerns, this must NOT run in the daily trading path,
which is why the double call lives here and not in prompt_builder/execution).

Uses the real db/schema.sql against an in-memory SQLite DB (same pattern as
test_execution.py) and monkeypatches src.pipeline.get_trading_decision to
avoid any real Anthropic API call - the point of these tests is the
orchestration (called twice with an identical prompt, both responses
persisted/compared), not Claude's actual output.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import db, pipeline
from src.deep_reflection_prompt import OFFICIAL_STUDY_START
from src.deep_reflection_schema import DeepReflectionRunResult

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

# Genau die erste faellige Periode (Woche 5) - siehe
# deep_reflection_prompt.due_reflection_period.
DUE_TODAY = OFFICIAL_STUDY_START + pd.Timedelta(days=28)

RESPONSE_A = """{
  "theses_confirmed": ["NVDA-These bestätigt"],
  "theses_falsified_or_overdue": [],
  "pattern_matching_concerns": "keine",
  "portfolio_stance_assessment": "kohärent",
  "reflection_commentary": "Erster Aufruf."
}"""

RESPONSE_B_IDENTICAL = RESPONSE_A.replace("Erster Aufruf.", "Zweiter Aufruf, gleiche Kernaussagen.")

RESPONSE_C_CONTRADICTS = """{
  "theses_confirmed": [],
  "theses_falsified_or_overdue": ["NVDA-These bestätigt"],
  "pattern_matching_concerns": "keine",
  "portfolio_stance_assessment": "kohärent",
  "reflection_commentary": "Zweiter Aufruf, widersprechend."
}"""


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def make_portfolio(conn: sqlite3.Connection, name: str = "test") -> sqlite3.Row:
    conn.execute(
        "INSERT INTO portfolios (name, currency, initial_cash_balance, cash_balance) VALUES (?, ?, ?, ?)",
        (name, "USD", 100_000.0, 100_000.0),
    )
    conn.commit()
    return db.get_portfolio(conn, name)


def make_nav_history() -> SimpleNamespace:
    return SimpleNamespace(
        dates=[DUE_TODAY - pd.Timedelta(days=1)],
        nav=[100_000.0],
        benchmark_normalized=[100_000.0],
        baseline_normalized=[100_000.0],
        initial_nav=100_000.0,
    )


def make_metrics_result() -> SimpleNamespace:
    return SimpleNamespace(current_nav=110_000.0)


def make_app_config() -> SimpleNamespace:
    return SimpleNamespace(anthropic_api_key="x", claude_model="claude-sonnet-5")


class RecordingClaudeStub:
    """Stand-in for src.pipeline.get_trading_decision - returns each entry of
    `responses` in order and records every call's (system_prompt, user_prompt)
    so tests can assert the second call used an IDENTICAL prompt."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt, user_prompt, api_key, model, **kwargs):
        self.calls.append((system_prompt, user_prompt))
        return self._responses[len(self.calls) - 1]


def test_calls_claude_exactly_twice_with_identical_prompt(monkeypatch):
    stub = RecordingClaudeStub([RESPONSE_A, RESPONSE_B_IDENTICAL])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(), today=DUE_TODAY,
    )

    assert len(stub.calls) == 2
    assert stub.calls[0] == stub.calls[1]  # identischer System- UND User-Prompt


def test_returns_deep_reflection_run_result_with_both_reflections(monkeypatch):
    stub = RecordingClaudeStub([RESPONSE_A, RESPONSE_B_IDENTICAL])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    result = pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(), today=DUE_TODAY,
    )

    assert isinstance(result, DeepReflectionRunResult)
    assert result.primary.reflection_commentary == "Erster Aufruf."
    assert result.secondary.reflection_commentary == "Zweiter Aufruf, gleiche Kernaussagen."


def test_consistent_when_both_calls_agree_on_core_claims(monkeypatch):
    stub = RecordingClaudeStub([RESPONSE_A, RESPONSE_B_IDENTICAL])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    result = pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(), today=DUE_TODAY,
    )

    assert result.consistency.consistent
    assert result.consistency.mismatch_details == []


def test_inconsistent_when_calls_contradict_and_no_auto_resolution(monkeypatch):
    """Kernanforderung: bei Abweichung KEINE automatische Konfliktloesung -
    `primary` bleibt unveraendert die erste Antwort, unabhaengig vom
    Konsistenz-Ergebnis; die zweite Antwort bleibt vollstaendig erhalten."""
    stub = RecordingClaudeStub([RESPONSE_A, RESPONSE_C_CONTRADICTS])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    result = pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(), today=DUE_TODAY,
    )

    assert not result.consistency.consistent
    assert result.consistency.mismatch_details  # nicht leer
    # Keine automatische Konfliktloesung: primary ist unveraendert die erste Antwort.
    assert result.primary.theses_confirmed == ["NVDA-These bestätigt"]
    assert result.secondary.theses_falsified_or_overdue == ["NVDA-These bestätigt"]


def test_persists_exactly_one_decision_row_with_both_raw_responses(monkeypatch):
    """Kein zweiter Decision-Eintrag fuer den zweiten Aufruf - beide
    Antworten landen in EINER Zeile (raw_response = 1. Aufruf,
    risk_check_result traegt den 2. Aufruf + das Konsistenz-Ergebnis)."""
    stub = RecordingClaudeStub([RESPONSE_A, RESPONSE_C_CONTRADICTS])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(), today=DUE_TODAY,
    )

    rows = conn.execute(
        "SELECT * FROM decisions WHERE portfolio_id = ? AND model = 'deep_reflection'", (portfolio_row["id"],)
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["raw_response"] == RESPONSE_A

    import json

    risk_check_result = json.loads(row["risk_check_result"])
    check = risk_check_result[0]["self_consistency_check"]
    assert check["consistent"] is False
    assert check["second_raw_response"] == RESPONSE_C_CONTRADICTS
    assert check["mismatch_details"]


def test_returns_none_and_persists_nothing_when_not_due(monkeypatch):
    stub = RecordingClaudeStub([RESPONSE_A, RESPONSE_B_IDENTICAL])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    result = pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(),
        today=OFFICIAL_STUDY_START,  # noch keine volle Periode vergangen
    )

    assert result is None
    assert stub.calls == []
    rows = conn.execute(
        "SELECT * FROM decisions WHERE portfolio_id = ? AND model = 'deep_reflection'", (portfolio_row["id"],)
    ).fetchall()
    assert rows == []


def test_second_call_failure_persists_nothing(monkeypatch):
    """Symmetrisch zu einem fehlschlagenden ersten Aufruf (bestehendes
    Verhalten): scheitert der ZWEITE Aufruf (hier: kein valides JSON), darf
    trotzdem nichts in der DB landen, damit dieselbe Periode beim naechsten
    faelligen Lauf automatisch erneut versucht wird."""
    stub = RecordingClaudeStub([RESPONSE_A, "Das ist kein JSON."])
    monkeypatch.setattr(pipeline, "get_trading_decision", stub)

    conn = make_conn()
    portfolio_row = make_portfolio(conn)
    result = pipeline._maybe_run_deep_reflection(
        conn, make_app_config(), portfolio_row, make_nav_history(), make_metrics_result(), today=DUE_TODAY,
    )

    assert result is None
    assert len(stub.calls) == 2  # beide Aufrufe fanden statt, nur das Parsen des 2. schlug fehl
    rows = conn.execute(
        "SELECT * FROM decisions WHERE portfolio_id = ? AND model = 'deep_reflection'", (portfolio_row["id"],)
    ).fetchall()
    assert rows == []
