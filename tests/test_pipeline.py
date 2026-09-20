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
from src.config import AppConfig
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


# --- _pilot_phase_positions_still_open (2026-09-21, EINMALIGE Uebergangs- ---
# --- Sicherheitspruefung Pilotphase -> offizielle Studie, siehe pipeline.py --
# --- Block-Kommentar und RESET_2026-09-21.md) --------------------------------


def make_position_row(opened_at: str) -> dict:
    return {"opened_at": opened_at}


def test_pilot_phase_check_flags_position_opened_before_study_start():
    positions = [make_position_row("2026-09-08 16:50:16")]
    assert pipeline._pilot_phase_positions_still_open(positions) == positions


def test_pilot_phase_check_ignores_position_opened_after_study_start():
    positions = [make_position_row("2026-09-22 10:00:00")]
    assert pipeline._pilot_phase_positions_still_open(positions) == []


def test_pilot_phase_check_boundary_exactly_at_study_start_is_not_flagged():
    """OFFICIAL_STUDY_START selbst (Mitternacht) gilt bereits als Tag 1 - eine
    Position, die an diesem Tag eroeffnet wurde (zwingend NACH Mitternacht,
    der Handel beginnt fruehestens 09:30 ET), ist ein legitimer Trade der
    offiziellen Studie, kein Pilotphase-Rest."""
    positions = [make_position_row(f"{OFFICIAL_STUDY_START.date()} 09:30:00")]
    assert pipeline._pilot_phase_positions_still_open(positions) == []


def test_pilot_phase_check_empty_when_no_open_positions():
    assert pipeline._pilot_phase_positions_still_open([]) == []


def test_pilot_phase_check_mixed_returns_only_pilot_positions():
    old = make_position_row("2026-09-08 16:50:16")
    new = make_position_row("2026-09-22 10:00:00")
    assert pipeline._pilot_phase_positions_still_open([old, new]) == [old]


def test_pilot_phase_check_becomes_permanently_empty_after_reset():
    """Kernanforderung: sobald der Reset durchgefuehrt wurde (Pilotphase-
    Positionen also nicht mehr in open_position_rows enthalten, weil
    geschlossen), muss der Check dauerhaft leer bleiben - unabhaengig davon,
    an welchem Tag der Reset nachgeholt wird."""
    fresh_position_weeks_later = make_position_row("2026-10-15 11:00:00")
    assert pipeline._pilot_phase_positions_still_open([fresh_position_weeks_later]) == []


# --- pipeline.run(): frueher, sauberer Abbruch bei ausstehendem Reset -------


def _make_file_db_with_pilot_position(tmp_path) -> Path:
    """Ein datei-basiertes SQLite-DB (nicht :memory:) - noetig, weil
    pipeline.run() ueber db.get_connection(app_config.db_path) eine EIGENE
    Verbindung zu genau diesem Pfad oeffnet, siehe Modul-Docstring."""
    db_path = tmp_path / "portfolio.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO portfolios (name, currency, initial_cash_balance, cash_balance) VALUES (?, ?, ?, ?)",
        ("test", "USD", 1_000_000.0, 78_328.42),
    )
    portfolio_id = conn.execute("SELECT id FROM portfolios WHERE name = 'test'").fetchone()[0]
    conn.execute(
        """
        INSERT INTO positions (portfolio_id, symbol, instrument_type, side, quantity, avg_entry_price, opened_at)
        VALUES (?, 'NVDA', 'equity', 'long', 216.838139955, 226.010092, '2026-09-08 16:50:16')
        """,
        (portfolio_id,),
    )
    conn.commit()
    conn.close()
    return db_path


def make_fake_app_config(db_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        anthropic_api_key="x",
        claude_model="claude-sonnet-5",
        alpaca_api_key="x",
        alpaca_secret_key="x",
        alpaca_base_url="https://paper-api.alpaca.markets",
        db_path=str(db_path),
        portfolio_name="test",
        initial_cash_balance=1_000_000.0,
        portfolio_currency="USD",
        watchlist_path="./config/watchlist.yaml",
        risk_config_path="./config/risk_config.yaml",
        reports_dir=str(db_path.parent / "reports"),
        risk_free_rate_annual=0.04,
    )


def _fail_if_called(*args, **kwargs):
    raise AssertionError("darf bei ausstehendem Reset nicht aufgerufen werden")


def test_run_aborts_cleanly_when_pilot_phase_positions_still_open(tmp_path, monkeypatch):
    """Integrationstest der eigentlichen Verdrahtung in run() (nicht nur der
    reinen Hilfsfunktion oben): kein Claude-Aufruf, kein Marktdaten-Abruf
    (echter 'sauberer Abbruch', nicht nur ein uebersprungener Handelsteil),
    genau ein dokumentierender Decision-Eintrag, DB unveraendert."""
    db_path = _make_file_db_with_pilot_position(tmp_path)
    fake_config = make_fake_app_config(db_path)

    monkeypatch.setattr(AppConfig, "load", lambda: fake_config)
    monkeypatch.setattr(pipeline.broker_alpaca, "get_trading_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "get_trading_decision", _fail_if_called)
    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _fail_if_called)

    pipeline.run()  # darf NICHT raisen - ein erkannter ausstehender Reset ist kein Fehler

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    decisions = conn.execute("SELECT * FROM decisions WHERE model = 'pipeline_guard'").fetchall()
    assert len(decisions) == 1
    assert "Reset noch nicht durchgeführt" in decisions[0]["rationale"]
    assert decisions[0]["approved"] == 0
    assert decisions[0]["executed"] == 0

    portfolio = conn.execute("SELECT * FROM portfolios WHERE name = 'test'").fetchone()
    assert portfolio["cash_balance"] == pytest.approx(78_328.42)  # unveraendert
    open_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'").fetchone()[0]
    assert open_positions == 1  # unveraendert - der Guard fasst die Position selbst nicht an
    conn.close()


def test_run_proceeds_past_guard_when_no_pilot_phase_positions_remain(tmp_path, monkeypatch):
    """Beweist, dass der Guard nach einem erfolgreichen Reset (keine offene
    Pilotphase-Position mehr) NICHT mehr greift - der Lauf muss ueber ihn
    hinauskommen und den naechsten Schritt (Marktdaten-Abruf) erreichen.
    Bricht dort bewusst mit einer eigenen Marker-Exception ab, statt den
    kompletten weiteren Lauf (yfinance/Alpaca/Claude) zu mocken - das genuegt
    als Beweis, dass der Guard passiert wurde."""
    db_path = _make_file_db_with_pilot_position(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE positions SET status = 'closed', quantity = 0")  # wie nach erfolgreichem Reset
    conn.commit()
    conn.close()

    fake_config = make_fake_app_config(db_path)
    monkeypatch.setattr(AppConfig, "load", lambda: fake_config)
    monkeypatch.setattr(pipeline.broker_alpaca, "get_trading_client", lambda *a, **k: object())

    def _guard_passed_marker(*args, **kwargs):
        raise RuntimeError("guard_passed")

    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _guard_passed_marker)

    with pytest.raises(RuntimeError, match="guard_passed"):
        pipeline.run()
