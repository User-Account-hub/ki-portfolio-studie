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


# --- Fund #4 (17-Punkte-Audit): _run_short_stop_loss_sweep_safely -----------


def make_risk_config_for_sweep() -> SimpleNamespace:
    return SimpleNamespace(short_stop_loss_pct=-0.20, transaction_cost_pct_of_notional=0.001)


def test_run_short_stop_loss_sweep_safely_normal_case():
    conn = make_conn()
    portfolio = make_portfolio(conn)

    forced_actions, sweep_error = pipeline._run_short_stop_loss_sweep_safely(
        conn, portfolio, {}, make_risk_config_for_sweep(), broker_client=None,
    )

    assert forced_actions == []
    assert sweep_error is None


def test_run_short_stop_loss_sweep_safely_survives_sweep_failure(monkeypatch, caplog):
    """Regressionsschutz (v18, Fund #4): ein Fehler IM SWEEP SELBST (nicht
    nur bei einem einzelnen Pflicht-Cover) darf den Lauf nicht abstuerzen
    lassen - vorher lief die Exception ungefangen durch und verhinderte
    jeden Report fuer diesen Lauf."""
    conn = make_conn()
    portfolio = make_portfolio(conn)

    def _raise(*a, **k):
        raise RuntimeError("Broker down")

    monkeypatch.setattr(pipeline.execution, "run_short_stop_loss_sweep", _raise)

    with caplog.at_level("ERROR"):
        forced_actions, sweep_error = pipeline._run_short_stop_loss_sweep_safely(
            conn, portfolio, {}, make_risk_config_for_sweep(), broker_client=None,
        )

    assert forced_actions == []
    assert sweep_error == "Broker down"
    assert any("Short-Stop-Loss-Sweep fehlgeschlagen" in r.message for r in caplog.records)


# --- Fund #9 (17-Punkte-Audit): _run_data_quality_checks --------------------


def make_app_config_with_alpaca() -> SimpleNamespace:
    return SimpleNamespace(alpaca_api_key="x", alpaca_secret_key="x")


def test_run_data_quality_checks_builds_report_from_fetched_sources(monkeypatch):
    """Normalfall: Alpaca-Stichprobenkurs weicht > 1% vom yfinance-Kurs ab -
    muss als price_deviation im zurueckgegebenen Report landen."""
    monkeypatch.setattr(pipeline.broker_alpaca, "get_market_data_client", lambda *a, **k: object())
    monkeypatch.setattr(
        pipeline.broker_alpaca, "get_latest_trade_prices", lambda client, symbols: {"AAPL": 101.5}
    )
    monkeypatch.setattr(pipeline.data_fetch, "fetch_price_histories", lambda *a, **k: {})

    report = pipeline._run_data_quality_checks(
        make_app_config_with_alpaca(),
        tradable_symbols=["AAPL"],
        structured_product_symbols=set(),
        yfinance_prices={"AAPL": 100.0},
        open_position_rows=[],
    )

    assert len(report.price_deviations) == 1
    assert report.price_deviations[0].symbol == "AAPL"


def test_run_data_quality_checks_survives_alpaca_fetch_failure(monkeypatch):
    """Ein fehlschlagender Alpaca-Kursvergleich darf den Check nicht crashen -
    nur der Preisvergleich-Teil des Reports bleibt leer."""
    def _raise(*a, **k):
        raise RuntimeError("Alpaca down")

    monkeypatch.setattr(pipeline.broker_alpaca, "get_market_data_client", _raise)
    monkeypatch.setattr(pipeline.data_fetch, "fetch_price_histories", lambda *a, **k: {})

    report = pipeline._run_data_quality_checks(
        make_app_config_with_alpaca(),
        tradable_symbols=["AAPL"],
        structured_product_symbols=set(),
        yfinance_prices={"AAPL": 100.0},
        open_position_rows=[],
    )

    assert report.price_deviations == []


def test_run_data_quality_checks_survives_price_history_fetch_failure(monkeypatch):
    """Ein fehlschlagender Kurshistorien-Abruf darf den Check ebenfalls nicht
    crashen - nur der Luecken-/Ausreisser-Teil des Reports bleibt leer."""
    monkeypatch.setattr(pipeline.broker_alpaca, "get_market_data_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.broker_alpaca, "get_latest_trade_prices", lambda client, symbols: {})

    def _raise(*a, **k):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(pipeline.data_fetch, "fetch_price_histories", _raise)

    report = pipeline._run_data_quality_checks(
        make_app_config_with_alpaca(),
        tradable_symbols=["AAPL"],
        structured_product_symbols=set(),
        yfinance_prices={"AAPL": 100.0},
        open_position_rows=[],
    )

    assert report.missing_trading_days == []
    assert report.outlier_moves == []


def test_run_data_quality_checks_flags_stale_open_position(monkeypatch):
    """open_position_rows wird unveraendert an data_quality.build_report
    durchgereicht - eine offene Position ohne Kurs weder fuer ihr eigenes
    Symbol noch (hier: keinen) Basiswert muss als stale_position auftauchen,
    unabhaengig vom Alpaca-/Historien-Abruf."""
    conn = make_conn()
    portfolio = make_portfolio(conn)
    db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="DELISTED", instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10, fill_price=50.0,
    )
    open_rows = db.get_open_positions(conn, portfolio["id"])

    monkeypatch.setattr(pipeline.broker_alpaca, "get_market_data_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.broker_alpaca, "get_latest_trade_prices", lambda client, symbols: {})
    monkeypatch.setattr(pipeline.data_fetch, "fetch_price_histories", lambda *a, **k: {})

    report = pipeline._run_data_quality_checks(
        make_app_config_with_alpaca(),
        tradable_symbols=["AAPL"],
        structured_product_symbols=set(),
        yfinance_prices={"AAPL": 100.0},  # kein Kurs fuer DELISTED
        open_position_rows=open_rows,
    )

    assert len(report.stale_positions) == 1
    assert report.stale_positions[0].symbol == "DELISTED"


# --- Fund #9 (17-Punkte-Audit): _check_boundary_conditions ------------------


def test_check_boundary_conditions_marks_triggered_in_db_and_returns_both_lists():
    conn = make_conn()
    portfolio = make_portfolio(conn)
    position_id = db.upsert_open_position(
        conn, portfolio_id=portfolio["id"], symbol="NVDA", instrument_type="equity",
        underlying_symbol=None, side="long", delta_quantity=10, fill_price=100.0,
    )
    triggered_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio["id"], position_id=position_id, symbol="NVDA",
        description="Ausbruch ueber SMA50", check_type="price_above", threshold_price=120.0,
    )
    open_id = db.insert_boundary_condition(
        conn, portfolio_id=portfolio["id"], position_id=position_id, symbol="NVDA",
        description="Bruch unter Einstand", check_type="price_below", threshold_price=80.0,
    )

    triggered, still_open = pipeline._check_boundary_conditions(conn, portfolio["id"], {"NVDA": 125.0})

    assert [c.id for c in triggered] == [triggered_id]
    assert [c.id for c in still_open] == [open_id]

    # In der DB muss NUR die ausgeloeste Randbedingung als 'triggered' markiert
    # sein - die offene bleibt fuer den naechsten Lauf bestehen.
    remaining_open = db.get_open_boundary_conditions(conn, portfolio["id"])
    assert [r["id"] for r in remaining_open] == [open_id]


def test_check_boundary_conditions_returns_empty_lists_without_crashing_on_db_error(monkeypatch, caplog):
    """Analog zu _run_data_quality_checks: ein Fehler in der Pruefung selbst
    (z.B. DB-Problem) darf den Pipeline-Lauf nicht abbrechen."""
    conn = make_conn()
    portfolio = make_portfolio(conn)

    def _raise(*a, **k):
        raise RuntimeError("DB kaputt")

    monkeypatch.setattr(pipeline.db, "get_open_boundary_conditions", _raise)

    with caplog.at_level("ERROR"):
        triggered, still_open = pipeline._check_boundary_conditions(conn, portfolio["id"], {"NVDA": 100.0})

    assert triggered == []
    assert still_open == []
    assert any("Randbedingungs" in r.message for r in caplog.records)


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


def _guard_passed_marker(*args, **kwargs):
    raise RuntimeError("guard_passed")


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

    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _guard_passed_marker)

    with pytest.raises(RuntimeError, match="guard_passed"):
        pipeline.run()


# --- _fetch_news_context (Kap. 12.2, 2026-09-21) -----------------------------


def test_fetch_news_context_delegates_to_news_feed(monkeypatch):
    monkeypatch.setattr(pipeline.news_feed, "fetch_all_feeds", lambda: "sentinel-results")
    monkeypatch.setattr(pipeline.news_feed, "build_news_text_block", lambda results: f"block-for-{results}")
    assert pipeline._fetch_news_context() == "block-for-sentinel-results"


def test_fetch_news_context_returns_empty_string_on_unexpected_error(monkeypatch):
    """Wie bei den anderen weichen Kontext-Quellen (_check_upcoming_events,
    _fetch_universe_fundamentals): ein Fehler hier darf den Lauf nicht
    gefaehrden, nur diesen einen Kontext-Baustein fuer den Lauf entfallen
    lassen."""
    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline.news_feed, "fetch_all_feeds", _raise)
    assert pipeline._fetch_news_context() == ""


# --- Idempotenz-Sperre (2026-09-21, Kap. 12.7) -------------------------------


def _insert_todays_successful_decision(db_path: Path, model: str = "claude-sonnet-5") -> None:
    conn = sqlite3.connect(str(db_path))
    portfolio_id = conn.execute("SELECT id FROM portfolios WHERE name = 'test'").fetchone()[0]
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    conn.execute(
        """
        INSERT INTO decisions (portfolio_id, created_at, model, prompt, proposed_orders, approved, executed)
        VALUES (?, ?, ?, 'p', '[]', 1, 1)
        """,
        (portfolio_id, f"{today} 10:00:00", model),
    )
    conn.commit()
    conn.close()


def _make_file_db_without_open_positions(tmp_path) -> Path:
    db_path = _make_file_db_with_pilot_position(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE positions SET status = 'closed', quantity = 0")  # wie nach erfolgreichem Reset
    conn.commit()
    conn.close()
    return db_path


def test_is_force_rerun_requested_parameter_takes_precedence(monkeypatch):
    monkeypatch.setenv("FORCE_RERUN", "false")
    assert pipeline._is_force_rerun_requested(True) is True
    monkeypatch.setenv("FORCE_RERUN", "true")
    assert pipeline._is_force_rerun_requested(False) is False


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes"])
def test_is_force_rerun_requested_accepts_truthy_env_values(monkeypatch, value):
    monkeypatch.setenv("FORCE_RERUN", value)
    assert pipeline._is_force_rerun_requested(None) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "garbage"])
def test_is_force_rerun_requested_rejects_non_truthy_env_values(monkeypatch, value):
    monkeypatch.setenv("FORCE_RERUN", value)
    assert pipeline._is_force_rerun_requested(None) is False


def test_is_force_rerun_requested_false_when_env_unset(monkeypatch):
    monkeypatch.delenv("FORCE_RERUN", raising=False)
    assert pipeline._is_force_rerun_requested(None) is False


def test_run_aborts_cleanly_when_already_decided_today(tmp_path, monkeypatch):
    """Kernanforderung: eine bereits abgeschlossene heutige Handelsentscheidung
    muss einen zweiten Lauf sauber stoppen - kein Claude-Aufruf, kein
    Marktdaten-Abruf, kein zusaetzlicher Trade, aber ein dokumentierender
    pipeline_guard-Eintrag."""
    db_path = _make_file_db_without_open_positions(tmp_path)
    _insert_todays_successful_decision(db_path)
    fake_config = make_fake_app_config(db_path)

    monkeypatch.delenv("FORCE_RERUN", raising=False)
    monkeypatch.setattr(AppConfig, "load", lambda: fake_config)
    monkeypatch.setattr(pipeline.broker_alpaca, "get_trading_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "get_trading_decision", _fail_if_called)
    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _fail_if_called)

    pipeline.run()  # darf NICHT raisen

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    guard_decisions = conn.execute("SELECT * FROM decisions WHERE model = 'pipeline_guard'").fetchall()
    assert len(guard_decisions) == 1
    assert "abgeschlossene Handelsentscheidung" in guard_decisions[0]["rationale"]
    all_decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert all_decisions == 2  # die urspruengliche + genau der eine Guard-Eintrag, kein zweiter Claude-Trade
    conn.close()


def test_run_with_force_parameter_true_bypasses_idempotency_lock(tmp_path, monkeypatch):
    """Beweist, dass force=True ueber die Sperre hinauskommt - bricht bewusst
    beim naechsten Schritt (Marktdaten-Abruf) mit einer Marker-Exception ab,
    statt den kompletten weiteren Lauf zu mocken."""
    db_path = _make_file_db_without_open_positions(tmp_path)
    _insert_todays_successful_decision(db_path)
    fake_config = make_fake_app_config(db_path)

    monkeypatch.delenv("FORCE_RERUN", raising=False)
    monkeypatch.setattr(AppConfig, "load", lambda: fake_config)
    monkeypatch.setattr(pipeline.broker_alpaca, "get_trading_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _guard_passed_marker)

    with pytest.raises(RuntimeError, match="guard_passed"):
        pipeline.run(force=True)


def test_run_with_force_env_var_bypasses_idempotency_lock(tmp_path, monkeypatch):
    """Wie oben, aber ueber die Umgebungsvariable statt den Funktionsparameter -
    der Weg, den ein manueller GitHub-Actions-workflow_dispatch nutzen wuerde."""
    db_path = _make_file_db_without_open_positions(tmp_path)
    _insert_todays_successful_decision(db_path)
    fake_config = make_fake_app_config(db_path)

    monkeypatch.setenv("FORCE_RERUN", "true")
    monkeypatch.setattr(AppConfig, "load", lambda: fake_config)
    monkeypatch.setattr(pipeline.broker_alpaca, "get_trading_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _guard_passed_marker)

    with pytest.raises(RuntimeError, match="guard_passed"):
        pipeline.run()


def test_run_ignores_decision_from_a_previous_day(tmp_path, monkeypatch):
    """Eine erfolgreiche Entscheidung von GESTERN darf einen heutigen Lauf
    nicht blockieren."""
    db_path = _make_file_db_without_open_positions(tmp_path)
    conn = sqlite3.connect(str(db_path))
    portfolio_id = conn.execute("SELECT id FROM portfolios WHERE name = 'test'").fetchone()[0]
    yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    conn.execute(
        """
        INSERT INTO decisions (portfolio_id, created_at, model, prompt, proposed_orders, approved, executed)
        VALUES (?, ?, 'claude-sonnet-5', 'p', '[]', 1, 1)
        """,
        (portfolio_id, f"{yesterday} 10:00:00"),
    )
    conn.commit()
    conn.close()

    fake_config = make_fake_app_config(db_path)
    monkeypatch.setattr(AppConfig, "load", lambda: fake_config)
    monkeypatch.setattr(pipeline.broker_alpaca, "get_trading_client", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.data_fetch, "fetch_market_snapshots", _guard_passed_marker)

    with pytest.raises(RuntimeError, match="guard_passed"):
        pipeline.run()
