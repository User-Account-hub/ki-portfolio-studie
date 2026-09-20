"""Tests for src/data_quality.py's price-comparison and gap/outlier detection
(2026-09-17). Pure computation, no network - all inputs are injected directly.
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest

from src.data_quality import (
    DataQualityReport,
    StalePosition,
    build_report,
    compare_source_prices,
    detect_missing_trading_days,
    detect_outlier_moves,
    detect_stale_open_positions,
    select_price_comparison_sample,
)


# --- compare_source_prices ------------------------------------------------


def test_compare_source_prices_flags_deviation_above_threshold():
    yfinance_prices = {"AAPL": 100.0}
    alpaca_prices = {"AAPL": 102.0}  # +2%, über dem 1%-Default
    deviations = compare_source_prices(yfinance_prices, alpaca_prices)
    assert len(deviations) == 1
    assert deviations[0].symbol == "AAPL"
    assert deviations[0].deviation_pct == pytest.approx(0.02)


def test_compare_source_prices_ignores_deviation_within_threshold():
    yfinance_prices = {"AAPL": 100.0}
    alpaca_prices = {"AAPL": 100.5}  # +0.5%, unter dem 1%-Default
    assert compare_source_prices(yfinance_prices, alpaca_prices) == []


def test_compare_source_prices_only_compares_symbols_present_in_both_sources():
    yfinance_prices = {"AAPL": 100.0, "ONLY_YF": 50.0}
    alpaca_prices = {"AAPL": 100.0, "ONLY_ALPACA": 75.0}
    assert compare_source_prices(yfinance_prices, alpaca_prices) == []


def test_compare_source_prices_negative_deviation_also_flagged():
    yfinance_prices = {"AAPL": 100.0}
    alpaca_prices = {"AAPL": 95.0}  # -5%
    deviations = compare_source_prices(yfinance_prices, alpaca_prices)
    assert len(deviations) == 1
    assert deviations[0].deviation_pct == pytest.approx(-0.05)


def test_compare_source_prices_custom_threshold():
    yfinance_prices = {"AAPL": 100.0}
    alpaca_prices = {"AAPL": 103.0}  # +3%
    assert compare_source_prices(yfinance_prices, alpaca_prices, threshold_pct=0.05) == []
    assert len(compare_source_prices(yfinance_prices, alpaca_prices, threshold_pct=0.02)) == 1


# --- select_price_comparison_sample ---------------------------------------


def test_select_price_comparison_sample_returns_all_if_below_sample_size():
    symbols = ["AAPL", "MSFT", "NVDA"]
    assert select_price_comparison_sample(symbols, sample_size=10) == sorted(symbols)


def test_select_price_comparison_sample_is_deterministic_for_same_date():
    symbols = [f"SYM{i}" for i in range(30)]
    as_of = datetime.date(2026, 9, 17)
    first = select_price_comparison_sample(symbols, sample_size=10, as_of=as_of)
    second = select_price_comparison_sample(symbols, sample_size=10, as_of=as_of)
    assert first == second
    assert len(first) == 10


def test_select_price_comparison_sample_rotates_across_days():
    """Über genügend Tage hinweg müssen verschiedene Symbole an die Reihe
    kommen, statt immer dieselben (z.B. alphabetisch ersten)."""
    symbols = [f"SYM{i:02d}" for i in range(30)]
    day1 = select_price_comparison_sample(symbols, sample_size=10, as_of=datetime.date(2026, 9, 17))
    day2 = select_price_comparison_sample(symbols, sample_size=10, as_of=datetime.date(2026, 9, 18))
    assert day1 != day2


def test_select_price_comparison_sample_empty_input():
    assert select_price_comparison_sample([], sample_size=10) == []


# --- detect_missing_trading_days ------------------------------------------


def _series(dates: list[str], prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=pd.to_datetime(dates))


def test_detect_missing_trading_days_flags_gap_most_other_symbols_have():
    histories = {
        "A": _series(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], [1, 2, 3, 4]),
        "B": _series(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], [1, 2, 3, 4]),
        # C fehlt 01-07, obwohl A und B (100% der uebrigen Symbole) diesen Tag haben.
        "C": _series(["2026-01-05", "2026-01-06", "2026-01-08"], [1, 2, 4]),
    }
    issues = detect_missing_trading_days(histories)
    assert len(issues) == 1
    assert issues[0].symbol == "C"
    assert issues[0].kind == "missing_trading_day"
    assert "2026-01-07" in issues[0].detail


def test_detect_missing_trading_days_ignores_market_holiday():
    """Ein Tag, den ALLE Symbole nicht haben (echter Markt-Feiertag), darf
    nicht als Luecke gelten - er erreicht nie die Referenzkalender-Schwelle."""
    histories = {
        "A": _series(["2026-01-05", "2026-01-07"], [1, 3]),  # 01-06 fehlt allen
        "B": _series(["2026-01-05", "2026-01-07"], [1, 3]),
    }
    assert detect_missing_trading_days(histories) == []


def test_detect_missing_trading_days_ignores_gap_outside_own_history_range():
    """Ein spaeterer Listing-Beginn (IPO) ist keine Luecke, nur ein Tag
    INNERHALB der eigenen Start-/End-Spanne zaehlt."""
    histories = {
        "A": _series(["2026-01-05", "2026-01-06", "2026-01-07"], [1, 2, 3]),
        "B": _series(["2026-01-05", "2026-01-06", "2026-01-07"], [1, 2, 3]),
        # IPO erst am 01-07 - 01-05/01-06 fehlen, sind aber vor Listing-Beginn.
        "NEW_IPO": _series(["2026-01-07"], [10]),
    }
    assert detect_missing_trading_days(histories) == []


def test_detect_missing_trading_days_noop_with_single_symbol():
    histories = {"A": _series(["2026-01-05", "2026-01-07"], [1, 3])}
    assert detect_missing_trading_days(histories) == []


# --- detect_outlier_moves --------------------------------------------------


def test_detect_outlier_moves_flags_move_above_threshold():
    histories = {"A": _series(["2026-01-05", "2026-01-06"], [100.0, 160.0])}  # +60%
    issues = detect_outlier_moves(histories)
    assert len(issues) == 1
    assert issues[0].symbol == "A"
    assert issues[0].kind == "outlier_move"
    assert "+60.0%" in issues[0].detail


def test_detect_outlier_moves_ignores_normal_move():
    histories = {"A": _series(["2026-01-05", "2026-01-06"], [100.0, 105.0])}  # +5%
    assert detect_outlier_moves(histories) == []


def test_detect_outlier_moves_flags_large_drop_too():
    histories = {"A": _series(["2026-01-05", "2026-01-06"], [100.0, 40.0])}  # -60%
    issues = detect_outlier_moves(histories)
    assert len(issues) == 1
    assert "-60.0%" in issues[0].detail


def test_detect_outlier_moves_custom_threshold():
    histories = {"A": _series(["2026-01-05", "2026-01-06"], [100.0, 130.0])}  # +30%
    assert detect_outlier_moves(histories, threshold_pct=0.50) == []
    assert len(detect_outlier_moves(histories, threshold_pct=0.20)) == 1


def test_detect_outlier_moves_noop_with_single_data_point():
    histories = {"A": _series(["2026-01-05"], [100.0])}
    assert detect_outlier_moves(histories) == []


# --- detect_stale_open_positions (2026-09-21) -------------------------------


def make_position(**overrides) -> dict:
    defaults = dict(
        symbol="ACME", underlying_symbol=None, instrument_type="equity", side="long",
        quantity=10.0, avg_entry_price=100.0,
    )
    defaults.update(overrides)
    return defaults


def test_detect_stale_open_positions_flags_symbol_with_no_current_price():
    positions = [make_position(symbol="DELISTED")]
    stale = detect_stale_open_positions(positions, current_prices={})
    assert len(stale) == 1
    assert stale[0] == StalePosition(
        symbol="DELISTED", instrument_type="equity", side="long", quantity=10.0, avg_entry_price=100.0,
    )


def test_detect_stale_open_positions_not_flagged_when_own_symbol_has_price():
    positions = [make_position(symbol="AAPL")]
    assert detect_stale_open_positions(positions, current_prices={"AAPL": 190.0}) == []


def test_detect_stale_open_positions_falls_back_to_underlying_symbol():
    """Ein strukturiertes Produkt, dessen Basiswert weiterhin gehandelt wird,
    ist NICHT 'stale' - das ist der bestehende, gewollte Proxy-Mechanismus
    (siehe execution.resolve_price), keine fehlende Kurslage."""
    positions = [make_position(symbol="MINI-NVDA-LONG-1", underlying_symbol="NVDA", instrument_type="mini_future")]
    assert detect_stale_open_positions(positions, current_prices={"NVDA": 200.0}) == []


def test_detect_stale_open_positions_flagged_when_underlying_also_missing():
    """Selbst der Basiswert-Fallback findet keinen Kurs - z.B. wenn auch der
    Basiswert nicht mehr gehandelt wird."""
    positions = [make_position(symbol="MINI-NVDA-LONG-1", underlying_symbol="NVDA", instrument_type="mini_future")]
    stale = detect_stale_open_positions(positions, current_prices={})
    assert len(stale) == 1
    assert stale[0].symbol == "MINI-NVDA-LONG-1"


def test_detect_stale_open_positions_mixed_only_flags_affected():
    positions = [
        make_position(symbol="AAPL"),
        make_position(symbol="DELISTED"),
        make_position(symbol="MSFT"),
    ]
    stale = detect_stale_open_positions(positions, current_prices={"AAPL": 190.0, "MSFT": 410.0})
    assert [p.symbol for p in stale] == ["DELISTED"]


def test_detect_stale_open_positions_empty_when_no_open_positions():
    assert detect_stale_open_positions([], current_prices={}) == []


def test_detect_stale_open_positions_preserves_side_and_quantity():
    positions = [make_position(symbol="SHORTED", side="short", quantity=25.0, avg_entry_price=42.5)]
    stale = detect_stale_open_positions(positions, current_prices={})
    assert stale[0].side == "short"
    assert stale[0].quantity == 25.0
    assert stale[0].avg_entry_price == 42.5


# --- build_report / DataQualityReport --------------------------------------


def test_build_report_has_findings_true_when_any_check_flags_something():
    report = build_report(
        yfinance_prices={"AAPL": 100.0},
        alpaca_prices={"AAPL": 110.0},
        price_histories={},
    )
    assert report.has_findings is True
    assert len(report.price_deviations) == 1


def test_build_report_has_findings_false_when_clean():
    report = build_report(
        yfinance_prices={"AAPL": 100.0},
        alpaca_prices={"AAPL": 100.2},
        price_histories={"AAPL": _series(["2026-01-05", "2026-01-06"], [100.0, 100.2])},
    )
    assert report.has_findings is False


def test_build_report_has_findings_false_without_open_positions_argument():
    """Rueckwaertskompatibilitaet: bestehende Aufrufer, die `open_positions`
    (noch) nicht kennen, muessen weiterhin gueltige Reports mit leerer
    stale_positions-Liste erhalten."""
    report = build_report(yfinance_prices={}, alpaca_prices={}, price_histories={})
    assert report.stale_positions == []


def test_build_report_includes_stale_open_positions():
    report = build_report(
        yfinance_prices={"AAPL": 190.0},
        alpaca_prices={},
        price_histories={},
        open_positions=[make_position(symbol="AAPL"), make_position(symbol="DELISTED")],
    )
    assert report.has_findings is True
    assert [p.symbol for p in report.stale_positions] == ["DELISTED"]


def test_data_quality_report_log_lines_covers_all_three_categories():
    report = DataQualityReport(
        price_deviations=list(compare_source_prices({"AAPL": 100.0}, {"AAPL": 110.0})),
        missing_trading_days=detect_missing_trading_days(
            {
                "MSFT": _series(["2026-01-05", "2026-01-06", "2026-01-07"], [1, 2, 3]),
                "NVDA": _series(["2026-01-05", "2026-01-06", "2026-01-07"], [1, 2, 3]),
                # GAPSYM fehlt 01-06, das MSFT/NVDA (100% der uebrigen) haben.
                "GAPSYM": _series(["2026-01-05", "2026-01-07"], [1, 3]),
            }
        ),
        outlier_moves=detect_outlier_moves({"OUTLIER": _series(["2026-01-05", "2026-01-06"], [100.0, 200.0])}),
    )
    lines = report.log_lines()
    assert len(lines) == 3
    assert any("AAPL" in line for line in lines)
    assert any(line.startswith("GAPSYM:") and "fehlende" in line for line in lines)
    assert any(line.startswith("OUTLIER:") and "Tagesbewegung" in line for line in lines)


def test_data_quality_report_log_lines_includes_stale_positions():
    report = DataQualityReport(
        price_deviations=[],
        missing_trading_days=[],
        outlier_moves=[],
        stale_positions=detect_stale_open_positions([make_position(symbol="DELISTED")], current_prices={}),
    )
    lines = report.log_lines()
    assert len(lines) == 1
    assert lines[0].startswith("DELISTED:")
    assert "manuelle Prüfung nötig" in lines[0]
