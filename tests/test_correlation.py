"""Tests for src/correlation.py's rolling correlation matrix, per-BUY-order
warning check (documentary only, no veto), and scipy-based cluster count.
Pure functions, no network/DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.correlation import (
    CorrelationWarning,
    check_correlation_to_existing_positions,
    compute_correlation_clusters,
    compute_correlation_matrix,
)


def _price_series(n_days: int, seed: int, trend: float = 0.0) -> pd.Series:
    """Deterministische Pseudo-Zufallskursreihe (reproduzierbar ueber `seed`)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=trend, scale=0.01, size=n_days)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.Series(prices, index=dates)


def _identical_series(base: pd.Series, noise_seed: int | None = None) -> pd.Series:
    if noise_seed is None:
        return base.copy()
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=1e-6, size=len(base))
    return base * (1 + noise)


# --- compute_correlation_matrix ------------------------------------------------


def test_correlation_matrix_perfectly_correlated_symbols():
    base = _price_series(100, seed=1)
    histories = {"A": base, "B": base.copy()}
    matrix = compute_correlation_matrix(histories, window_days=60)
    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    assert matrix.loc["A", "A"] == pytest.approx(1.0)


def test_correlation_matrix_is_symmetric():
    a = _price_series(100, seed=1)
    b = _price_series(100, seed=2)
    c = _price_series(100, seed=3)
    matrix = compute_correlation_matrix({"A": a, "B": b, "C": c}, window_days=60)
    assert matrix.loc["A", "B"] == pytest.approx(matrix.loc["B", "A"])
    assert matrix.loc["A", "C"] == pytest.approx(matrix.loc["C", "A"])


def test_correlation_matrix_inverse_series_gives_negative_one():
    base = _price_series(100, seed=1)
    returns = base.pct_change().dropna()
    inverse_prices = 100.0 * np.cumprod(1 - returns.values)
    inverse = pd.Series(inverse_prices, index=returns.index)
    matrix = compute_correlation_matrix({"A": base.iloc[1:], "B": inverse}, window_days=60)
    assert matrix.loc["A", "B"] == pytest.approx(-1.0, abs=1e-6)


def test_correlation_matrix_excludes_symbol_with_insufficient_history():
    long_history = _price_series(100, seed=1)
    short_history = _price_series(30, seed=2)  # < window_days + 1
    matrix = compute_correlation_matrix({"A": long_history, "B": short_history}, window_days=60)
    assert "B" not in matrix.columns
    assert matrix.empty  # nur noch 1 Symbol uebrig -> keine Matrix moeglich


def test_correlation_matrix_empty_with_fewer_than_two_symbols():
    assert compute_correlation_matrix({}, window_days=60).empty
    assert compute_correlation_matrix({"A": _price_series(100, seed=1)}, window_days=60).empty


# --- check_correlation_to_existing_positions -----------------------------------


def test_check_correlation_flags_high_correlation():
    base = _price_series(100, seed=1)
    matrix = compute_correlation_matrix({"A": base, "B": base.copy()}, window_days=60)
    warnings = check_correlation_to_existing_positions("A", ["B"], matrix, threshold=0.85)
    assert warnings == [CorrelationWarning(candidate_symbol="A", existing_symbol="B", correlation=pytest.approx(1.0))]


def test_check_correlation_ignores_low_correlation():
    a = _price_series(100, seed=1)
    b = _price_series(100, seed=42)
    matrix = compute_correlation_matrix({"A": a, "B": b}, window_days=60)
    # Zwei unabhaengige Zufallsreihen liegen praktisch nie ueber 0.85.
    assert matrix.loc["A", "B"] < 0.85
    assert check_correlation_to_existing_positions("A", ["B"], matrix, threshold=0.85) == []


def test_check_correlation_ignores_strong_negative_correlation():
    """Bewusst: >Schwelle heisst positiv > threshold, KEIN Betrag - eine
    stark negative Korrelation ist aus Redundanz-Sicht unproblematisch."""
    base = _price_series(100, seed=1)
    returns = base.pct_change().dropna()
    inverse_prices = 100.0 * np.cumprod(1 - returns.values)
    inverse = pd.Series(inverse_prices, index=returns.index)
    matrix = compute_correlation_matrix({"A": base.iloc[1:], "B": inverse}, window_days=60)
    assert check_correlation_to_existing_positions("A", ["B"], matrix, threshold=0.85) == []


def test_check_correlation_skips_candidate_itself():
    base = _price_series(100, seed=1)
    matrix = compute_correlation_matrix({"A": base, "B": base.copy()}, window_days=60)
    assert check_correlation_to_existing_positions("A", ["A", "B"], matrix, threshold=0.85) == [
        CorrelationWarning(candidate_symbol="A", existing_symbol="B", correlation=pytest.approx(1.0))
    ]


def test_check_correlation_skips_symbol_missing_from_matrix():
    base = _price_series(100, seed=1)
    matrix = compute_correlation_matrix({"A": base, "B": base.copy()}, window_days=60)
    assert check_correlation_to_existing_positions("A", ["NOT_IN_MATRIX"], matrix, threshold=0.85) == []


def test_check_correlation_candidate_not_in_matrix_returns_empty():
    base = _price_series(100, seed=1)
    matrix = compute_correlation_matrix({"A": base, "B": base.copy()}, window_days=60)
    assert check_correlation_to_existing_positions("UNKNOWN", ["A"], matrix, threshold=0.85) == []


def test_check_correlation_empty_matrix_returns_empty():
    assert check_correlation_to_existing_positions("A", ["B"], pd.DataFrame(), threshold=0.85) == []


# --- compute_correlation_clusters ----------------------------------------------


def test_clusters_zero_symbols_returns_zero():
    matrix = compute_correlation_matrix({"A": _price_series(100, 1), "B": _price_series(100, 2)}, 60)
    assert compute_correlation_clusters([], matrix) == 0


def test_clusters_single_symbol_returns_one():
    matrix = compute_correlation_matrix({"A": _price_series(100, 1), "B": _price_series(100, 2)}, 60)
    assert compute_correlation_clusters(["A"], matrix) == 1


def test_clusters_two_identical_symbols_form_one_cluster():
    base = _price_series(100, seed=1)
    matrix = compute_correlation_matrix({"A": base, "B": base.copy()}, window_days=60)
    assert compute_correlation_clusters(["A", "B"], matrix) == 1


def test_clusters_two_independent_symbols_form_two_clusters():
    a = _price_series(100, seed=1)
    b = _price_series(100, seed=99)
    matrix = compute_correlation_matrix({"A": a, "B": b}, window_days=60)
    assert matrix.loc["A", "B"] < 0.85
    assert compute_correlation_clusters(["A", "B"], matrix) == 2


def test_clusters_two_groups_of_correlated_symbols():
    """A/B stark korreliert (ein Cluster), C unabhaengig (eigenes Cluster) -> 2 Cluster."""
    a = _price_series(100, seed=1)
    b = _identical_series(a, noise_seed=7)
    c = _price_series(100, seed=123)
    matrix = compute_correlation_matrix({"A": a, "B": b, "C": c}, window_days=60)
    assert matrix.loc["A", "B"] > 0.85
    assert matrix.loc["A", "C"] < 0.85
    assert compute_correlation_clusters(["A", "B", "C"], matrix) == 2


def test_clusters_symbol_without_matrix_entry_is_excluded():
    matrix = compute_correlation_matrix({"A": _price_series(100, 1), "B": _price_series(100, 2)}, 60)
    # "UNKNOWN" hat keinen Matrix-Eintrag - darf weder crashen noch als eigenes Cluster zaehlen.
    result_with = compute_correlation_clusters(["A", "B", "UNKNOWN"], matrix)
    result_without = compute_correlation_clusters(["A", "B"], matrix)
    assert result_with == result_without
