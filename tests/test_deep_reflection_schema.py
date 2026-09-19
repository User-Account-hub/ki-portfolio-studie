"""Tests for src/deep_reflection_schema.py's JSON parsing/validation -
mirrors tests/test_order_schema.py-style coverage (tolerant JSON extraction,
schema validation), just without expecting any order fields - plus the
2026-09-19 self-consistency check used exclusively by the monthly deep
reflection (see pipeline._maybe_run_deep_reflection).
"""
from __future__ import annotations

import pytest

from src.deep_reflection_schema import (
    DeepReflectionOutput,
    DeepReflectionParsingError,
    check_self_consistency,
    parse_reflection_from_json,
)

VALID_JSON = """{
  "theses_confirmed": ["NVDA Wide-Moat-These bestätigt"],
  "theses_falsified_or_overdue": ["CCJ Zyklusphasen-These überfällig"],
  "pattern_matching_concerns": "keine",
  "portfolio_stance_assessment": "kohärent",
  "reflection_commentary": "Alles im Rahmen der Methodik."
}"""


def test_parse_reflection_from_plain_json():
    result = parse_reflection_from_json(VALID_JSON)
    assert result.theses_confirmed == ["NVDA Wide-Moat-These bestätigt"]
    assert result.theses_falsified_or_overdue == ["CCJ Zyklusphasen-These überfällig"]
    assert result.pattern_matching_concerns == "keine"
    assert result.portfolio_stance_assessment == "kohärent"
    assert result.reflection_commentary == "Alles im Rahmen der Methodik."


def test_parse_reflection_from_fenced_markdown_json():
    fenced = f"```json\n{VALID_JSON}\n```"
    result = parse_reflection_from_json(fenced)
    assert result.reflection_commentary == "Alles im Rahmen der Methodik."


def test_parse_reflection_tolerates_surrounding_prose():
    wrapped = f"Hier ist meine Reflexion:\n\n{VALID_JSON}\n\nEnde."
    result = parse_reflection_from_json(wrapped)
    assert result.reflection_commentary == "Alles im Rahmen der Methodik."


def test_parse_reflection_uses_defaults_for_optional_fields():
    minimal = '{"reflection_commentary": "Kurz und knapp."}'
    result = parse_reflection_from_json(minimal)
    assert result.theses_confirmed == []
    assert result.theses_falsified_or_overdue == []
    assert result.pattern_matching_concerns == ""
    assert result.portfolio_stance_assessment == ""
    assert result.reflection_commentary == "Kurz und knapp."


def test_parse_reflection_missing_required_field_raises():
    """reflection_commentary ist das einzige Pflichtfeld - ohne es muss die
    Validierung fehlschlagen, nicht stillschweigend einen leeren String
    annehmen (das waere eine leere Reflexion, kein Parsing-Fehler)."""
    with pytest.raises(DeepReflectionParsingError):
        parse_reflection_from_json('{"theses_confirmed": []}')


def test_parse_reflection_no_json_object_raises():
    with pytest.raises(DeepReflectionParsingError, match="Keine JSON-Struktur"):
        parse_reflection_from_json("Das ist nur Fliesstext, kein JSON.")


def test_parse_reflection_invalid_json_raises():
    """Enthaelt sowohl '{' als auch '}' (die Extraktion greift), aber ist
    wegen des trailing comma kein gueltiges JSON - muss beim json.loads
    scheitern, nicht schon bei der Struktur-Erkennung."""
    with pytest.raises(DeepReflectionParsingError, match="Ungültiges JSON"):
        parse_reflection_from_json('{"reflection_commentary": "kaputt",}')


# --- check_self_consistency (2026-09-19) -------------------------------------


def make_reflection(**overrides) -> DeepReflectionOutput:
    defaults = dict(reflection_commentary="Kommentar")
    defaults.update(overrides)
    return DeepReflectionOutput(**defaults)


def test_self_consistency_identical_reflections_are_consistent():
    a = make_reflection(
        theses_confirmed=["NVDA-These bestätigt"],
        theses_falsified_or_overdue=["CCJ-These überfällig"],
    )
    b = make_reflection(
        theses_confirmed=["NVDA-These bestätigt"],
        theses_falsified_or_overdue=["CCJ-These überfällig"],
    )
    result = check_self_consistency(a, b)
    assert result.consistent
    assert result.mismatch_details == []


def test_self_consistency_ignores_order_of_theses():
    a = make_reflection(theses_confirmed=["A-These", "B-These"])
    b = make_reflection(theses_confirmed=["B-These", "A-These"])
    assert check_self_consistency(a, b).consistent


def test_self_consistency_ignores_case_and_whitespace():
    a = make_reflection(theses_confirmed=["  NVDA-These bestätigt  "])
    b = make_reflection(theses_confirmed=["nvda-these bestätigt"])
    assert check_self_consistency(a, b).consistent


def test_self_consistency_detects_mismatch_in_confirmed_theses():
    a = make_reflection(theses_confirmed=["NVDA-These bestätigt"])
    b = make_reflection(theses_confirmed=[])
    result = check_self_consistency(a, b)
    assert not result.consistent
    assert any("theses_confirmed" in d and "1. Aufruf" in d for d in result.mismatch_details)


def test_self_consistency_detects_mismatch_in_falsified_theses():
    a = make_reflection(theses_falsified_or_overdue=[])
    b = make_reflection(theses_falsified_or_overdue=["CCJ-These überfällig"])
    result = check_self_consistency(a, b)
    assert not result.consistent
    assert any("theses_falsified_or_overdue" in d and "2. Aufruf" in d for d in result.mismatch_details)


def test_self_consistency_detects_direct_contradiction():
    """Dieselbe These einmal bestaetigt, einmal widerlegt - der klarste
    Fall, den dieser Check ueberhaupt abdecken soll."""
    a = make_reflection(theses_confirmed=["NVDA-These"])
    b = make_reflection(theses_falsified_or_overdue=["NVDA-These"])
    result = check_self_consistency(a, b)
    assert not result.consistent
    assert len(result.mismatch_details) == 2  # eine Abweichung je betroffenem Feld


def test_self_consistency_empty_reflections_are_consistent():
    assert check_self_consistency(make_reflection(), make_reflection()).consistent
