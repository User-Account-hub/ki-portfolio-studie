"""Tests for src/stress_test_schema.py's JSON parsing/validation - notably
that `contamination_caveat` is a REQUIRED field (Thesis Kap. 11.2's whole
point is forcing this self-assessment every time, not hoping for it).
"""
from __future__ import annotations

import pytest

from src.stress_test_schema import (
    StressTestParsingError,
    parse_stress_test_commentary_from_json,
)

VALID_JSON = """{
  "resilience_assessment": "Portfolio zeigt hohe Konzentration im KI-Halbleiter-Segment.",
  "most_vulnerable_exposure": "NVDA-Position, stark korreliert mit dem Gesamtmarkt.",
  "contamination_caveat": "Ich kenne den tatsaechlichen Verlauf von NVDA 2020 aus Trainingsdaten."
}"""


def test_parse_commentary_from_plain_json():
    result = parse_stress_test_commentary_from_json(VALID_JSON)
    assert "Konzentration" in result.resilience_assessment
    assert "NVDA" in result.most_vulnerable_exposure
    assert "Trainingsdaten" in result.contamination_caveat


def test_parse_commentary_from_fenced_markdown_json():
    fenced = f"```json\n{VALID_JSON}\n```"
    result = parse_stress_test_commentary_from_json(fenced)
    assert "Trainingsdaten" in result.contamination_caveat


def test_parse_commentary_tolerates_surrounding_prose():
    wrapped = f"Hier ist meine Analyse:\n\n{VALID_JSON}\n\nEnde."
    result = parse_stress_test_commentary_from_json(wrapped)
    assert "Trainingsdaten" in result.contamination_caveat


def test_parse_commentary_missing_contamination_caveat_raises():
    """Kernaussage von Kap. 11.2: contamination_caveat ist PFLICHT, keine
    optionale Ergaenzung - eine Antwort ohne dieses Feld muss scheitern."""
    payload = '{"resilience_assessment": "ok", "most_vulnerable_exposure": "ok"}'
    with pytest.raises(StressTestParsingError):
        parse_stress_test_commentary_from_json(payload)


def test_parse_commentary_missing_resilience_assessment_raises():
    payload = '{"most_vulnerable_exposure": "ok", "contamination_caveat": "ok"}'
    with pytest.raises(StressTestParsingError):
        parse_stress_test_commentary_from_json(payload)


def test_parse_commentary_no_json_object_raises():
    with pytest.raises(StressTestParsingError, match="Keine JSON-Struktur"):
        parse_stress_test_commentary_from_json("Nur Fliesstext, kein JSON.")


def test_parse_commentary_invalid_json_raises():
    with pytest.raises(StressTestParsingError, match="Ungültiges JSON"):
        parse_stress_test_commentary_from_json('{"resilience_assessment": "kaputt",}')
