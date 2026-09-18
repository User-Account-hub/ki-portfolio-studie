"""Pydantic model + parser for the historical-stress-test commentary
(Thesis Kap. 11.2). `contamination_caveat` is a REQUIRED field, not
optional - the whole point of this feature is to force an explicit,
per-run self-assessment of hindsight-bias risk, not to hope Claude mentions
it unprompted. See stress_test_prompt.py for the system prompt that asks
for it and src/stress_test.py's module docstring for the full rationale.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel


class StressTestCommentary(BaseModel):
    resilience_assessment: str
    most_vulnerable_exposure: str
    contamination_caveat: str


class StressTestParsingError(Exception):
    """Raised when the stress-test commentary response cannot be parsed into a valid StressTestCommentary."""


def parse_stress_test_commentary_from_json(raw_text: str) -> StressTestCommentary:
    """Extracts the first top-level JSON object from raw_text and validates
    it - same tolerant approach as order_schema.parse_orders_from_json."""
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        raise StressTestParsingError("Keine JSON-Struktur in der Antwort gefunden.")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StressTestParsingError(f"Ungültiges JSON: {exc}") from exc
    try:
        return StressTestCommentary.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise StressTestParsingError(f"Schema-Validierung fehlgeschlagen: {exc}") from exc


def _extract_json_object(text: str) -> Optional[str]:
    """Mirrors order_schema._extract_json_object - kept as its own small
    copy rather than a shared import, to keep this schema module self-
    contained."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
