"""Pydantic model + parser for the monthly deep-reflection prompt's JSON
output (Thesis Kap. 6.12.3) - purely analytical, no orders. See
deep_reflection_prompt.py for the prompt text and trigger logic."""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field


class DeepReflectionOutput(BaseModel):
    theses_confirmed: list[str] = Field(default_factory=list)
    theses_falsified_or_overdue: list[str] = Field(default_factory=list)
    pattern_matching_concerns: str = ""
    portfolio_stance_assessment: str = ""
    reflection_commentary: str


class DeepReflectionParsingError(Exception):
    """Raised when the reflection response cannot be parsed into a valid DeepReflectionOutput."""


def parse_reflection_from_json(raw_text: str) -> DeepReflectionOutput:
    """Extracts the first top-level JSON object from raw_text and validates
    it - same tolerant approach as order_schema.parse_orders_from_json."""
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        raise DeepReflectionParsingError("Keine JSON-Struktur in der Antwort gefunden.")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise DeepReflectionParsingError(f"Ungültiges JSON: {exc}") from exc
    try:
        return DeepReflectionOutput.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise DeepReflectionParsingError(f"Schema-Validierung fehlgeschlagen: {exc}") from exc


def _extract_json_object(text: str) -> Optional[str]:
    """Mirrors order_schema._extract_json_object - kept as its own small
    copy rather than a shared import, to keep this schema module self-
    contained. Same tolerant behaviour: fenced ```json block first, else
    the outermost {...} span."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
