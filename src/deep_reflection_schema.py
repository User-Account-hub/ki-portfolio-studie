"""Pydantic model + parser for the monthly deep-reflection prompt's JSON
output (Thesis Kap. 6.12.3) - purely analytical, no orders. See
deep_reflection_prompt.py for the prompt text and trigger logic."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


class DeepReflectionOutput(BaseModel):
    theses_confirmed: list[str] = Field(default_factory=list)
    theses_falsified_or_overdue: list[str] = Field(default_factory=list)
    pattern_matching_concerns: str = ""
    portfolio_stance_assessment: str = ""
    reflection_commentary: str


@dataclass(frozen=True)
class SelfConsistencyCheckResult:
    """Ergebnis von check_self_consistency (siehe dort für die Begründung,
    warum dieser Vergleich EXKLUSIV für die monatliche Tiefenreflexion
    existiert, nicht für den täglichen Ablauf)."""
    consistent: bool
    mismatch_details: list[str] = field(default_factory=list)


def _normalized_thesis_set(theses: list[str]) -> set[str]:
    """Trim/lowercase, damit rein kosmetische Abweichungen (Gross-/
    Kleinschreibung, führende/nachgestellte Leerzeichen) nicht fälschlich
    als inhaltliche Abweichung gewertet werden. Erkennt WEITERHIN keine
    semantisch gleichwertige, aber anders formulierte These als identisch -
    das ist eine bewusste, dokumentierte Grenze dieses einfachen
    Mengenvergleichs (siehe check_self_consistency's Docstring)."""
    return {t.strip().lower() for t in theses if t.strip()}


def check_self_consistency(
    first: DeepReflectionOutput, second: DeepReflectionOutput
) -> SelfConsistencyCheckResult:
    """Selbstkonsistenz-Prüfung EXKLUSIV für die monatliche Tiefenreflexion
    (2026-09-19, Kap. 6.12.3) - aus Kostengründen NICHT im täglichen Ablauf:
    Claude wird für dieselbe Reflexions-Periode zweimal mit identischem
    Prompt aufgerufen (siehe pipeline._maybe_run_deep_reflection), und die
    Kernaussagen - hier: die beiden These-Listen `theses_confirmed` und
    `theses_falsified_or_overdue` - werden verglichen.

    Bewusst ein einfacher, deterministischer Mengenvergleich (nach
    Normalisierung, siehe _normalized_thesis_set) statt eines weiteren
    LLM-Aufrufs zur "semantischen" Übereinstimmungsprüfung - das würde das
    Problem (unzuverlässige LLM-Ausgabe) mit demselben Werkzeugtyp zu lösen
    versuchen, das geprüft werden soll. Konsequenz: zwei inhaltlich
    identische, aber unterschiedlich formulierte Thesen gelten als
    Abweichung. Das ist eine bewusst in Kauf genommene Grenze - dieser Check
    ist ein grobes Frühwarnsignal für grundlegend widersprüchliche
    Einschätzungen (z.B. eine These einmal bestätigt, einmal widerlegt),
    kein Nachweis inhaltlicher Übereinstimmung im Detail.

    KEINE automatische Konfliktlösung: bei einer Abweichung wird hier nur
    dokumentiert, WAS abweicht - nicht entschieden, welche der beiden
    Antworten "richtig" ist (siehe pipeline._maybe_run_deep_reflection für
    die Persistierung beider Antworten und reporting.py für die
    Report-Darstellung beider bei einer Abweichung).
    """
    details: list[str] = []
    for field_name in ("theses_confirmed", "theses_falsified_or_overdue"):
        first_set = _normalized_thesis_set(getattr(first, field_name))
        second_set = _normalized_thesis_set(getattr(second, field_name))
        if first_set == second_set:
            continue
        only_first = sorted(first_set - second_set)
        only_second = sorted(second_set - first_set)
        if only_first:
            details.append(f"{field_name}: nur im 1. Aufruf genannt: {only_first}")
        if only_second:
            details.append(f"{field_name}: nur im 2. Aufruf genannt: {only_second}")
    return SelfConsistencyCheckResult(consistent=not details, mismatch_details=details)


@dataclass(frozen=True)
class DeepReflectionRunResult:
    """Kapselt EINEN Tiefenreflexions-Lauf inkl. der Selbstkonsistenz-Prüfung
    (siehe check_self_consistency). `primary` (die erste der beiden
    Antworten) dient unverändert als Grundlage für
    db.get_latest_reflection/den nächsten Tagesprompt - eine willkürliche,
    aber deterministische Wahl, KEINE inhaltliche Konfliktlösung. `secondary`
    und `consistency` existieren ausschliesslich zur Dokumentation (siehe
    reporting.py), nicht zur weiteren Verarbeitung."""
    primary: DeepReflectionOutput
    secondary: DeepReflectionOutput
    consistency: SelfConsistencyCheckResult


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
