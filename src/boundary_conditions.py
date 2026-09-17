"""Randbedingungs-Tracking (Thesis Kap. 7, 2026-09-17): jede von Claude bei
einer Kauf-/Short-Empfehlung genannte Randbedingung (siehe
order_schema.BoundaryCondition) wird gespeichert (db.insert_boundary_condition,
aufgerufen aus execution.py) und bei jedem Lauf gegen die aktuellen Kurse
geprüft, solange die zugehörige Position offen ist.

Rein dokumentarisch: ein Trigger löst KEINE automatische Order/Konsequenz
aus - siehe pipeline.py, wo das Ergebnis nur geloggt, in den Report
geschrieben und dem nächsten Tagesprompt als Kontext mitgegeben wird.

Nur "price_above"/"price_below"-Randbedingungen sind mechanisch prüfbar
(deterministisch anhand der ohnehin geladenen current_prices, kein
zusätzlicher Claude-Call). "qualitative" Randbedingungen (z.B. Makro-/
Earnings-Ereignisse) werden NICHT automatisch geprüft - sie bleiben offen
und werden bei jedem Lauf unverändert weitergeführt/dokumentiert, statt eine
Automatisierung vorzutäuschen, die es nicht gibt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryConditionCheck:
    id: int
    position_id: int
    symbol: str
    description: str
    check_type: str  # "price_above" | "price_below" | "qualitative"
    threshold_price: float | None


def evaluate_boundary_conditions(
    open_conditions: list[BoundaryConditionCheck],
    current_prices: dict[str, float],
) -> tuple[list[BoundaryConditionCheck], list[BoundaryConditionCheck]]:
    """Returns (triggered, still_open).

    `still_open` enthält sowohl qualitative Randbedingungen (nie automatisch
    geprüft) als auch price-basierte, deren Symbol in diesem Lauf keinen
    aktuellen Kurs hat - letztere werden übersprungen (bleiben offen), NICHT
    stillschweigend als ausgelöst gewertet, nur weil kein Kurs vorliegt.
    """
    triggered: list[BoundaryConditionCheck] = []
    still_open: list[BoundaryConditionCheck] = []
    for cond in open_conditions:
        if cond.check_type == "qualitative":
            still_open.append(cond)
            continue
        price = current_prices.get(cond.symbol)
        if price is None:
            still_open.append(cond)
            continue
        if cond.check_type == "price_above" and price > cond.threshold_price:
            triggered.append(cond)
        elif cond.check_type == "price_below" and price < cond.threshold_price:
            triggered.append(cond)
        else:
            still_open.append(cond)
    return triggered, still_open
