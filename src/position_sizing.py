"""Volatilitätsadjustierte Positionsgrössen-Skalierung (2026-09-19).

Nutzt die pro Symbol ohnehin bereits aus den yfinance-Kurshistorien
berechnete annualisierte 20-Tage-Volatilität (rollierende Standardabweichung
der Tagesrenditen, siehe data_fetch.MarketSnapshot.volatility_20d_annualized
und data_fetch._annualized_volatility) - kein zusätzlicher Datenabruf nötig.

Faktor je Symbol = Universums-Durchschnittsvolatilität / Symbol-Volatilität,
geclippt auf [min_factor, max_factor]. Unterdurchschnittlich volatile Titel
werden dadurch hochskaliert (mehr Risikobudget für dieselbe Kap.-6.8-Grenze),
überdurchschnittlich volatile Titel runterskaliert - eine einfache Näherung
an inverse-Volatility/Risk-Parity-Sizing.

Wird in execution.py auf die von Claude vorgeschlagene Grösse einer
positionsaufbauenden Order (buy/short) angewendet, BEVOR risk_guardrails.
evaluate_order läuft. Das ist bewusst nur eine Verfeinerung INNERHALB der
bestehenden Kap.-6.8-Limiten, kein zusätzliches Veto: die resultierende
(skalierte) Order durchläuft danach dieselbe Guardrail-Prüfung wie jede
andere Order und kann keines der dortigen Limits aushebeln - sie kann eine
an sich zu grosse Position also nicht "durchschleusen", nur innerhalb eines
engen Bandes (Default 0.5x-1.5x, siehe risk_config.yaml) verkleinern oder
vergrössern.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MIN_SCALING_FACTOR = 0.5
DEFAULT_MAX_SCALING_FACTOR = 1.5


@dataclass(frozen=True)
class VolatilityScaling:
    symbol: str
    annualized_volatility: float
    universe_avg_volatility: float
    scaling_factor: float


def compute_scaling_factors(
    volatilities: dict[str, float],
    min_factor: float = DEFAULT_MIN_SCALING_FACTOR,
    max_factor: float = DEFAULT_MAX_SCALING_FACTOR,
) -> dict[str, VolatilityScaling]:
    """Berechnet den Skalierungsfaktor je Symbol relativ zur Ø-Volatilität
    des übergebenen Universums.

    `volatilities` enthält nur Symbole mit tatsächlich berechenbarer
    20-Tage-Volatilität (siehe pipeline.py's Aufrufstelle) - Symbole ohne
    Eintrag erhalten hier bewusst KEINEN Default-Faktor (z.B. 1.0), sondern
    fehlen im Ergebnis-Dict; der Caller wendet dann keine Skalierung an,
    statt eine ungeprüfte Annahme zu treffen. Symbole mit Volatilität <= 0
    (z.B. eine konstante Kurshistorie) werden analog übersprungen, um eine
    Division durch Null zu vermeiden.
    """
    usable = {s: v for s, v in volatilities.items() if v is not None and v > 0}
    if not usable:
        return {}
    avg_vol = sum(usable.values()) / len(usable)
    result: dict[str, VolatilityScaling] = {}
    for symbol, vol in usable.items():
        factor = max(min_factor, min(max_factor, avg_vol / vol))
        result[symbol] = VolatilityScaling(
            symbol=symbol,
            annualized_volatility=vol,
            universe_avg_volatility=avg_vol,
            scaling_factor=factor,
        )
    return result


def scale_order_size(
    quantity: float | None,
    notional: float | None,
    scaling: VolatilityScaling,
) -> tuple[float | None, float | None]:
    """Skaliert quantity und/oder notional (je nachdem, was die Order trägt -
    ProposedOrder verlangt genau eines von beiden, siehe order_schema.py)
    mit `scaling.scaling_factor`."""
    factor = scaling.scaling_factor
    new_quantity = quantity * factor if quantity is not None else None
    new_notional = notional * factor if notional is not None else None
    return new_quantity, new_notional
