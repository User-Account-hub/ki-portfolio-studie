"""Automatische, regelbasierte Markt-Phasen-Klassifikation je Titel
(2026-09-20) - Bull/Bear/Seitwärts aus gleitenden Durchschnitten (SMA20/50)
und Volatilität, ausschliesslich aus bereits vorhandenen Kursdaten
(data_fetch.MarketSnapshot, dieselben Werte wie im Report/Prompt) - kein
zusätzlicher Datenabruf.

Dient als unabhängiger, mechanischer Gegencheck zu Claudes v7-SYSTEM_PROMPT-
Anforderung 1 (Zyklus-Position, Kap. 3: Akkumulation/Aufmerksamkeit/Manie/
Crash/Rückkehr zum Mittel, siehe order_schema.CyclePosition) - NICHT als
weiteres Guardrail-Veto: ein Widerspruch zwischen Claudes Einschätzung und
der Regel-Klassifikation wird nur dokumentiert (siehe execution.py/
reporting.py), keine Order wird deswegen abgelehnt. Ein 3-Phasen-Regelwerk
(Bull/Bear/Seitwärts) und ein 5-Phasen-Zyklusmodell sind unterschiedliche
Abstraktionen - die Zuordnung `PLAUSIBLE_PHASES_FOR_CYCLE_POSITION` unten ist
eine bewusst grobe, dokumentierte Heuristik, kein exaktes Mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.data_fetch import MarketSnapshot
from src.order_schema import CyclePosition

# Volatilitaets-skalierte Bandbreite um SMA50, innerhalb derer der Titel als
# "kein klarer Trend" (Seitwaerts) gilt statt jede kleine Abweichung schon
# als Bull/Bear zu werten - höher bei volatileren Titeln (die schwanken
# ohnehin staerker, ohne dass das automatisch einen Trend bedeutet), mit
# einer festen Mindestbandbreite fuer Titel ohne (oder mit sehr niedriger)
# Volatilitaetsangabe.
DEFAULT_SIDEWAYS_BAND_VOLATILITY_MULTIPLIER = 0.5
MIN_SIDEWAYS_BAND_PCT = 0.03


class MarketPhase(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    # Zu wenig Daten fuer eine Klassifikation (z.B. neu gelistetes Symbol
    # ohne volle SMA50-Historie) - bewusst kein Default wie SIDEWAYS, damit
    # "nicht geprueft" nicht wie ein durchgefuehrter Befund aussieht.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MarketPhaseClassification:
    symbol: str
    phase: MarketPhase
    price: float
    sma20: float | None
    sma50: float | None
    volatility_20d_annualized: float | None


@dataclass(frozen=True)
class MarketPhaseContradiction:
    symbol: str
    claude_cycle_position: CyclePosition
    rule_based_phase: MarketPhase
    detail: str


def classify_market_phase(
    price: float,
    sma20: float | None,
    sma50: float | None,
    volatility_20d_annualized: float | None,
    sideways_band_volatility_multiplier: float = DEFAULT_SIDEWAYS_BAND_VOLATILITY_MULTIPLIER,
) -> MarketPhase:
    """Regelbasierte 3-Phasen-Klassifikation für EIN Symbol.

    UNKNOWN ohne sma20/sma50 (zu kurze Historie) oder sma50 == 0.

    SIDEWAYS, wenn der Kurs innerhalb einer volatilitätsskalierten Bandbreite
    um sma50 liegt (siehe Modul-Konstanten oben) - unabhängig davon, ob
    sma20 gerade knapp über oder unter sma50 liegt, das wäre bei einem Titel
    ohne klaren Trend ohnehin nur Rauschen.

    Ausserhalb der Bandbreite: BULL, wenn sowohl der aktuelle Kurs als auch
    sma20 über sma50 liegen (kurz- UND länger-fristiger Trend stimmen
    überein); BEAR, wenn beide darunter liegen. Ein gemischtes Signal
    (z.B. Kurs > sma50, aber sma20 < sma50 - ein möglicher Trendwechsel im
    Gange) wird konservativ als SIDEWAYS gewertet statt geraten.
    """
    if sma20 is None or sma50 is None or sma50 == 0:
        return MarketPhase.UNKNOWN

    band = max(MIN_SIDEWAYS_BAND_PCT, sideways_band_volatility_multiplier * (volatility_20d_annualized or 0.0))
    deviation_from_sma50 = (price - sma50) / sma50
    if abs(deviation_from_sma50) <= band:
        return MarketPhase.SIDEWAYS
    if price > sma50 and sma20 > sma50:
        return MarketPhase.BULL
    if price < sma50 and sma20 < sma50:
        return MarketPhase.BEAR
    return MarketPhase.SIDEWAYS


def classify_universe_market_phases(snapshots: dict[str, MarketSnapshot]) -> dict[str, MarketPhaseClassification]:
    """Wendet classify_market_phase auf jedes Symbol in `snapshots`
    (data_fetch.fetch_market_snapshots' Rückgabe) an - keine eigene
    Datenabfrage, reine Ableitung aus bereits geladenen Werten."""
    return {
        symbol: MarketPhaseClassification(
            symbol=symbol,
            phase=classify_market_phase(
                snap.last_price, snap.sma20, snap.sma50, snap.volatility_20d_annualized
            ),
            price=snap.last_price,
            sma20=snap.sma20,
            sma50=snap.sma50,
            volatility_20d_annualized=snap.volatility_20d_annualized,
        )
        for symbol, snap in snapshots.items()
    }


# Grobe, dokumentierte Heuristik (siehe Modul-Docstring) für plausible
# Regel-Phasen je Claude-Zyklusposition - EIN Widerspruch bedeutet nicht
# zwingend, dass Claude falsch liegt (die Regel-Klassifikation ist ihrerseits
# nur ein einfaches SMA/Vola-Signal), sondern ist ein Hinweis, der im Report
# dokumentiert wird:
#   ACCUMULATION (Basisbildung, typischerweise nach einem Rückgang oder in
#     ruhiger Seitwärtsphase, noch KEIN bestätigter Aufwärtstrend) ->
#     plausibel bei SIDEWAYS/BEAR, Widerspruch bei BULL.
#   ATTENTION (aufkommendes Interesse, früher bis mittlerer Aufwärtstrend) ->
#     plausibel bei BULL/SIDEWAYS (im Übergang), Widerspruch bei BEAR.
#   MANIA (euphorische Übertreibung) -> plausibel NUR bei BULL.
#   CRASH (scharfer Einbruch) -> plausibel NUR bei BEAR.
#   REVERSION_TO_MEAN (Rückkehr zum Mittel, per Definition kein bestehender
#     starker Trend mehr) -> plausibel NUR bei SIDEWAYS.
PLAUSIBLE_PHASES_FOR_CYCLE_POSITION: dict[CyclePosition, frozenset[MarketPhase]] = {
    CyclePosition.ACCUMULATION: frozenset({MarketPhase.SIDEWAYS, MarketPhase.BEAR}),
    CyclePosition.ATTENTION: frozenset({MarketPhase.BULL, MarketPhase.SIDEWAYS}),
    CyclePosition.MANIA: frozenset({MarketPhase.BULL}),
    CyclePosition.CRASH: frozenset({MarketPhase.BEAR}),
    CyclePosition.REVERSION_TO_MEAN: frozenset({MarketPhase.SIDEWAYS}),
}


def check_cycle_position_against_market_phase(
    symbol: str,
    claude_cycle_position: CyclePosition | None,
    rule_based_phase: MarketPhase,
) -> MarketPhaseContradiction | None:
    """None, wenn kein Vergleich möglich/nötig ist (Claude hat keine
    Zyklusposition angegeben - optionales Feld, siehe order_schema.py) oder
    keine Regel-Klassifikation vorliegt (UNKNOWN, zu kurze Historie) - eine
    fehlende Prüfung ist etwas anderes als eine durchgeführte Prüfung ohne
    Befund. Sonst None bei Übereinstimmung, sonst ein
    MarketPhaseContradiction mit Begründungstext fürs Reporting.

    AUSDRÜCKLICH kein Veto (siehe Modul-Docstring) - der Rückgabewert wird
    nur dokumentiert, nie zur Order-Ablehnung verwendet.
    """
    if claude_cycle_position is None or rule_based_phase == MarketPhase.UNKNOWN:
        return None
    plausible = PLAUSIBLE_PHASES_FOR_CYCLE_POSITION[claude_cycle_position]
    if rule_based_phase in plausible:
        return None
    return MarketPhaseContradiction(
        symbol=symbol,
        claude_cycle_position=claude_cycle_position,
        rule_based_phase=rule_based_phase,
        detail=(
            f"{symbol}: Claude ordnet die Zyklus-Position als '{claude_cycle_position.value}' ein, "
            f"die regelbasierte Marktphasen-Klassifikation (SMA20/50 + Volatilität) sieht das Symbol "
            f"aber in '{rule_based_phase.value}'."
        ),
    )
