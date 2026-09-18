"""Korrelations-/Redundanz-Check: eine rollierende 60-Tage-Korrelations-
matrix der Tagesrenditen über das gesamte Anlage-Universum, die (a) vor
jeder BUY-Order den Kandidaten gegen jede bestehende Position prüft und (b)
die Anzahl der Korrelations-Cluster im aktuellen Portfolio ermittelt.

Ausdrücklich KEIN Guardrail: anders als die harten Kap.-6.8-Limiten
(risk_guardrails.py) löst eine hohe Korrelation (>0.85) keine Ablehnung
aus - nur eine Log-Zeile und einen Report-Eintrag. Das ist eine neue
Beobachtungsgrösse, kein zusätzliches Veto (siehe execution.py für die
Einbindung: die Prüfung läuft NACH der Guardrail-Freigabe, beeinflusst sie
aber nicht).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DEFAULT_CORRELATION_WINDOW_DAYS = 60
DEFAULT_CORRELATION_WARNING_THRESHOLD = 0.85
# Distanz = 1 - Korrelation; an denselben Schwellenwert wie die Warnung
# gekoppelt (0.15 = 1 - 0.85), damit "ein Cluster" und "eine Warnung"
# dieselbe wirtschaftliche Bedeutung haben (durchschnittliche Korrelation
# innerhalb eines Clusters > 0.85, bei "average"-Linkage).
DEFAULT_CLUSTER_DISTANCE_THRESHOLD = 1.0 - DEFAULT_CORRELATION_WARNING_THRESHOLD


@dataclass(frozen=True)
class CorrelationWarning:
    candidate_symbol: str
    existing_symbol: str
    correlation: float


def compute_correlation_matrix(
    price_histories: dict[str, pd.Series],
    window_days: int = DEFAULT_CORRELATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Korrelationsmatrix der Tagesrenditen über die letzten `window_days`
    HANDELSTAGE (nicht Kalendertage), gemeinsam für alle Symbole mit
    ausreichender Historie.

    Symbole mit weniger als `window_days` + 1 Kursen (zu kurz für auch nur
    eine volle Fensterbreite an Renditen) werden VORHER ausgeschlossen,
    nicht mit NaN aufgefüllt - ein Symbol mit halbwegs kurzer Historie soll
    nicht so aussehen, als wäre es geprüft und einfach unkorreliert.
    Verbleibende Lücken (z.B. leicht abweichende Handelskalender) werden
    von `DataFrame.corr()` paarweise über die jeweils gemeinsamen
    Datenpunkte behandelt (Pandas-Standardverhalten).

    Liefert eine leere DataFrame, falls weniger als zwei Symbole übrig
    bleiben (keine Korrelation zwischen weniger als zwei Reihen möglich).
    """
    returns = {
        symbol: series.sort_index().pct_change().dropna()
        for symbol, series in price_histories.items()
        if len(series) >= window_days + 1
    }
    if len(returns) < 2:
        return pd.DataFrame()
    returns_df = pd.DataFrame(returns).tail(window_days)
    return returns_df.corr()


def check_correlation_to_existing_positions(
    candidate_symbol: str,
    existing_position_symbols: list[str],
    correlation_matrix: pd.DataFrame,
    threshold: float = DEFAULT_CORRELATION_WARNING_THRESHOLD,
) -> list[CorrelationWarning]:
    """Rein dokumentarisch (siehe Modul-Docstring) - kein Veto. Meldet jede
    bestehende Position, deren Korrelation zum Kandidaten `threshold`
    ÜBERSCHREITET (echtes >, keine Betragsbildung - eine stark NEGATIVE
    Korrelation ist aus Redundanz-Sicht unproblematisch, im Gegenteil).

    Symbole ohne Eintrag in `correlation_matrix` (zu wenig Historie, siehe
    compute_correlation_matrix) werden übersprungen, nicht als
    "unkorreliert" gewertet - eine fehlende Prüfung ist etwas anderes als
    eine durchgeführte Prüfung mit niedrigem Ergebnis.
    """
    if candidate_symbol not in correlation_matrix.columns:
        return []
    warnings: list[CorrelationWarning] = []
    for existing_symbol in existing_position_symbols:
        if existing_symbol == candidate_symbol or existing_symbol not in correlation_matrix.columns:
            continue
        corr = correlation_matrix.loc[candidate_symbol, existing_symbol]
        if pd.notna(corr) and corr > threshold:
            warnings.append(CorrelationWarning(candidate_symbol, existing_symbol, float(corr)))
    return warnings


def compute_correlation_clusters(
    symbols: list[str],
    correlation_matrix: pd.DataFrame,
    distance_threshold: float = DEFAULT_CLUSTER_DISTANCE_THRESHOLD,
) -> int:
    """Einfaches hierarchisches Clustering (scipy, "average"-Linkage) über
    die Distanz 1 - Korrelation, geschnitten bei `distance_threshold`.

    Symbole ohne Matrixeintrag (zu wenig Historie) werden ausgeschlossen -
    sie zählen weder als eigenes Cluster noch verzerren sie die
    Distanzberechnung der übrigen. 0 offene/geprüfte Symbole ergeben 0
    Cluster, genau 1 Symbol trivial 1 Cluster (Clustering braucht
    mindestens 2 Punkte).
    """
    usable = [s for s in symbols if s in correlation_matrix.columns]
    if len(usable) == 0:
        return 0
    if len(usable) == 1:
        return 1

    sub = correlation_matrix.loc[usable, usable]
    # Fehlende paarweise Korrelationen (z.B. zu wenig gemeinsame Historie
    # zwischen zwei bestimmten Symbolen) als maximale Distanz (unkorreliert)
    # behandeln, statt linkage() an NaN scheitern zu lassen.
    distance = (1.0 - sub.fillna(0.0)).clip(lower=0.0)
    # `.values`/`.to_numpy()` kann bei neueren Pandas-Versionen ein
    # schreibgeschuetztes Array liefern (z.B. nach .clip()) - `copy=True`
    # erzwingt ein beschreibbares Array fuer fill_diagonal.
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0.0)
    condensed = squareform(distance_values, checks=False)
    z = linkage(condensed, method="average")
    cluster_labels = fcluster(z, t=distance_threshold, criterion="distance")
    return len(set(cluster_labels))
