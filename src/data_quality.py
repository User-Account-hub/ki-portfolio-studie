"""Datenqualitäts-Checks bei jedem Pipeline-Lauf (2026-09-17):

(1) Kursvergleich yfinance vs. Alpaca (Stichprobe) - Abweichungen über einem
    Schwellwert (Default 1%) werden dokumentiert.
(2) Einfache Lücken-/Ausreisser-Erkennung in den yfinance-Kurshistorien:
    fehlende Handelstage innerhalb der eigenen Historie eines Symbols, und
    Tagesbewegungen über einem unrealistischen Schwellwert (Default ±50%),
    die eher auf einen Datenfehler (Split/Fetch-Glitch) als auf eine echte
    Kursbewegung hindeuten.
(3) Fehlende Kursdaten für eine noch offene Position (2026-09-21) - z.B.
    Delisting/Übernahme: siehe detect_stale_open_positions.

Reine Berechnungslogik, kein Netzwerkzugriff - die Werte (yfinance-/Alpaca-
Preise, Kurshistorien) werden vom Aufrufer (siehe pipeline.py) beschafft und
hier nur ausgewertet. Auffälligkeiten sind ein Beobachtungssignal für Log/
Report, KEIN Abbruchgrund - siehe pipeline.py, wo dieser Check bewusst nie
eine Exception weiterreicht, die den Lauf stoppen würde. Das gilt
AUSDRÜCKLICH auch für (3): fehlende Kursdaten lösen KEINE automatische Order
(z.B. eine Zwangsschliessung) aus - zu riskant für einen Automatismus, siehe
detect_stale_open_positions' Docstring - sondern werden nur klar markiert,
damit ein Mensch das manuell prüft.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import pandas as pd

DEFAULT_PRICE_DEVIATION_THRESHOLD = 0.01  # 1%
DEFAULT_OUTLIER_MOVE_THRESHOLD = 0.50  # ±50%
DEFAULT_SAMPLE_SIZE = 10
# Ein Handelstag zaehlt nur dann als "erwartet" (Referenzkalender fuer die
# Luecken-Erkennung), wenn mindestens diese Quote der Symbole im Universum
# an diesem Tag tatsaechlich einen Kurs hat - filtert Markt-Feiertage (an
# denen ALLEN Symbolen der Tag fehlt) automatisch heraus, ohne einen festen
# Feiertagskalender pflegen zu muessen.
DEFAULT_MIN_REFERENCE_COVERAGE = 0.5


@dataclass(frozen=True)
class PriceDeviation:
    symbol: str
    yfinance_price: float
    alpaca_price: float
    deviation_pct: float


@dataclass(frozen=True)
class DataQualityIssue:
    symbol: str
    kind: str  # "missing_trading_day" | "outlier_move"
    detail: str


@dataclass(frozen=True)
class StalePosition:
    """Eine noch offene Position, für die weder ihr eigenes Symbol noch (bei
    strukturierten Produkten) ihr Basiswert einen aktuellen Kurs liefert -
    siehe detect_stale_open_positions."""
    symbol: str
    instrument_type: str
    side: str
    quantity: float
    avg_entry_price: float


@dataclass(frozen=True)
class DataQualityReport:
    price_deviations: list[PriceDeviation]
    missing_trading_days: list[DataQualityIssue]
    outlier_moves: list[DataQualityIssue]
    # 2026-09-21 - siehe detect_stale_open_positions. Default statt Pflichtfeld,
    # damit bestehende Aufrufer/Tests ohne dieses Argument gueltig bleiben.
    stale_positions: list[StalePosition] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(
            self.price_deviations or self.missing_trading_days or self.outlier_moves or self.stale_positions
        )

    def log_lines(self) -> list[str]:
        """Eine Zeile pro Auffaelligkeit, fuer log.warning je Zeile (siehe
        pipeline.py) - dokumentiert, bricht nichts ab."""
        lines = [
            f"Kursabweichung {d.symbol}: yfinance={d.yfinance_price:.2f} "
            f"Alpaca={d.alpaca_price:.2f} ({d.deviation_pct:+.2%})"
            for d in self.price_deviations
        ]
        lines += [f"{issue.symbol}: {issue.detail}" for issue in self.missing_trading_days]
        lines += [f"{issue.symbol}: {issue.detail}" for issue in self.outlier_moves]
        lines += [
            f"{p.symbol}: Kursdaten fehlen (offene {p.side}-Position, {p.quantity:.4f} Stück, "
            f"Ø-Einstand {p.avg_entry_price:.2f}) - manuelle Prüfung nötig (Delisting/Übernahme?)."
            for p in self.stale_positions
        ]
        return lines


def compare_source_prices(
    yfinance_prices: dict[str, float],
    alpaca_prices: dict[str, float],
    threshold_pct: float = DEFAULT_PRICE_DEVIATION_THRESHOLD,
) -> list[PriceDeviation]:
    """Vergleicht nur die Symbole, fuer die BEIDE Quellen einen Preis liefern
    (welche Symbole das sind - z.B. eine Stichprobe statt des vollen
    Universums - entscheidet der Aufrufer). Meldet jede Abweichung, deren
    Betrag `threshold_pct` uebersteigt.

    Bekannte Einschraenkung (dokumentiert, kein Blocker fuer diesen einfachen
    Check): yfinance liefert den letzten TAGESSCHLUSSKURS, Alpaca den
    zuletzt gehandelten TRADE-Preis - ausserhalb der reguraeren Handelszeit
    (oder bei starker Kursbewegung zwischen den beiden Abrufen) koennen echte,
    harmlose Abweichungen > 1% auftreten, nicht nur Datenfehler.
    """
    deviations = []
    for symbol in sorted(set(yfinance_prices) & set(alpaca_prices)):
        yf_price = yfinance_prices[symbol]
        alpaca_price = alpaca_prices[symbol]
        if yf_price <= 0:
            continue
        deviation_pct = (alpaca_price - yf_price) / yf_price
        if abs(deviation_pct) > threshold_pct:
            deviations.append(PriceDeviation(symbol, yf_price, alpaca_price, deviation_pct))
    return deviations


def select_price_comparison_sample(
    symbols: list[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    as_of: datetime.date | None = None,
) -> list[str]:
    """Deterministische, taeglich rotierende Stichprobe (kein Zufall, daher
    reproduzierbar) - ueber genuegend Laeufe hinweg deckt das trotzdem das
    gesamte Universum ab, statt bei jedem Lauf dieselben (z.B. alphabetisch
    ersten) Symbole zu pruefen."""
    if not symbols or sample_size <= 0:
        return []
    ordered = sorted(set(symbols))
    if len(ordered) <= sample_size:
        return ordered
    as_of = as_of or datetime.date.today()
    offset = as_of.toordinal() % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    return rotated[:sample_size]


def detect_missing_trading_days(
    price_histories: dict[str, pd.Series],
    min_reference_coverage: float = DEFAULT_MIN_REFERENCE_COVERAGE,
) -> list[DataQualityIssue]:
    """Referenzkalender = alle Handelstage, die mindestens
    `min_reference_coverage` der Symbole im Universum aufweisen (statt eines
    starren Mo-Fr-Rasters, das an jedem Markt-Feiertag faelschlich anschlagen
    wuerde - an einem echten Feiertag fehlt der Tag ALLEN Symbolen und faellt
    dadurch automatisch unter die Schwelle).

    Ein Symbol wird nur fuer Luecken INNERHALB seiner eigenen Historie
    gemeldet (zwischen seinem ersten und letzten bekannten Kurs) - ein
    spaeterer Listing-Beginn (IPO) oder ein frueheres Ende ist keine Luecke.
    """
    if len(price_histories) < 2:
        return []

    date_counts: dict[pd.Timestamp, int] = {}
    for series in price_histories.values():
        for d in series.index:
            date_counts[d] = date_counts.get(d, 0) + 1
    n_symbols = len(price_histories)
    reference_dates = {d for d, count in date_counts.items() if count / n_symbols >= min_reference_coverage}

    issues = []
    for symbol, series in price_histories.items():
        if series.empty:
            continue
        symbol_dates = set(series.index)
        start, end = series.index.min(), series.index.max()
        expected = {d for d in reference_dates if start <= d <= end}
        missing = sorted(expected - symbol_dates)
        if missing:
            example = missing[0].date().isoformat()
            issues.append(
                DataQualityIssue(
                    symbol=symbol,
                    kind="missing_trading_day",
                    detail=(
                        f"{len(missing)} fehlende(r) Handelstag(e) innerhalb der eigenen "
                        f"Kurshistorie (z.B. {example}), obwohl an diesem Tag mindestens "
                        f"{min_reference_coverage:.0%} der uebrigen Symbole gehandelt wurden."
                    ),
                )
            )
    return issues


def detect_outlier_moves(
    price_histories: dict[str, pd.Series],
    threshold_pct: float = DEFAULT_OUTLIER_MOVE_THRESHOLD,
) -> list[DataQualityIssue]:
    """Tagesbewegungen mit |Rendite| > `threshold_pct` - bei reguraeren
    Aktien/ETFs (kein Hebelprodukt) so unrealistisch, dass ein Datenfehler
    (z.B. ein nicht bereinigter Split, ein Fetch-Glitch) wahrscheinlicher ist
    als eine echte Kursbewegung."""
    issues = []
    for symbol, series in price_histories.items():
        if len(series) < 2:
            continue
        returns = series.sort_index().pct_change().dropna()
        for date, move in returns[returns.abs() > threshold_pct].items():
            issues.append(
                DataQualityIssue(
                    symbol=symbol,
                    kind="outlier_move",
                    detail=(
                        f"Tagesbewegung {move:+.1%} am {date.date().isoformat()} - "
                        f"über dem {threshold_pct:.0%}-Schwellwert, möglicher Datenfehler "
                        "statt echte Kursbewegung."
                    ),
                )
            )
    return issues


def detect_stale_open_positions(
    open_positions,
    current_prices: dict[str, float],
) -> list[StalePosition]:
    """Erkennt eine noch offene Position, für die WEDER ihr eigenes Symbol
    NOCH (falls vorhanden) ihr Basiswert einen aktuellen Kurs in
    `current_prices` liefert - typischerweise ein Delisting oder eine
    Übernahme, bei der yfinance (die alleinige Quelle von `current_prices`,
    siehe pipeline.py) keine Kurse mehr für das Symbol zurückgibt.

    Dieselbe Fallback-Reihenfolge wie execution.resolve_price/
    risk_guardrails.compute_nav (eigenes Symbol zuerst, sonst Basiswert bei
    strukturierten Produkten) - ein Treffer hier bedeutet also: selbst diese
    bestehenden Fallbacks würden keinen Kurs finden. Für ein strukturiertes
    Produkt, dessen Basiswert weiterhin gehandelt wird, wird NICHTS
    gemeldet (das ist der bereits bestehende, gewollte Proxy-Mechanismus,
    keine fehlende Kurslage).

    AUSDRÜCKLICH KEINE automatische Order (z.B. eine Zwangsschliessung) -
    zu riskant für einen Automatismus (siehe Modul-Docstring). Nur eine
    klare Markierung für Log (DataQualityReport.log_lines) und Report
    (reporting.py), damit ein Mensch das manuell prüft.

    `open_positions` sind sqlite3.Row-artige Objekte mit den Spalten
    `symbol`, `underlying_symbol`, `instrument_type`, `side`, `quantity`,
    `avg_entry_price` (siehe db.get_open_positions)."""
    stale = []
    for p in open_positions:
        symbol = p["symbol"]
        underlying = p["underlying_symbol"]
        has_price = symbol in current_prices or (underlying and underlying in current_prices)
        if not has_price:
            stale.append(
                StalePosition(
                    symbol=symbol,
                    instrument_type=p["instrument_type"],
                    side=p["side"],
                    quantity=p["quantity"],
                    avg_entry_price=p["avg_entry_price"],
                )
            )
    return stale


def build_report(
    yfinance_prices: dict[str, float],
    alpaca_prices: dict[str, float],
    price_histories: dict[str, pd.Series],
    open_positions=None,
    price_deviation_threshold: float = DEFAULT_PRICE_DEVIATION_THRESHOLD,
    outlier_move_threshold: float = DEFAULT_OUTLIER_MOVE_THRESHOLD,
    min_reference_coverage: float = DEFAULT_MIN_REFERENCE_COVERAGE,
) -> DataQualityReport:
    return DataQualityReport(
        price_deviations=compare_source_prices(yfinance_prices, alpaca_prices, price_deviation_threshold),
        missing_trading_days=detect_missing_trading_days(price_histories, min_reference_coverage),
        outlier_moves=detect_outlier_moves(price_histories, outlier_move_threshold),
        stale_positions=detect_stale_open_positions(open_positions or [], yfinance_prices),
    )
