"""Kap. 12.2 der Thesis ("Plattform- und Werkzeugwahl"): "Definierte RSS-/
News-API-Feeds, automatisch aggregiert zu einem identischen Textblock je
Berichtszyklus." Das Prinzip war seit Beginn vorregistriert, aber nie
umgesetzt - dieses Modul ist die ERSTE tatsächliche Implementierung
(eingeführt 2026-09-21, unmittelbar VOR dem offiziellen Studienstart,
OFFICIAL_STUDY_START in deep_reflection_prompt.py). Das ist ausdrücklich KEIN
Regimewechsel während der laufenden Studie, sondern das erstmalige Einlösen
eines von Anfang an vorgesehenen, aber bislang nicht gebauten Bausteins - die
Pilotphase (Kap. 6.3) ist ohnehin nicht Teil der offiziellen Auswertung.

Die Feed-Liste (NEWS_FEEDS) ist ab dem offiziellen Studienstart FEST UND
UNVERÄNDERLICH, wie in Kap. 12.2 gefordert ("identisch je Berichtszyklus").
Eine spätere Änderung der Liste wäre ein methodisch relevanter Regimewechsel
während der laufenden Studie und müsste - wie jede andere Prompt-/Methodik-
Änderung - explizit begründet und dokumentiert werden, nicht stillschweigend
vorgenommen werden.

Fünf Feeds, alle am 2026-09-21 live per HTTP verifiziert (Status 200, echter
RSS-Inhalt, kein API-Key nötig):
  - CNBC "US Top News and Analysis" (breite Marktnachrichten)
  - CNBC "Tech" (das Anhang-A-Universum, Kap. 6.7, ist stark AI-/Halbleiter-
    lastig - ein reiner Markt-Feed deckt das schlecht ab)
  - MarketWatch "Top Stories" (redaktionell unabhängige Zweitquelle, Dow
    Jones statt NBCUniversal)
  - Federal Reserve "Monetary Policy" Press Releases (liefert echte Inhalte
    zu den bereits in event_calendar.py hardcodierten FOMC-Terminen, statt
    nur das Datum zu kennen)
  - Yahoo Finance "News" (fünfte, breite Quelle - Yahoo hat eigene RSS-Feeds
    in der Vergangenheit mehrfach abgeschaltet/verschoben; diese konkrete
    URL war am 2026-09-21 live und lieferte aktuelle Einträge)

Bewusst NICHT NewsAPI.org oder vergleichbare kommerzielle News-APIs: deren
kostenlose Free-Tiers erlauben laut AGB überwiegend nur lokale Entwicklung,
keinen produktiven/geplanten Cron-Lauf wie diese GitHub-Actions-Pipeline.

Aggregation: pro Feed nur Titel + Datum der aktuellsten ENTRIES_PER_FEED
Einträge (kein Volltext) - hält den Textblock kompakt und deterministisch,
statt den ohnehin schon grossen Prompt weiter aufzublähen (siehe die
max_tokens-Pufferanalyse vom 2026-09-21).

Rein informativer Kontext wie fundamentals.py/event_calendar.py - KEIN
Guardrail, KEIN Ausschlusskriterium. Ein Fehler beim Abruf EINES Feeds darf
weder die anderen Feeds noch den Pipeline-Lauf gefährden (siehe
fetch_all_feeds) - dasselbe Prinzip wie bei allen anderen weichen
Kontext-Quellen in diesem Projekt (data_quality.py, correlation.py,
fundamentals.py, event_calendar.py).
"""
from __future__ import annotations

import logging
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger("pipeline")

ENTRIES_PER_FEED = 8
REQUEST_TIMEOUT_SECONDS = 10
# Manche Feeds (z.B. CNBC/Yahoo) liefern ohne einen Browser-artigen User-Agent
# eine Fehlerseite statt des RSS-XML.
USER_AGENT = "Mozilla/5.0 (compatible; ki-portfolio-studie/1.0; +https://github.com/)"

# FEST ab OFFICIAL_STUDY_START (siehe Modul-Docstring) - (Anzeigename, URL).
NEWS_FEEDS: tuple[tuple[str, str], ...] = (
    ("CNBC - US Top News and Analysis", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC - Tech", "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ("MarketWatch - Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Federal Reserve - Monetary Policy", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("Yahoo Finance - News", "https://finance.yahoo.com/news/rssindex"),
)


@dataclass(frozen=True)
class NewsEntry:
    title: str
    published: pd.Timestamp | None


@dataclass(frozen=True)
class FeedResult:
    feed_name: str
    entries: list[NewsEntry] = field(default_factory=list)
    # None = erfolgreich (auch bei 0 Eintraegen); gesetzt = Abruf/Parsing
    # fehlgeschlagen. Bewusst unterschieden, statt einen fehlgeschlagenen
    # Abruf wie "keine News" aussehen zu lassen (siehe build_news_text_block).
    error: str | None = None


def _parse_rss(xml_bytes: bytes, max_entries: int) -> list[NewsEntry]:
    """Bewusst ein einfacher, generischer RSS-2.0-Parser (Standardbibliothek,
    kein feedparser o.ae. als zusaetzliche Abhaengigkeit) - alle fuenf Feeds
    oben folgen dem Standard-<item><title>/<pubDate>-Schema. `pd.Timestamp`
    parst sowohl das klassische RFC-822-Format (z.B. CNBC/MarketWatch/Fed:
    "Sun, 20 Sep 2026 08:06 GMT") als auch ISO-8601 (Yahoo Finance:
    "2026-09-19T02:59:56Z") ohne Formatunterscheidung noetig."""
    root = ET.fromstring(xml_bytes)
    entries: list[NewsEntry] = []
    for item in root.iter("item"):
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue
        pubdate_el = item.find("pubDate")
        published = None
        if pubdate_el is not None and pubdate_el.text:
            try:
                published = pd.Timestamp(pubdate_el.text.strip())
            except (ValueError, TypeError):
                published = None
        entries.append(NewsEntry(title=title, published=published))
        if len(entries) >= max_entries:
            break
    return entries


def fetch_feed(feed_name: str, url: str, max_entries: int = ENTRIES_PER_FEED) -> FeedResult:
    """Holt und parst EINEN Feed. Faengt jeden Fehler (Netzwerk, HTTP-Status,
    kaputtes XML) ab und liefert ihn als FeedResult.error zurueck statt zu
    werfen - siehe fetch_all_feeds fuer die Begruendung, warum ein Feed den
    Lauf nicht gefaehrden darf."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            xml_bytes = response.read()
        entries = _parse_rss(xml_bytes, max_entries)
        return FeedResult(feed_name=feed_name, entries=entries)
    except Exception as exc:
        return FeedResult(feed_name=feed_name, entries=[], error=str(exc))


def fetch_all_feeds(feeds: tuple[tuple[str, str], ...] = NEWS_FEEDS) -> list[FeedResult]:
    """Holt alle Feeds SEQUENZIELL ab (nur 5 Feeds, anders als z.B.
    event_calendar.check_upcoming_earnings mit 100+ Symbolen - keine
    Parallelisierung noetig). Ein fehlschlagender Feed wird geloggt und
    uebersprungen, die uebrigen werden trotzdem abgerufen."""
    results = []
    for feed_name, url in feeds:
        result = fetch_feed(feed_name, url)
        if result.error:
            log.warning("News-Feed '%s' fehlgeschlagen (%s) - wird übersprungen.", feed_name, result.error)
        results.append(result)
    return results


def build_news_text_block(feed_results: list[FeedResult]) -> str:
    """Baut den festen Textblock (Kap. 12.2: "identisch je Berichtszyklus")
    aus den Ergebnissen von fetch_all_feeds - EIN deterministisches Format je
    Feed (Name, dann je Eintrag Datum + Titel), unabhaengig davon, wie viele
    Feeds an einem gegebenen Tag erfolgreich waren. Ein fehlgeschlagener oder
    leerer Feed erscheint MIT explizitem Hinweis statt einfach zu fehlen -
    "kein Abruf moeglich" ist etwas anderes als "keine aktuellen News"
    (dieselbe Unterscheidung wie bei check_cycle_position_against_market_
    phase's None-Fall in market_phase.py)."""
    lines: list[str] = []
    for result in feed_results:
        lines.append(f"[{result.feed_name}]")
        if result.error:
            lines.append(f"  (Abruf fehlgeschlagen: {result.error})")
        elif not result.entries:
            lines.append("  (keine Einträge)")
        else:
            for entry in result.entries:
                date_str = entry.published.strftime("%Y-%m-%d") if entry.published is not None else "?"
                lines.append(f"  - [{date_str}] {entry.title}")
    return "\n".join(lines)
