"""Tests for src/news_feed.py's Kap.-12.2 News-Feed-Aggregation (2026-09-21,
erstmalige Umsetzung eines seit Beginn vorregistrierten, aber nie gebauten
Prinzips - siehe Modul-Docstring dort für die vollständige Begründung).

Pure/isolated: kein echter Netzwerk-Zugriff - urllib.request.urlopen wird
gemockt, damit die Tests deterministisch und offline laufen.
"""
from __future__ import annotations

import pytest

from src.news_feed import (
    NEWS_FEEDS,
    FeedResult,
    NewsEntry,
    _parse_rss,
    build_news_text_block,
    fetch_all_feeds,
    fetch_feed,
)

RFC822_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<item><title>Erste Meldung</title><pubDate>Sun, 20 Sep 2026 08:06:00 GMT</pubDate></item>
<item><title>Zweite Meldung</title><pubDate>Sat, 19 Sep 2026 14:30:00 GMT</pubDate></item>
</channel>
</rss>"""

ISO8601_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Yahoo-artig</title>
<item><title>Dritte Meldung</title><pubDate>2026-09-19T02:59:56Z</pubDate></item>
</channel></rss>"""


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._data


# --- _parse_rss ----------------------------------------------------------------


def test_parse_rss_extracts_title_and_date_rfc822():
    entries = _parse_rss(RFC822_RSS, max_entries=8)
    assert entries[0].title == "Erste Meldung"
    assert entries[0].published.strftime("%Y-%m-%d") == "2026-09-20"
    assert entries[1].title == "Zweite Meldung"


def test_parse_rss_extracts_date_iso8601():
    """Yahoo Finance liefert pubDate im ISO-8601-Format statt RFC-822 - beide
    muessen ohne Formatunterscheidung geparst werden (siehe pd.Timestamp)."""
    entries = _parse_rss(ISO8601_RSS, max_entries=8)
    assert entries[0].title == "Dritte Meldung"
    assert entries[0].published.strftime("%Y-%m-%d") == "2026-09-19"


def test_parse_rss_respects_max_entries():
    entries = _parse_rss(RFC822_RSS, max_entries=1)
    assert len(entries) == 1
    assert entries[0].title == "Erste Meldung"


def test_parse_rss_skips_items_without_title():
    xml = b"""<rss><channel>
    <item><pubDate>Sun, 20 Sep 2026 08:06:00 GMT</pubDate></item>
    <item><title>Hat einen Titel</title></item>
    </channel></rss>"""
    entries = _parse_rss(xml, max_entries=8)
    assert len(entries) == 1
    assert entries[0].title == "Hat einen Titel"


def test_parse_rss_missing_pubdate_is_none_not_a_crash():
    xml = b"<rss><channel><item><title>Ohne Datum</title></item></channel></rss>"
    entries = _parse_rss(xml, max_entries=8)
    assert entries[0].published is None


def test_parse_rss_unparseable_pubdate_is_none_not_a_crash():
    xml = b"<rss><channel><item><title>Kaputtes Datum</title><pubDate>nicht-ein-datum</pubDate></item></channel></rss>"
    entries = _parse_rss(xml, max_entries=8)
    assert entries[0].published is None


def test_parse_rss_empty_feed_returns_empty_list():
    assert _parse_rss(b"<rss><channel></channel></rss>", max_entries=8) == []


# --- fetch_feed ------------------------------------------------------------------


def test_fetch_feed_success(monkeypatch):
    monkeypatch.setattr(
        "src.news_feed.urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(RFC822_RSS)
    )
    result = fetch_feed("Test Feed", "https://example.com/rss")
    assert result.error is None
    assert result.feed_name == "Test Feed"
    assert len(result.entries) == 2


def test_fetch_feed_network_error_returns_error_result_not_raise(monkeypatch):
    def _raise(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("src.news_feed.urllib.request.urlopen", _raise)
    result = fetch_feed("Kaputter Feed", "https://example.com/rss")
    assert result.error == "timed out"
    assert result.entries == []


def test_fetch_feed_invalid_xml_returns_error_result_not_raise(monkeypatch):
    monkeypatch.setattr(
        "src.news_feed.urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(b"nicht valides XML <<<")
    )
    result = fetch_feed("Kaputtes XML", "https://example.com/rss")
    assert result.error is not None
    assert result.entries == []


# --- fetch_all_feeds ---------------------------------------------------------------


def test_fetch_all_feeds_one_failure_does_not_block_others(monkeypatch):
    """Kernanforderung (siehe Modul-Docstring): ein fehlschlagender Feed darf
    die uebrigen nicht verhindern."""
    def _fake_urlopen(request, timeout=None):
        if "good" in request.full_url:
            return _FakeResponse(RFC822_RSS)
        raise ConnectionError("nicht erreichbar")

    monkeypatch.setattr("src.news_feed.urllib.request.urlopen", _fake_urlopen)
    feeds = (("Guter Feed", "https://example.com/good"), ("Schlechter Feed", "https://example.com/bad"))
    results = fetch_all_feeds(feeds)

    assert len(results) == 2
    assert results[0].error is None
    assert len(results[0].entries) == 2
    assert results[1].error is not None
    assert results[1].entries == []


def test_fetch_all_feeds_defaults_to_the_five_fixed_feeds(monkeypatch):
    """Ohne explizites `feeds`-Argument muessen alle fuenf ab Studienstart
    festen Feeds abgefragt werden (Kap. 12.2: 'identisch je Berichtszyklus')."""
    seen_urls = []

    def _fake_urlopen(request, timeout=None):
        seen_urls.append(request.full_url)
        return _FakeResponse(RFC822_RSS)

    monkeypatch.setattr("src.news_feed.urllib.request.urlopen", _fake_urlopen)
    results = fetch_all_feeds()

    assert len(results) == 5
    assert len(NEWS_FEEDS) == 5
    assert seen_urls == [url for _, url in NEWS_FEEDS]


# --- build_news_text_block ---------------------------------------------------------


def test_build_news_text_block_formats_entries_with_date_and_title():
    results = [
        FeedResult(
            feed_name="Test Feed",
            entries=[
                NewsEntry(title="Erste Meldung", published=__import__("pandas").Timestamp("2026-09-20")),
            ],
        )
    ]
    block = build_news_text_block(results)
    assert "[Test Feed]" in block
    assert "2026-09-20" in block
    assert "Erste Meldung" in block


def test_build_news_text_block_marks_failed_feed_explicitly():
    results = [FeedResult(feed_name="Kaputter Feed", entries=[], error="timed out")]
    block = build_news_text_block(results)
    assert "[Kaputter Feed]" in block
    assert "Abruf fehlgeschlagen" in block
    assert "timed out" in block


def test_build_news_text_block_marks_empty_feed_explicitly_not_silently_missing():
    """'kein Abruf moeglich' (Fehler) und 'keine aktuellen News' (leerer,
    aber erfolgreicher Abruf) muessen unterscheidbar bleiben."""
    results = [FeedResult(feed_name="Leerer Feed", entries=[], error=None)]
    block = build_news_text_block(results)
    assert "[Leerer Feed]" in block
    assert "keine Einträge" in block
    assert "fehlgeschlagen" not in block


def test_build_news_text_block_handles_missing_date():
    results = [FeedResult(feed_name="Test Feed", entries=[NewsEntry(title="Ohne Datum", published=None)])]
    block = build_news_text_block(results)
    assert "[?]" in block
    assert "Ohne Datum" in block


def test_build_news_text_block_is_deterministic_for_same_input():
    """Kap. 12.2: 'identisch je Berichtszyklus' - derselbe Input muss immer
    denselben Textblock ergeben (keine Zufaelligkeit/Nichtdeterminismus,
    z.B. durch Set-Iteration oder Zeitstempel im Format selbst)."""
    results = [
        FeedResult(feed_name="A", entries=[NewsEntry(title="X", published=None)]),
        FeedResult(feed_name="B", entries=[], error="down"),
    ]
    assert build_news_text_block(results) == build_news_text_block(results)


def test_build_news_text_block_empty_results_returns_empty_string():
    assert build_news_text_block([]) == ""
