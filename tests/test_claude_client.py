"""Tests for src/claude_client.py's max_tokens-truncation detection and
Extended Thinking configuration (v3/v4/v5/v6, Thesis Kap. 6.2).

Monkeypatches anthropic.Anthropic so these run without any network call.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import anthropic
import pytest

import src.claude_client as claude_client


class FakeMessageStream:
    """Stand-in for the `MessageStreamManager`/`MessageStream` pair that
    `client.messages.stream()` returns - just enough of the context-manager
    + `get_final_message()` protocol for claude_client.py's v6 usage."""

    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None
        self.create_calls = 0
        self.stream_calls = 0

    def create(self, **kwargs):
        # Bewusst weiterhin vorhanden (nicht entfernt) statt zu raisen: falls
        # ein zukuenftiger Code-Pfad hierauf zurueckfaellt, soll das ueber
        # create_calls sichtbar werden (siehe
        # test_uses_streaming_not_create_to_avoid_10_minute_timeout_guard),
        # nicht stillschweigend durchlaufen.
        self.create_calls += 1
        self.last_kwargs = kwargs
        return self._response

    def stream(self, **kwargs):
        self.stream_calls += 1
        self.last_kwargs = kwargs
        return FakeMessageStream(self._response)


class FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = FakeMessages(FakeAnthropicClient.next_response)
        FakeAnthropicClient.last_messages = self.messages


def make_response(text: str, stop_reason: str):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_no_warning_when_response_completes_normally(monkeypatch, caplog):
    FakeAnthropicClient.next_response = make_response('{"orders": []}', "end_turn")
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    with caplog.at_level(logging.WARNING, logger="pipeline"):
        result = claude_client.get_trading_decision("system", "user", api_key="x")

    assert result == '{"orders": []}'
    assert not any("abgeschnitten" in r.message for r in caplog.records)


def test_warns_when_truncated_at_max_tokens(monkeypatch, caplog):
    FakeAnthropicClient.next_response = make_response('{"orders": [', "max_tokens")
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    with caplog.at_level(logging.WARNING, logger="pipeline"):
        result = claude_client.get_trading_decision("system", "user", api_key="x", max_tokens=16000)

    assert result == '{"orders": ['
    assert any("abgeschnitten" in r.message and "16000" in r.message for r in caplog.records)


def test_extended_thinking_is_enabled_with_high_effort(monkeypatch):
    """v3+v5 (Thesis Kap. 6.2): Extended Thinking muss aktiv sein
    (`thinking={"type": "adaptive"}`), mit effort="high" (v5 - zurueckgestuft
    von v4s "max", nachdem "max" am 2026-09-11 zu einer komplett leeren
    Antwort fuehrte - siehe die v5-Docstring in claude_client.py und den
    Regressionstest unten), NICHT ueber das fuer claude-sonnet-5 nicht mehr
    existierende `budget_tokens`."""
    FakeAnthropicClient.next_response = make_response('{"orders": []}', "end_turn")
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    claude_client.get_trading_decision("system", "user", api_key="x")

    sent_kwargs = FakeAnthropicClient.last_messages.last_kwargs
    assert sent_kwargs["thinking"] == {"type": "adaptive"}
    assert sent_kwargs["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in sent_kwargs.get("thinking", {})


def test_default_max_tokens_gives_headroom_after_max_tokens_incident(monkeypatch):
    """v5 (2026-09-11): Reproduziert den bestaetigten Produktionsfehler vom
    2026-09-11 15:14 UTC - bei effort="max" und max_tokens=16000 hatte das
    Thinking das gesamte Budget aufgebraucht, bevor irgendein Text-Content-
    Block geschrieben wurde (stop_reason=="max_tokens", raw_response=="" mit
    0 Zeichen - schlimmer als eine reine Mid-JSON-Truncation). Vor dem Fix
    (effort="max", max_tokens=16000) war dieses Szenario mit der damaligen
    Konfiguration reproduzierbar; der Fix stellt zwar keine Garantie gegen
    jedes Szenario dar, muss aber zumindest den damaligen Default
    (max_tokens=16000) durch eine hoehere Sicherheitsmarge ersetzt haben."""
    only_thinking_response = SimpleNamespace(
        stop_reason="max_tokens",
        content=[SimpleNamespace(type="thinking", thinking="...")],
    )
    FakeAnthropicClient.next_response = only_thinking_response
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    result = claude_client.get_trading_decision("system", "user", api_key="x")

    assert result == ""
    sent_kwargs = FakeAnthropicClient.last_messages.last_kwargs
    assert sent_kwargs["max_tokens"] == 32000 > 16000
    assert sent_kwargs["output_config"] == {"effort": "high"}


def test_uses_streaming_not_create_to_avoid_10_minute_timeout_guard(monkeypatch):
    """v6 (2026-09-12, bestaetigter Produktionsfehler): reproduziert erst die
    Ursache direkt gegen die echte SDK-Methode, dann prueft es den Fix.

    Ursache: `client.messages.create()` (synchron) berechnet intern ueber
    `_calculate_nonstreaming_timeout` eine Worst-Case-Dauer rein aus
    `max_tokens` (3600s * max_tokens / 128000) und verweigert den Call per
    ValueError, wenn das > 600s (10 Min) ergibt - bei max_tokens=24000 (v5)
    sind das 675s, bei den vorherigen 16000 nur 450s. Das haengt NICHT von
    `effort` ab, nur von `max_tokens`.

    Fix: `.stream()` hat diesen Guard nicht (nur `create()`), ist also fuer
    lange Anfragen zulaessig, ohne effort/max_tokens zurueckzudrehen. v7
    (2026-09-17) erhoeht max_tokens weiter auf 32000 (siehe claude_client.py
    v7-Docstring) - der Test nutzt bewusst diesen aktuellen Default, um zu
    zeigen, dass auch der hoehere Wert `create()` ausloesen wuerde
    (900s > 600s), `stream()` aber weiterhin unbetroffen bleibt."""
    real_client = anthropic.Anthropic(api_key="x")
    with pytest.raises(ValueError, match="Streaming is required"):
        real_client._calculate_nonstreaming_timeout(32000, None)

    FakeAnthropicClient.next_response = make_response('{"orders": []}', "end_turn")
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    result = claude_client.get_trading_decision("system", "user", api_key="x", max_tokens=32000)

    assert result == '{"orders": []}'
    assert FakeAnthropicClient.last_messages.stream_calls == 1
    assert FakeAnthropicClient.last_messages.create_calls == 0


def test_thinking_blocks_are_excluded_from_the_returned_text(monkeypatch):
    """Ein `thinking`-Content-Block darf nicht in die extrahierte Antwort
    einfliessen - nur `text`-Bloecke zaehlen."""
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking="Ueberlegung..."),
            SimpleNamespace(type="text", text='{"orders": []}'),
        ],
    )
    FakeAnthropicClient.next_response = response
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    result = claude_client.get_trading_decision("system", "user", api_key="x")
    assert result == '{"orders": []}'
