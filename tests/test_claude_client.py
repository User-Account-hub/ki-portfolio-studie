"""Tests for src/claude_client.py's max_tokens-truncation detection and
Extended Thinking configuration (v3, Thesis Kap. 6.2).

Monkeypatches anthropic.Anthropic so these run without any network call.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import src.claude_client as claude_client


class FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


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


def test_extended_thinking_is_enabled_with_moderate_effort(monkeypatch):
    """v3 (Thesis Kap. 6.2): Extended Thinking muss aktiv sein
    (`thinking={"type": "adaptive"}`), mit gedaempftem statt maximalem
    Budget (`output_config={"effort": "medium"}"), NICHT ueber das fuer
    claude-sonnet-5 nicht mehr existierende `budget_tokens`."""
    FakeAnthropicClient.next_response = make_response('{"orders": []}', "end_turn")
    monkeypatch.setattr(claude_client.anthropic, "Anthropic", FakeAnthropicClient)

    claude_client.get_trading_decision("system", "user", api_key="x")

    sent_kwargs = FakeAnthropicClient.last_messages.last_kwargs
    assert sent_kwargs["thinking"] == {"type": "adaptive"}
    assert sent_kwargs["output_config"] == {"effort": "medium"}
    assert "budget_tokens" not in sent_kwargs.get("thinking", {})


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
