"""Tests for src/claude_client.py's max_tokens-truncation detection.

Monkeypatches anthropic.Anthropic so these run without any network call.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import src.claude_client as claude_client


class FakeMessages:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = FakeMessages(FakeAnthropicClient.next_response)


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
