"""Wrapper around the Anthropic Messages API for the trading-decision call."""
from __future__ import annotations

import sys

import anthropic


def get_trading_decision(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str = "claude-sonnet-5",
    max_tokens: int = 4096,
) -> str:
    """Calls Claude and returns the raw text of its response.

    Parsing/validation of the (structurally consistent-enough) JSON output
    happens downstream in order_schema.py.

    Note: `temperature` is intentionally not set - current-generation Claude
    models (e.g. claude-sonnet-5) reject it with a 400 ("`temperature` is
    deprecated for this model"), so this call relies on the model's default
    sampling behaviour instead.
    """
    # TEMP DEBUG (see task: httpcore.LocalProtocolError persists after strip()).
    # Never print the key itself - only length/whitespace metadata - so this
    # is safe to leave in a CI log. Remove once the root cause is confirmed.
    stripped_key = api_key.strip()
    print(
        "[DEBUG anthropic key] "
        f"raw_len={len(api_key)} stripped_len={len(stripped_key)} "
        f"raw_has_newline={chr(10) in api_key} raw_has_cr={chr(13) in api_key} "
        f"stripped_has_newline={chr(10) in stripped_key} stripped_has_cr={chr(13) in stripped_key}",
        file=sys.stderr,
    )

    # Strips accidental whitespace/newlines from the secret (e.g. a trailing
    # "\n" from how the value was pasted into a CI secret store) - the HTTP
    # client rejects header values containing raw newlines outright.
    client = anthropic.Anthropic(api_key=stripped_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
