"""Wrapper around the Anthropic Messages API for the trading-decision call."""
from __future__ import annotations

import logging
import sys

import anthropic

log = logging.getLogger("pipeline")


def get_trading_decision(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str = "claude-sonnet-5",
    max_tokens: int = 16000,
) -> str:
    """Calls Claude and returns the raw text of its response.

    Parsing/validation of the (structurally consistent-enough) JSON output
    happens downstream in order_schema.py.

    Note: `temperature` is intentionally not set - current-generation Claude
    models (e.g. claude-sonnet-5) reject it with a 400 ("`temperature` is
    deprecated for this model"), so this call relies on the model's default
    sampling behaviour instead.

    v3 (2026-09-10, Thesis Kap. 6.2): Extended Thinking explicit gemacht
    (`thinking={"type": "adaptive"}` + `output_config={"effort": "medium"}`).
    claude-sonnet-5 laeuft Extended Thinking bereits standardmaessig, wenn
    `thinking` weggelassen wird - dieser Wechsel macht es nur explizit/
    dokumentiert und daempft die Denktiefe auf "medium" (statt des impliziten
    Default "high"), analog zu einem massvollen statt maximalen Budget.
    `budget_tokens` (fester Token-Betrag) existiert fuer dieses Modell NICHT
    mehr - die aktuelle SDK/Modellversion lehnt
    `thinking={"type": "enabled", "budget_tokens": N}` mit HTTP 400 ab (das
    war ein Pre-4.6-Mechanismus fuer aeltere Modelle). Denk-Tokens werden
    stattdessen ueber `effort` gesteuert und zaehlen in DASSELBE `max_tokens`
    hinein statt zusaetzlich dazu - `max_tokens` ist ein harter Deckel ueber
    die gesamte Antwort (Denken + Text). Bei effort="medium" bleibt genug
    Spielraum unter den aktuellen 16000 fuer die JSON-Order-Antwort; die
    stop_reason=="max_tokens"-Warnung unten bleibt trotzdem die massgebliche
    Absicherung, falls sich das aendert.
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
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_prompt}],
    )
    if response.stop_reason == "max_tokens":
        # Antwort wurde hart am Token-Limit abgeschnitten - typischerweise
        # mitten in einem JSON-String/Objekt, was order_schema.py als
        # "ungueltiges JSON" meldet. Ohne diesen Hinweis sieht das im Log wie
        # ein echter Syntaxfehler in Claudes Antwort aus, ist es aber nicht -
        # siehe INCIDENT-artige Verwirrung am 2026-09-08. max_tokens erhoehen
        # ist der richtige Hebel, nicht die Parsing-Logik reparieren.
        log.warning(
            "Claude-Antwort wurde bei max_tokens=%d abgeschnitten (stop_reason=max_tokens) - "
            "das ist wahrscheinlich die Ursache, falls die Antwort gleich als ungueltiges JSON scheitert.",
            max_tokens,
        )
    return "".join(block.text for block in response.content if block.type == "text")
