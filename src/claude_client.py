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
    max_tokens: int = 24000,
) -> str:
    """Calls Claude and returns the raw text of its response.

    Parsing/validation of the (structurally consistent-enough) JSON output
    happens downstream in order_schema.py.

    Note: `temperature` is intentionally not set - current-generation Claude
    models (e.g. claude-sonnet-5) reject it with a 400 ("`temperature` is
    deprecated for this model"), so this call relies on the model's default
    sampling behaviour instead.

    v3 (2026-09-10, Thesis Kap. 6.2): Extended Thinking explicit gemacht
    (`thinking={"type": "adaptive"}` + `output_config={"effort": ...}`).
    claude-sonnet-5 laeuft Extended Thinking bereits standardmaessig, wenn
    `thinking` weggelassen wird - dieser Wechsel macht es nur explizit/
    dokumentiert. `budget_tokens` (fester Token-Betrag) existiert fuer dieses
    Modell NICHT mehr - die aktuelle SDK/Modellversion lehnt
    `thinking={"type": "enabled", "budget_tokens": N}` mit HTTP 400 ab (das
    war ein Pre-4.6-Mechanismus fuer aeltere Modelle). Denk-Tokens werden
    stattdessen ueber `effort` gesteuert und zaehlen in DASSELBE `max_tokens`
    hinein statt zusaetzlich dazu - `max_tokens` ist ein harter Deckel ueber
    die gesamte Antwort (Denken + Text). Die stop_reason=="max_tokens"-
    Warnung unten bleibt die massgebliche Absicherung, falls das Budget mal
    nicht reicht.

    v4 (2026-09-10, Thesis Kap. 6.2): `effort` von "medium" (v3) auf "max"
    erhoeht - reine Effort-Aenderung, `thinking={"type": "adaptive"}"` bleibt
    unveraendert (weiterhin der einzige "on"-Modus fuer claude-sonnet-5,
    unabhaengig vom effort-Wert). "max" laesst dem Modell deutlich mehr
    Denk-Tokens als "medium" - das erhoeht das Risiko, dass die Denk- +
    Text-Tokens zusammen die damaligen 16000 max_tokens ausschoepfen, bevor
    die JSON-Order-Antwort fertig ist.

    v5 (2026-09-11, bestaetigter Produktionsfehler - kein Ad-hoc-Wunsch,
    daher kein Verstoss gegen die Governance-Regel zu spontanen Prompt-
    Aenderungen): genau das v4-Risiko ist eingetreten. Der Lauf vom
    2026-09-11 15:14 UTC hatte stop_reason=="max_tokens" UND einen komplett
    leeren `raw_response` (0 Zeichen, kein einziger "text"-Content-Block) -
    schlimmer als die reine Truncation-mitten-im-JSON vom 2026-09-08-Incident,
    weil das Thinking bei effort="max" das gesamte 16000-Token-Budget
    aufgebraucht hat, bevor ueberhaupt Text-Output begann. Gegenmassnahme:
    `effort` zurueck auf "high" (direkter Hebel gegen den Ursprung des
    Problems, weniger angefordertes Thinking) UND `max_tokens` von 16000 auf
    24000 erhoeht (Sicherheitsmarge, falls "high" bei einem komplexen Prompt
    trotzdem mal knapp wird) - siehe Thesis-Diagnose-Session vom 2026-09-12.
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
        output_config={"effort": "high"},
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
