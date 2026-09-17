"""Wrapper around the Anthropic Messages API for the trading-decision call."""
from __future__ import annotations

import logging

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

    v6 (2026-09-12, bestaetigter Produktionsfehler, kein Ad-hoc-Wunsch): der
    Lauf danach brach schon VOR jedem Netzwerk-Call mit
    `ValueError: Streaming is required for operations that may take longer
    than 10 minutes` ab. Ursache (im SDK-Quellcode verifiziert,
    `anthropic._base_client.Anthropic._calculate_nonstreaming_timeout`,
    installierte Version 1.3.0): bei einem synchronen (nicht-streamenden)
    `messages.create()`-Call schaetzt das SDK rein aus `max_tokens` eine
    Worst-Case-Dauer ab (`3600s * max_tokens / 128000`) und verweigert den
    Call, wenn das ueber 10 Minuten liegt - bei max_tokens=24000 sind das
    675s (>600s), bei den vorherigen 16000 waren es nur 450s (<600s), daher
    ist erst der v5-Fix in dieses Limit gelaufen. WICHTIG: diese Schaetzung
    haengt NUR an `max_tokens`, nicht an `effort` - `effort` fliesst in die
    SDK-interne Berechnung ueberhaupt nicht ein, auch wenn es materiell fuer
    die tatsaechliche Dauer mitverantwortlich ist. Gegenmassnahme (statt
    max_tokens/effort wieder zu reduzieren und damit die v4/v5-Analysequalitaet
    zu verlieren): `messages.stream()` statt `messages.create()` - dieser Pfad
    hat in der SDK keinen `_calculate_nonstreaming_timeout`-Check (nur der
    synchrone `create()`-Pfad hat ihn), ist also fuer beliebig lange
    Anfragen zulaessig. `stream.get_final_message()` liefert danach dasselbe
    `Message`-Objekt (gleiche `content`-Blockliste, gleiches `stop_reason`)
    wie zuvor `client.messages.create(...)` direkt - die Extraktionslogik
    unten (Text-Bloecke joinen, stop_reason=="max_tokens" pruefen) bleibt
    unveraendert korrekt.
    """
    # Strips accidental whitespace/newlines from the secret (e.g. a trailing
    # "\n" from how the value was pasted into a CI secret store) - the HTTP
    # client rejects header values containing raw newlines outright.
    client = anthropic.Anthropic(api_key=api_key.strip())
    # v6: `.stream()` statt `.create()` - siehe Docstring oben. `max_tokens`
    # und `effort` bleiben unveraendert (v5), nur der Transportweg aendert sich.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        response = stream.get_final_message()
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
