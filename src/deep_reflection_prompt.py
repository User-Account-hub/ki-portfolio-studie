"""Monatlicher Tiefenreflexions-Prompt (Thesis Kap. 6.12.3): ein separater,
rein analytischer Claude-Aufruf zusätzlich zum täglichen Handelsablauf
(prompt_builder.py), ausgelöst ab Woche 5 der offiziellen Studie
(OFFICIAL_STUDY_START, 2026-09-21) - rollierende 4-Wochen-Perioden
(DEEP_REFLECTION_PERIOD_DAYS), bewusst nicht Kalendermonate: Woche 5 beginnt
nicht am Monatsanfang, rollierende Perioden geben gleichmässige Abstände.

Prüft die sechs SYSTEM_PROMPT-Anforderungen (prompt_builder.py, v7)
rückblickend auf Portfolio-Ebene: These-Falsifizierung, Pattern-Matching-
Drift, empirischer Vergleich gegen Benchmark UND Momentum-Baseline (Kap.
6.9), Portfolio-weite Kohärenz, strukturelle Risiko-Drift. Erzeugt KEINE
Handelsorders - siehe pipeline.py's _maybe_run_deep_reflection für die
Einbettung (eigener 'deep_reflection'-Decision-Eintrag, kein Risk-
Guardrail-Pfad). Das Ergebnis fliesst ab dem nächsten Lauf in
build_user_prompt als Kontext für die tägliche Entscheidung ein.
"""
from __future__ import annotations

import json

import pandas as pd

OFFICIAL_STUDY_START = pd.Timestamp("2026-09-21")
DEEP_REFLECTION_PERIOD_DAYS = 28  # 4 Wochen


def periods_elapsed_since_study_start(
    today: pd.Timestamp, study_start: pd.Timestamp = OFFICIAL_STUDY_START
) -> int:
    """Anzahl vollständig abgeschlossener 4-Wochen-Perioden seit Studien-
    beginn. 0 vor Ablauf der ersten Periode (Wochen 1-4); wird 1, sobald
    Woche 5 beginnt (Tag 28) - das ist der Ausloeser fuer die erste
    Reflexion."""
    days_elapsed = (today.normalize() - study_start.normalize()).days
    return max(0, days_elapsed // DEEP_REFLECTION_PERIOD_DAYS)


def due_reflection_period(
    existing_reflection_count: int,
    today: pd.Timestamp,
    study_start: pd.Timestamp = OFFICIAL_STUDY_START,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """None, wenn keine weitere Reflexion fällig ist, sonst (period_start,
    period_end) der nächsten noch ausstehenden Periode.

    Fängt höchstens EINE ausstehende Periode pro Aufruf ab, nicht alle
    rückständigen auf einmal - holt bei einem übersprungenen/fehlgeschlagenen
    Lauf im nächsten fälligen Lauf automatisch nach (siehe pipeline.py:
    ein Fehler beim Reflexions-Call selbst wird nirgends als "erledigt"
    vermerkt), ohne einen Rückstau in einem einzigen Lauf abzuarbeiten.
    """
    periods_elapsed = periods_elapsed_since_study_start(today, study_start)
    if existing_reflection_count >= periods_elapsed:
        return None
    period_start = study_start + pd.Timedelta(days=existing_reflection_count * DEEP_REFLECTION_PERIOD_DAYS)
    period_end = study_start + pd.Timedelta(days=(existing_reflection_count + 1) * DEEP_REFLECTION_PERIOD_DAYS)
    return period_start, period_end


DEEP_REFLECTION_SYSTEM_PROMPT = """\
Du bist derselbe Portfolio-Analyst wie im täglichen Ablauf, aber in dieser \
Rolle triffst du HEUTE keine Handelsentscheidung. Stattdessen führst du eine \
monatliche Tiefenreflexion durch: ein Schritt zurück von der taktischen \
Tagesarbeit, um zu prüfen, ob die Methodik der letzten Reflexions-Periode \
tatsächlich eingehalten wurde - nicht nur im Wortlaut der einzelnen \
Begründungen, sondern im Ergebnis.

Prüfe anhand der gelieferten Historie (Entscheidungen, Trades, Performance-\
Vergleich der Periode) folgende fünf Punkte:

1. THESE-FALSIFIZIERUNG
   Für jede in der Periode eröffnete oder noch offene Position: war die \
ursprünglich genannte These (Zyklusphase, Wirkungsmechanismus, Zeitfenster) \
durch die seitherige Kursentwicklung bestätigt, widerlegt, oder ist ihr \
Zeitfenster abgelaufen, ohne dass eine Konsequenz gezogen wurde? Eine These \
mit abgelaufenem Zeitfenster ohne Neubewertung ist ein Befund, kein Detail.

2. PATTERN-MATCHING-SELBSTAUDIT
   Stützen sich die Begründungen der letzten Periode zu häufig \
ausschliesslich auf SMA20/50-Momentum-Sprache statt auf Zyklus-Position, \
Moat-Qualität oder einen benannten strukturellen Vorteil? Nenne konkrete \
Beispiele, falls ja.

3. EMPIRISCHER VERGLEICH
   Vergleiche die Portfolio-Rendite der Periode mit der Benchmark-Rendite \
UND der regelbasierten Momentum-Baseline (beide geliefert). Hat sich der \
behauptete strukturelle Vorteil gegenüber einer simplen Regel überhaupt \
niedergeschlagen?

4. PORTFOLIO-WEITE KOHÄRENZ
   Ist die aktuelle Gesamthaltung (offensiv/zyklusfrüh vs. defensiv/\
zyklusspät) noch stimmig mit der seitherigen Zyklusentwicklung, oder ist \
sie unbegründet gedriftet?

5. STRUKTURELLE RISIKEN
   Ist über die Periode eine Konzentrations-, Segment- oder Korrelations-\
Drift entstanden, die einzeln unter den Limiten blieb, aber in der Summe \
eine Schieflage andeutet?

Du schlägst in dieser Reflexion KEINE Orders vor - das bleibt dem \
täglichen Ablauf vorbehalten. Antworte AUSSCHLIESSLICH mit einem einzigen \
validen JSON-Objekt, ohne Markdown-Fences, ohne Fliesstext davor oder \
danach.

JSON-Ausgabeschema:
{
  "theses_confirmed": ["string - kurz, je bestätigte These"],
  "theses_falsified_or_overdue": ["string - kurz, je widerlegte/überfällige These"],
  "pattern_matching_concerns": "string - konkrete Beispiele, oder 'keine' falls nicht zutreffend",
  "portfolio_stance_assessment": "string - kohärent/gedriftet, mit Begründung",
  "reflection_commentary": "string - zusammenfassende Einschätzung, 3-6 Sätze"
}
"""


def build_deep_reflection_user_prompt(
    portfolio_row,
    decisions_in_period: list,
    trades_in_period: list,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    period_portfolio_return_pct: float,
    period_benchmark_return_pct: float,
    period_baseline_return_pct: float,
) -> str:
    decisions_summary = [
        {
            "created_at": d["created_at"],
            "rationale": d["rationale"],
            "proposed_orders": json.loads(d["proposed_orders"]) if d["proposed_orders"] else None,
        }
        for d in decisions_in_period
    ]
    trades_summary = [
        {
            "executed_at": t["executed_at"],
            "symbol": t["symbol"],
            "side": t["side"],
            "quantity": t["quantity"],
            "price": t["price"],
        }
        for t in trades_in_period
    ]
    payload = {
        "portfolio": portfolio_row["name"],
        "period_start": period_start.date().isoformat(),
        "period_end": period_end.date().isoformat(),
        "period_portfolio_return_pct": period_portfolio_return_pct,
        "period_benchmark_return_pct": period_benchmark_return_pct,
        "period_momentum_baseline_return_pct": period_baseline_return_pct,
        "decisions_in_period": decisions_summary,
        "trades_in_period": trades_summary,
    }
    return (
        "Entscheidungen, Trades und Performance-Vergleich der letzten Reflexions-Periode (JSON):\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False, default=str)}\n\n"
        "Erstelle deine Tiefenreflexion gemäss dem im System-Prompt definierten JSON-Schema."
    )
