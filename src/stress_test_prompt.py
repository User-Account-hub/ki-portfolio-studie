"""Prompt für den analytischen (kontaminationsbehafteten) Teil des
historischen Stresstests (Thesis Kap. 11.2) - siehe src/stress_test.py für
den mechanischen, unkontaminierten Teil und die Gesamteinordnung.
"""
from __future__ import annotations

import json

STRESS_TEST_SYSTEM_PROMPT = """\
Du kommentierst das Ergebnis eines historischen Stresstests für das aktuelle \
Portfolio dieser KI-gestützten Portfolio-Fallstudie.

WICHTIGER HINWEIS ZUR KONTAMINATION (Thesis Kap. 11.2): Die untersuchten \
historischen Krisen (Globale Finanzkrise 2008, Covid-Crash 2020, \
Zinswende-Bärenmarkt 2022, Q4-Selloff 2018) sind öffentlich extensiv \
dokumentiert. Es ist wahrscheinlich, dass du aus deinen Trainingsdaten \
bereits weisst, wie einzelne der unten genannten Symbole/Segmente in diesen \
Zeiträumen tatsächlich abgeschnitten haben - deine Einschätzung ist damit \
potenziell durch Rückschau-Wissen (Hindsight Bias) verzerrt statt Ausdruck \
einer echten blinden Risikoanalyse allein anhand der gelieferten Zahlen.

Die gelieferten Drawdown-Werte sind reine, bereits fertig berechnete \
Kursdaten-Kennzahlen (mechanisch ermittelt, ohne dein Zutun) - deine Aufgabe \
ist NUR die Einordnung/Kommentierung dieser Zahlen, nicht eine eigene \
Neuberechnung oder Schätzung.

Antworte AUSSCHLIESSLICH mit einem einzigen validen JSON-Objekt, ohne \
Markdown-Fences, ohne Fliesstext davor oder danach.

JSON-Ausgabeschema:
{
  "resilience_assessment": "string - Einschätzung der Portfolio-Robustheit anhand der gelieferten Zahlen",
  "most_vulnerable_exposure": "string - welche Position/welches Segment über die Krisenfenster am anfälligsten wirkt und warum",
  "contamination_caveat": "string - PFLICHT: konkrete Selbsteinschätzung, ob und wie deine obige Einschätzung durch bereits bekanntes Wissen über diese spezifischen historischen Ereignisse beeinflusst sein könnte, statt sich rein aus den gelieferten Zahlen zu ergeben"
}
"""


def build_stress_test_user_prompt(portfolio_row, position_weights: list, period_results: list) -> str:
    payload = {
        "portfolio": portfolio_row["name"],
        "current_position_weights": [
            {"symbol": pw.symbol, "weight_pct": pw.weight} for pw in position_weights
        ],
        "stress_periods": [
            {
                "name": r.name,
                "start": r.start.date().isoformat(),
                "end": r.end.date().isoformat(),
                "max_drawdown_pct": r.max_drawdown_pct,
                "included_symbols": r.included_symbols,
                "excluded_symbols_no_historical_data": r.excluded_symbols_no_data,
            }
            for r in period_results
        ],
    }
    return (
        "Mechanisch berechnete historische Stresstest-Ergebnisse für das aktuelle Portfolio (JSON):\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False, default=str)}\n\n"
        "Erstelle deinen Kommentar gemäss dem im System-Prompt definierten JSON-Schema."
    )
