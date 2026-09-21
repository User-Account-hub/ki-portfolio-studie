# Portfolio-Report - ki-fallstudie-1 [NACHTRÄGLICH REKONSTRUIERT]

_Ursprünglicher Ausführungszeitpunkt der gezeigten Trades: 2026-09-21 17:36 UTC_
_Dieser Report wurde nachträglich rekonstruiert und ergänzt die Dokumentation - siehe Hinweis unten._

## ⚠️ Hinweis: Nachträglich rekonstruierter Report

Der ursprüngliche Pipeline-Lauf um 2026-09-21 17:36 UTC stürzte während der Order-Ausführung ab
(Alpaca-API-Fehler `{"code":42210000,"message":"fractional orders cannot be sold short"}` bei einer
fünften Order - siehe 17-Punkte-Audit Fund #3, seither behoben, Commit `d5f4461`). Dadurch wurde für
diesen Lauf ursprünglich **kein Report generiert**, und die vier zuvor bereits erfolgreich
ausgeführten Trades blieben zunächst ohne `decisions`-Eintrag zurück (nachträglich per Audit-Backfill
verknüpft, siehe `decisions.id=19`, Commit `7cda30a`).

Dieser Report wurde nachträglich rekonstruiert, um die Dokumentation zu vervollständigen. Er zeigt den
Portfolio-Zustand **unmittelbar nach genau den vier erfolgreich ausgeführten Trades dieses Laufs**
(AVGO, TSM, MU, CCJ - `decisions.id=19`), **VOR** den drei weiteren Trades (CEG, BWXT, ISRG) des
nachfolgenden, regulären Laufs um 18:14 UTC (`decisions.id=20`, siehe
[`report_2026-09-21_181416.md`](report_2026-09-21_181416.md)).

**Keine Trades wurden für diesen Report verändert** - reine Dokumentations-Nachbesserung anhand der
bereits vorhandenen `trades`-Zeilen (IDs 55-58).

**Was rekonstruierbar ist** (direkt aus den DB-Trade-Daten, exakt): NAV, Cash, offene Positionen,
Gesamtrendite seit dem Kap.-6.3-Reset, die vier ausgeführten Trades selbst mit echten
Alpaca-Order-IDs.

**Was NICHT rekonstruierbar ist:** Benchmark-/QQQ-/Momentum-Baseline-/Segment-Korb-Vergleiche,
Volatilität, Sharpe Ratio, Information Ratio - diese erfordern eine vollständige
Kurshistorien-Rekonstruktion für einen hypothetischen Zwischenzeitpunkt (17:36 UTC), die über eine
reine nachträgliche Dokumentations-Ergänzung hinausgeht (siehe `metrics.reconstruct_nav_history`,
das grundsätzlich bis "heute" rechnet, nicht bis zu einem vergangenen Zwischenzeitpunkt). Vol-Skalierung/
Konviktions-Faktoren der vier Orders sind ebenfalls nicht rekonstruierbar, da Claudes ursprüngliche
Antwort (`raw_response`) beim Absturz nie persistiert wurde - der Absturz erfolgte vor
`db.insert_decision`.

## Kennzahlen (Teilrekonstruktion)

| Kennzahl | Wert |
|---|---|
| NAV | 999'846.97 USD |
| Gesamtrendite (seit Reset, Kap. 6.3, Baseline 1'000'000 USD) | -0.0153% |
| Cash | 846'821.64 USD |
| Summe Notional der 4 Trades | 153'025.33 USD |
| Summe Transaktionskosten (0.1% je Trade) | 153.03 USD |

_Benchmark-/Volatilitäts-/Sharpe-/Momentum-Baseline-/Segment-Korb-Kennzahlen: nicht rekonstruiert,
siehe Hinweis oben. Die Gesamtrendite von -0.0153% entspricht exakt der Summe der
Transaktionskosten (153.03 USD) relativ zur 1'000'000-USD-Baseline - erwartungsgemäß, da unmittelbar
nach den Fills noch keine Kursbewegung stattgefunden hat._

## Offene Positionen (Stand nach `decisions.id=19`)

| Symbol | Menge | Ø Einstand | Notional |
|---|---|---|---|
| AVGO | 116.903725 | 359.38202 USD | 42'013.10 USD |
| TSM | 94.986115 | 442.259789 USD | 42'008.54 USD |
| MU | 30.285497 | 1039.56 USD | 31'483.59 USD |
| CCJ | 402.144772 | 93.30 USD | 37'520.11 USD |

## Trades in diesem (rekonstruierten) Lauf

| Symbol | Seite | Status | Notional | Transaktionskosten | Ausgeführt (UTC) | Broker-Order-ID |
|---|---|---|---|---|---|---|
| AVGO | buy | ✅ ausgeführt | 42'013.10 USD | 42.01 USD | 17:36:33 | `70c6861c-01b4-4914-95ba-a0c8a007a99a` |
| TSM | buy | ✅ ausgeführt | 42'008.54 USD | 42.01 USD | 17:36:35 | `8343cd93-f4fe-401c-aaf4-f2f418954fc4` |
| MU | buy | ✅ ausgeführt | 31'483.59 USD | 31.48 USD | 17:36:36 | `c200d25b-6be5-45a9-985b-c5936ac5a38e` |
| CCJ | buy | ✅ ausgeführt | 37'520.11 USD | 37.52 USD | 17:36:37 | `aa5c152c-af86-4551-a5ff-9f5b735599b1` |

_Vol-Skalierung/Konviktion je Order: nicht rekonstruierbar (Original-Claude-Antwort beim Absturz nicht
persistiert, siehe Hinweis oben)._

## Fünfte Order (abgestürzt, nicht ausgeführt)

Eine fünfte, hier nicht mehr identifizierbare Order (SHORT-Seite, fraktionierte/nicht-ganzzahlige
Menge) wurde vom Broker mit `{"code":42210000,"message":"fractional orders cannot be sold short"}`
abgelehnt. Die Exception lief zu diesem Zeitpunkt (vor dem Bugfix) ungefangen durch die Pipeline und
beendete den Lauf mit Exit-Code 1 - weder Symbol noch die genaue Position dieser Order in Claudes
ursprünglicher Vorschlagsliste sind aus der DB rekonstruierbar, da die Antwort nie persistiert wurde.

Siehe `decisions.id=19` (nachträglicher Audit-Backfill, `rationale`-Feld) und den Bugfix vom
2026-09-21 (17-Punkte-Audit Fund #3, Commit `d5f4461`): ab diesem Fix wird ein solcher Fehler pro
Order abgefangen, dokumentiert (`execution_error=True`) und die Order-Liste trotzdem zu Ende
verarbeitet, statt die Pipeline abstürzen zu lassen.
