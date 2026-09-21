# Zusammenfassung: Erster offizieller Studientag (2026-09-21)

_Konsolidierte Chronologie, zusammengeführt aus bereits vorhandenen, verifizierten Quellen
(`RESET_2026-09-21.md`, `reports/report_2026-09-21_160956.md`, `reports/report_2026-09-21_170645.md`,
`reports/report_2026-09-21_173638_RECONSTRUCTED.md`, `reports/report_2026-09-21_181416.md`,
`db/portfolio.db`, sowie die unten genannten Commits). Keine neuen Berechnungen, keine Änderungen an
Code oder Datenbank - reine Dokumentation._

## 1. Reset-Zeitpunkt und Ausgangsstand

Der Pilotphase-→-offizielle-Studie-Reset (Kap. 6.3) wurde um **2026-09-21 09:32 ET (13:32 UTC)**
durchgeführt und verifiziert (siehe `RESET_2026-09-21.md`, Commits `0d40488`/`673064c`):

- Alle 22 Pilotphase-Positionen bei Alpaca real verkauft, alle Sell-Orders vollständig gefüllt.
- Lokale DB gespiegelt: `positions` → `closed`, `cash_balance` → **1'000'000.00 USD**, neuer
  `nav_history`-Eintrag (`id=14`, `nav=1'000'000.00`, `recorded_at='2026-09-21 13:32:06'`).
- Ausgangsstand für die offizielle Studie: **1'000'000.00 USD, 0 offene Positionen.**

## 2. Lauf 1 (17:36 UTC): 4 erfolgreiche Trades, dann Absturz

Um **2026-09-21 17:36 UTC** (Commit `7840ee2`) rief die Pipeline Claude auf und begann, die
vorgeschlagenen Orders auszuführen. Vier Buy-Orders wurden erfolgreich bei Alpaca gefüllt und
gebucht, in dieser Reihenfolge:

| Symbol | Menge | Preis | Notional | Ausgeführt (UTC) |
|---|---|---|---|---|
| AVGO | 116.903725 | 359.38202 USD | 42'013.10 USD | 17:36:33 |
| TSM | 94.986115 | 442.259789 USD | 42'008.54 USD | 17:36:35 |
| MU | 30.285497 | 1039.56 USD | 31'483.59 USD | 17:36:36 |
| CCJ | 402.144772 | 93.30 USD | 37'520.11 USD | 17:36:37 |

Danach stürzte der Lauf bei einer fünften Order ab: Alpaca lehnte eine fraktionierte
(nicht-ganzzahlige) SHORT-Order mit `{"code":42210000,"message":"fractional orders cannot be sold
short"}` ab. Die Exception lief zu diesem Zeitpunkt ungefangen durch die Pipeline und beendete den
Prozess mit Exit-Code 1, **bevor** ein `decisions`-Eintrag für diesen Lauf geschrieben wurde - die
vier bereits gefüllten Trades blieben dadurch zunächst ohne `decisions`-Verknüpfung zurück
(17-Punkte-Audit Fund #3, real eingetreten).

Details: [`report_2026-09-21_173638_RECONSTRUCTED.md`](report_2026-09-21_173638_RECONSTRUCTED.md)
(nachträglich rekonstruierter Report für genau diesen Zwischenstand) und der Bugfix-Commit `d5f4461`.

## 3. Zwischenzeitlich behobene Bugs

Zwischen dem Reset und Lauf 2 wurden am selben Tag zwei unabhängige, real aufgetretene Probleme
behoben:

- **NAV-Mess-Fix (Commit `6bf5ede`, 16:41 UTC):** `reconstruct_nav_history`/`compute_metrics`
  rekonstruierten die NAV-Historie weiterhin aus ALLEN Trades seit Projektbeginn (05.09., inkl.
  Pilotphase) statt ab dem Reset-Zeitpunkt - sichtbar geworden im ersten Report des Tages
  ([`report_2026-09-21_160956.md`](report_2026-09-21_160956.md), 16:09 UTC), der fälschlich
  $1'012'060.50/+1.21% zeigte, obwohl an dem Punkt noch kein einziger Trade seit dem Reset ausgeführt
  worden war. Fix: neue Konstante `OFFICIAL_STUDY_START`, NAV-Anker jetzt der tatsächliche
  Reset-`nav_history`-Eintrag statt des statischen Projektstart-Werts.
- **Crash-Robustheit (Commit `d5f4461`, 17:55 UTC):** siehe Abschnitt 2 - `execute_proposed_orders`
  fängt seither jede unerwartete Exception PRO Order ab, statt die gesamte Order-Liste abstürzen zu
  lassen; der `decisions`-Eintrag wird garantiert immer geschrieben (hält die Idempotenz-Sperre
  zuverlässig). Die vier verwaisten Trades aus Lauf 1 wurden zusätzlich nachträglich per Audit-Backfill
  mit einem dokumentierenden `decisions`-Eintrag verknüpft (`decisions.id=19`, Commit `7cda30a`).

(Ergänzend, nicht Teil dieser beiden Bugfixes: die `SYSTEM_PROMPT`-Versionen v10-v12, Commits
`dfb8518`/`899ffbf`/`c66ee0c`/`1594700`, informierten Claude im Tagesverlauf zusätzlich über bis dahin
unsichtbare serverseitige Positionsgrössen-Skalierung, Liquiditätslimit, Drawdown-Stufen und die
Transaktionskosten-Pauschale.)

## 4. Lauf 2 (mit `force_rerun`, nach den Fixes): 3 weitere erfolgreiche Trades

Um **2026-09-21 18:14 UTC** (Commit `54f0727`) wurde die Pipeline erneut manuell mit
`force_rerun=true` ausgelöst - diesmal mit beiden Fixes bereits aktiv. Der Lauf rief Claude auf und
führte drei weitere Buy-Orders erfolgreich aus, ohne Absturz und mit korrekt geschriebenem
`decisions`-Eintrag (`decisions.id=20`):

| Symbol | Menge | Preis | Notional | Ausgeführt (UTC) |
|---|---|---|---|---|
| CEG | 142.005106 | 263.43 USD | 37'408.40 USD | 18:14:03 |
| BWXT | 253.815704 | 147.96 USD | 37'554.57 USD | 18:14:06 |
| ISRG | 116.536555 | 399.72 USD | 46'581.99 USD | 18:14:08 |

Details: [`report_2026-09-21_181416.md`](report_2026-09-21_181416.md).

## 5. Finaler Tagesabschluss-Stand

Stand nach Lauf 2, gemäss `report_2026-09-21_181416.md` und `db/portfolio.db`:

| Kennzahl | Wert |
|---|---|
| NAV | 1'000'000.00 USD |
| Gesamtrendite (seit Reset) | 0.00% |
| Cash | 725'155.13 USD |
| Offene Positionen | 7 |

**Alle 7 offenen Positionen:**

| Symbol | Menge | Ø Einstand |
|---|---|---|
| AVGO | 116.9037 | 359.38 USD |
| TSM | 94.9861 | 442.26 USD |
| MU | 30.2855 | 1039.56 USD |
| CCJ | 402.1448 | 93.30 USD |
| CEG | 142.0051 | 263.43 USD |
| BWXT | 253.8157 | 147.96 USD |
| ISRG | 116.5366 | 399.72 USD |

## 6. Hinweis: untypischer erster Tag

Der 2026-09-21 umfasste **zwei vollständige Entscheidungsrunden** (Lauf 1 und Lauf 2) statt der
regulär vorgesehenen einen - bedingt durch die produktive Fehlersuche und -behebung am selben Tag
(NAV-Mess-Fix, Crash-Robustheit, siehe Abschnitt 3), nicht durch eine methodische Änderung am
Anlageprozess selbst. **Ab dem nächsten Handelstag läuft die Pipeline regulär mit genau einem Lauf
pro Handelstag** (Cron-Schedule: werktäglich 15:00 UTC, siehe
`.github/workflows/weekly_pipeline.yml`).
