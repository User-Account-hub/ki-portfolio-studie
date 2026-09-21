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

## 6. Kaufbegründungen je Trade (wortgetreu aus der DB)

_Auf Nachfrage ergänzt: die Original-Kaufbegründungen aus `decisions.proposed_orders` (JSON-Feld je
Order, Quelle: Claudes strukturierte Antwort), vollständig und wortgetreu übernommen, nicht
zusammengefasst oder umformuliert._

### AVGO, TSM, MU, CCJ (`decisions.id=19`)

**Nicht verfügbar.** Wie in Abschnitt 2 beschrieben, stürzte Lauf 1 ab, bevor der `decisions`-Eintrag
für diesen Lauf geschrieben wurde - Claudes vollständige Antwort (`raw_response`, inkl. `proposed_orders`
mit den Feldern `rationale`, `cycle_position`, `conviction`, `boundary_conditions` je Order) wurde
dadurch **nie persistiert** und ist nicht rekonstruierbar. `decisions.id=19` enthält ausschliesslich
den nachträglichen Audit-Backfill-Text (siehe Abschnitt 3/Commit `7cda30a`), keine
Original-Investment-Begründung. Dieser Verlust betrifft ausschliesslich die Text-Begründung der
vier Trades - Symbol, Menge, Preis, Zeitpunkt und Broker-Order-ID sind über die `trades`-Tabelle
weiterhin vollständig und korrekt dokumentiert (siehe Abschnitt 2).

### CEG (`decisions.id=20`)

- **Zyklus-Position:** `reversion_to_mean`
- **Konviktion:** `medium`

> CEG (Wide Moat: bestehende Nuklearflotte + regulatorische Eintrittsbarrieren) notiert unter SMA20
> (277.5) und nahe SMA50 (271.2) nach Konsolidierung - keine Manie-Extension wie z.B. AMD/Quantum-Titel
> heute (+5-15% Tagesgewinne laut News). Wirkungsmechanismus (Kap.4): strukturelles
> Angebots-Nachfrage-Ungleichgewicht bei Stromkapazität durch AI-Rechenzentren treibt mehrjährige
> Neubewertung; Markt fokussiert kurzfristig auf Chip-Quartalszahlen, ignoriert die langsamere aber
> verlässlichere Power-Story - Zeithorizont-Vorteil (b). Ergänzt bestehende CCJ-Position (Brennstoff)
> um die Erzeugungsseite der Nuklear-These. Zeitfenster: 3-6 Monate bis zur Bestätigung über
> PPA-Ankündigungen/Quartalsguidance. Gegenrisiko: hohe Verschuldung (D/E ~76) und negativer FCF durch
> Capex.

Randbedingungen (Kap. 7):
- Bestätigung: Rückeroberung SMA50 (~271) und Ausbruch über 275 als Zeichen erneuter Nachfrage nach
  Nuklear-PPA-Kapazität für Hyperscaler (Kursschwelle: 275.00)
- Entkräftung: Bruch unter 245 (ca. -7%) würde auf strukturelle Probleme bei
  Nuklear-Kapazitätserweiterung/PPA-Pipeline hindeuten statt auf normale Konsolidierung
  (Kursschwelle: 245.00)

### BWXT (`decisions.id=20`)

- **Zyklus-Position:** `reversion_to_mean`
- **Konviktion:** `medium`

> BWXT (Wide Moat: einer von sehr wenigen zertifizierten Lieferanten für US-Navy-Reaktoren und
> medizinische Isotope, jahrzehntelange Regierungsverträge) ist ~9% unter SMA50 (162.9)
> zurückgefallen, Volatilität mit 0.39 unterdurchschnittlich - kein Mania-Muster. Informationsvorteil
> (a): Backlog aus Regierungsaufträgen wird von generalistischen Momentum-Investoren, die aktuell
> AI-Chip-/Krypto-Schlagzeilen jagen, kaum beachtet. Wirkungsmechanismus (Kap.4): langzyklische
> Auftragsvisibilität treibt planbares Ertragswachstum und Neubewertung. Zeitfenster: 3-6 Monate bis
> zum nächsten Quartalsbericht mit Backlog-Update. Gegenrisiko: moderate Verschuldung (D/E ~151),
> Abhängigkeit von Regierungsbudgets.

Randbedingungen (Kap. 7):
- Bestätigung: Rückkehr über 160 (Richtung SMA50 bei 163) als Signal für anziehende Nachfrage nach
  Naval-Reactor/Medical-Isotope-Backlog (Kursschwelle: 160.00)
- Entkräftung: Bruch unter 135 (ca. -9%) würde auf nachlassende staatliche Auftragsvergabe oder
  Margendruck hindeuten (Kursschwelle: 135.00)
- Qualitativ: nächster Quartalsbericht muss Backlog-Wachstum im Naval-Reactor/Advanced-Nuclear-Segment
  bestätigen

### ISRG (`decisions.id=20`)

- **Zyklus-Position:** `attention`
- **Konviktion:** `high`

> ISRG (Wide Moat: dominante Marktstellung bei Chirurgie-Robotik, Razor-Razorblade-Modell mit hohen
> Wechselkosten) notiert moderat über SMA20/50 (+7%), aber mit der niedrigsten annualisierten
> Volatilitat im gesamten Universum (0.28 vs. Durchschnitt >0.6) - Indikator für stetige
> institutionelle statt retail-getriebene Manie-Akkumulation, klar abgrenzbar von den
> 5-15%-Tagesgewinnen in Quantum-/Krypto-Mining-Titeln laut News heute. Verhaltensvorteil (c):
> bewusste Rotation in einen bilanzstarken Compounder statt Momentum-Chasing in überhitzten
> Segmenten. Wirkungsmechanismus (Kap.4): Netzwerkeffekt der installierten Basis
> (Da-Vinci-Systeme) treibt wiederkehrende, planbare Instrumentenumsätze und Margenexpansion.
> Zeitfenster: 3-6 Monate bis zum nächsten Quartalsbericht mit Prozedurenvolumen-Daten. Gegenrisiko:
> hohe Bewertung, Multiple-Kompression bei Zinsanstieg.

Randbedingungen (Kap. 7):
- Bestätigung: neues Hoch über 415 als Zeichen fortgesetzter institutioneller Akkumulation ohne
  Manie-Charakter (Kursschwelle: 415.00)
- Entkräftung: Bruch unter 355 (ca. -11%, deutlich unter SMA50) würde auf nachlassende
  Prozedurenvolumina/Systemplatzierungen hindeuten (Kursschwelle: 355.00)

## 7. Hinweis: untypischer erster Tag

Der 2026-09-21 umfasste **zwei vollständige Entscheidungsrunden** (Lauf 1 und Lauf 2) statt der
regulär vorgesehenen einen - bedingt durch die produktive Fehlersuche und -behebung am selben Tag
(NAV-Mess-Fix, Crash-Robustheit, siehe Abschnitt 3), nicht durch eine methodische Änderung am
Anlageprozess selbst. **Ab dem nächsten Handelstag läuft die Pipeline regulär mit genau einem Lauf
pro Handelstag** (Cron-Schedule: werktäglich 15:00 UTC, siehe
`.github/workflows/weekly_pipeline.yml`).
