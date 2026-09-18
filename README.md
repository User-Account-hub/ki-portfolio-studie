# KI-gestützte Portfolio-Fallstudie mit Claude

Eine Fallstudie, in der Claude zweimal wöchentlich (Montag/Donnerstag)
Handelsentscheidungen für ein
**Paper-Trading-Portfolio** vorschlägt. Alle Entscheidungen durchlaufen
serverseitige Risk-Guardrails, bevor sie über die **Alpaca Paper Trading
API** ausgeführt werden. Es fliesst zu keinem Zeitpunkt echtes Geld.

> ⚠️ **Kein Anlageberatungs-Tool.** Dieses Projekt dient ausschliesslich
> Demonstrations-/Forschungszwecken. Es handelt sich um Paper Trading
> (Simulation), keine Anlageempfehlung.

## Architektur

```
config/            Watchlist (Anlage-Universum) & Risk-Guardrail-Limiten
db/                 SQLite-Schema + Init-Skript
src/
  config.py         Lädt .env + YAML-Configs
  data_fetch.py      Marktdaten via yfinance
  prompt_builder.py  Baut System-/User-Prompt für Claude
  claude_client.py   Ruft die Anthropic Messages API auf
  order_schema.py    Pydantic-Modelle + JSON-Parsing der Claude-Antwort
  risk_guardrails.py Pre-Trade-Risikoprüfung (reines, getestetes Modul)
  broker_alpaca.py   Ausführung via Alpaca Paper Trading API
  execution.py       Orchestriert Risk-Check -> Ausführung -> DB-Update
  metrics.py         Rekonstruiert NAV-Verlauf & berechnet Kennzahlen
  reporting.py       Erzeugt Markdown-Report pro Lauf
  pipeline.py         Einstiegspunkt für einen kompletten Lauf
tests/               Unit-Tests (Fokus: risk_guardrails)
.github/workflows/   Wöchentlicher Cron-Lauf via GitHub Actions
reports/             Generierte Markdown-Reports (werden versioniert)
```

**Ablauf eines Laufs** (`src/pipeline.py`):

1. Marktdaten für Watchlist + offene Positionen laden (yfinance).
2. **Pflicht-Sweep:** offene Short-Positionen auf Stop-Loss prüfen und bei
   Bedarf zwangsweise schliessen - unabhängig von Claudes Vorschlag und
   unabhängig vom Tagesverlust-Stop (siehe unten).
3. Prompt bauen, Claude aufrufen, JSON-Antwort parsen & validieren
   (Pydantic).
4. Jede vorgeschlagene Order durch `risk_guardrails.evaluate_order` prüfen;
   nur freigegebene Orders werden ausgeführt.
5. Aktien/ETFs laufen über echte Alpaca-Paper-Orders (aktuell alle
   Watchlist-Titel inkl. der gehebelten ETF-Proxies NVDL/TSDD); ein
   `manual_simulation`-Pfad für echte strukturierte Produkte existiert
   weiterhin, ist aber ungenutzt (siehe Einschränkung unten).
6. NAV-Verlauf rekonstruieren, Kennzahlen berechnen, Report schreiben.

## Wichtige Einschränkungen & Design-Entscheidungen

- **Strukturierte Produkte (`leverage_certificate`/`mini_future`/`warrant`,
  simuliert über `trades.source = 'manual_simulation'`) sind aktuell
  ungenutzt.** Die ursprünglichen Platzhalter (`MINI-NVDA-LONG-1`,
  `WARRANT-TSLA-PUT-1`) wurden gemäss Kap. 6.7 der Thesis durch echte,
  bei Alpaca handelbare gehebelte ETFs ersetzt (`NVDL` - GraniteShares 2x
  Long NVDA, `TSDD` - GraniteShares 2x Short TSLA; `instrument_type: etf`),
  da Alpaca keine Schweizer/deutschen Retail-Derivate (Hebelzertifikate,
  Mini-Futures, Optionsscheine) abdeckt, wohl aber gehebelte/inverse
  Single-Stock-ETFs. Diese laufen jetzt über echte Alpaca-Paper-Orders wie
  jede andere Aktie/ETF. Der `manual_simulation`-Mechanismus bleibt im Code
  bestehen (falls später echte strukturierte Produkte mit echten
  Emittenten-Kennungen ergänzt werden), greift aber solange keine
  Watchlist-Einträge mit diesem `instrument_type` existieren.
  **Hinweis:** `check_structured_products_cap` und der Kap.-6.8-
  Circuit-Breaker (`check_circuit_breaker`) greifen ausschliesslich über
  `instrument_type in {leverage_certificate, mini_future, warrant}` - da
  NVDL/TSDD jetzt `etf` sind, zählen sie NICHT mehr zu diesen beiden
  Leitplanken, obwohl sie wirtschaftlich weiterhin 2x gehebelt sind. Beide
  Checks greifen wie zuvor auf ihre jeweiligen Segment-/Positionsgrössen-
  Limiten über die übrigen Guardrails.
- **"Kein Margin-Trading"** wird als "keine gehebelte Kaufkraft über 1x
  Cash hinaus" interpretiert (`risk_guardrails.check_no_margin`). Das für
  Shorting technisch nötige Alpaca-Margin-Konto ist davon ausgenommen, da
  Shorting explizit erlaubt ist.
- **Cashflow-Modell für Shorts:** Eröffnen eines Short erhöht das
  gebuchte Cash (Verkaufserlös), Cover reduziert es wieder - siehe
  Kommentar in `src/execution.py`. Damit bleibt der NAV beim Öffnen einer
  Position unverändert; P&L entsteht ausschliesslich durch Kursbewegung.
- **NAV-Verlauf ohne eigene Historien-Tabelle für den Report:** `metrics.py`
  rekonstruiert den Verlauf für Report/Sharpe/Max-Drawdown weiterhin durch
  Replay der `trades`-Tabelle plus historischen Kursen (yfinance), nicht über
  eine gespeicherte Kurve. Die Checkpoint-Frequenz (`reconstruct_nav_history`,
  Parameter `freqs`, Montag+Donnerstag) und die daraus abgeleitete
  Annualisierung (`periods_per_year`) folgen dem tatsächlichen Cron-Rhythmus
  und müssen bei einer erneuten Frequenzänderung mit angepasst werden. Seit
  dem Regimewechsel auf täglichen Handel am 2026-09-10 (Thesis Kap.
  6.3/11.2/15, Phase 1 → Phase 2) gilt das nur noch für Zeitpunkte vor
  `PHASE2_START`; ab dort erzeugt `reconstruct_nav_history` tägliche
  (werktägliche) Checkpoints, und `compute_metrics` annualisiert Vol/Sharpe
  mit `PHASE2_PERIODS_PER_YEAR` (252) - strikt getrennt von den älteren
  Phase-1-Renditen, um keine unterschiedlichen Checkpoint-Frequenzen in
  derselben Standardabweichung zu vermischen. Das Ganze ist eine Näherung
  (Kursbewegungen zwischen Checkpoints auf bereits geschlossenen Positionen
  fehlen), für diese Fallstudie aber ausreichend.
- **`nav_history`-Tabelle (5. Tabelle, seit Kap.-6.8-Guardrails):** eng
  zweckgebunden - pro Pipeline-Lauf genau ein Eintrag mit dem NAV zu
  Lauf-Beginn. Einziger Zweck: `risk_guardrails.check_circuit_breaker` einen
  echten historischen NAV-Höchststand (`db.get_peak_nav`, `MAX(nav)` über
  alle bisherigen Läufe) liefern, statt einer Näherung aus Startkapital und
  aktuellem Stand. Ersetzt nicht die obige Wochenverlauf-Rekonstruktion.
- **Defense in depth:** Der Prompt nennt Claude dieselben Limiten wie
  `config/risk_config.yaml`, aber `risk_guardrails.py` verlässt sich nie
  darauf, dass das Modell sie einhält - jede Order wird unabhängig
  geprüft.

## Setup

### Voraussetzungen

- Python 3.11+
- Ein [Anthropic API Key](https://console.anthropic.com/)
- Ein [Alpaca Paper Trading Account](https://app.alpaca.markets/paper/dashboard/overview)
  (kostenlos, keine echten Kontodaten nötig)

### 1. Repository & virtuelle Umgebung

```bash
git init   # falls noch nicht geschehen
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Umgebungsvariablen konfigurieren

```bash
cp .env.example .env
```

Trage in `.env` deinen `ANTHROPIC_API_KEY` sowie `ALPACA_API_KEY` /
`ALPACA_SECRET_KEY` (aus dem Alpaca-Paper-Dashboard) ein. Die übrigen
Werte haben sinnvolle Defaults.

### 3. Datenbank initialisieren

```bash
python db/init_db.py
```

Legt `db/portfolio.db` an und seedet ein Portfolio mit dem in `.env`
konfigurierten Startkapital (`INITIAL_CASH_BALANCE`, Default 100'000).

### 4. Anlage-Universum & Risk-Limiten anpassen

- `config/watchlist.yaml`: Symbole, die Claude vorschlagen darf. Das
  100-Titel-Aktienuniversum (Anhang A der Thesis) plus zwei gehebelte
  ETF-Proxies für Long-/Short-Exposure auf NVDA/TSLA (`NVDL`, `TSDD`,
  Kap. 6.7). Ein `instrument_type: mini_future/warrant/leverage_certificate`
  für echte strukturierte Produkte ist weiterhin unterstützt, aktuell aber
  nicht in der Watchlist vorhanden.
- `config/risk_config.yaml`: alle Guardrail-Limiten (Positionsgrösse,
  Trade-Notional, Tagesverlust-Stop, max. Trades/Tag, strukturierte-
  Produkte-Cap, Short-Stop-Loss, Margin-Verbot).

### 5. Tests ausführen

```bash
pytest
```

Die Tests decken `src/risk_guardrails.py` vollständig ab (Positionsgrössen-
Limit, Trade-Notional-Limit, Margin-Verbot, Tagesverlust-Stop, max.
Trades/Tag/Symbol, strukturierte-Produkte-Cap, Short-Stop-Loss-Sweep).

### 6. Pipeline manuell ausführen

```bash
python -m src.pipeline
```

Ein Report landet in `reports/report_<timestamp>.md`.

### 7. Historischer Stresstest (Thesis Kap. 11.2, manuell/on-demand)

```bash
python -m src.stress_test
```

Backtestet die **aktuellen offenen Positionen** (gewichtet nach heutigem
Marktwert) gegen vier feste historische Krisenfenster (GFC 2008, Covid-Crash
2020, Zinswende-Bärenmarkt 2022, Q4-Selloff 2018) und lässt Claude die
mechanisch berechneten Drawdown-Zahlen kommentieren. **Nicht** Teil des
automatischen Crons - rechenintensiv (mehrjährige Kurshistorien + ein
zusätzlicher Claude-Aufruf) und ein Validierungswerkzeug für die Thesis,
kein tägliches Betriebssignal. Ergebnis landet in
`reports/stress_test_<timestamp>.md`, mit einem expliziten
Kontaminations-Vorbehalt: die mechanischen Kennzahlen sind reine
Kursdaten-Berechnungen, aber Claudes Kommentar betrifft öffentlich
extensiv dokumentierte Ereignisse, deren tatsächlichen Verlauf das Modell
mit hoher Wahrscheinlichkeit bereits aus Trainingsdaten kennt (siehe
`src/stress_test.py`-Moduldocstring für die Einordnung).

## GitHub Actions: täglicher Lauf (seit Regimewechsel 2026-09-10)

Der Workflow `.github/workflows/weekly_pipeline.yml` läuft **werktäglich
(Montag-Freitag)** um 15:00 UTC (und ist manuell über "Run workflow"
auslösbar) - bewusst innerhalb der regulären NYSE-Handelszeit (9:30-16:00 ET),
unabhängig von der US-Sommerzeit. Die ursprüngliche Zeit (07:00 UTC) lag
ganzjährig vor Börsenöffnung, wodurch Market-Orders nie füllen konnten
(Alpaca queued sie bestenfalls für die nächste Session) - kein Code-Bug,
sondern ein falsch getimter Trigger.

Cadence-Historie: 1x/Woche (bis 2026-09-07) → 2x/Woche, Montag+Donnerstag
(2026-09-08 bis 2026-09-10) → täglich, Mo-Fr (ab 2026-09-10, Regimewechsel
Phase 1 → Phase 2 laut Thesis Kap. 6.3/11.2/15). `src/metrics.py` bildet
diesen Wechsel als Phasengrenze ab (`PHASE2_START`, `PHASE2_PERIODS_PER_YEAR`)
statt den gesamten Verlauf rückwirkend auf eine einzige Frequenz umzustellen
- vor `PHASE2_START` gelten weiterhin Mo/Do-Checkpoints, ab dort tägliche.

Bei einer weiteren Änderung des Rhythmus (Wochentage oder Häufigkeit) müssen
sowohl dieser Cron als auch `src/metrics.py`s `reconstruct_nav_history` (neue
`freqs`/`phase2_start`-Logik) und `compute_metrics`s
`PHASE2_PERIODS_PER_YEAR` entsprechend angepasst werden - sonst driftet die
daraus abgeleitete Annualisierung (Volatilität, Sharpe Ratio) von der
tatsächlichen Lauf-Frequenz weg. Der Dateiname `weekly_pipeline.yml` (und der
interne `chore: weekly pipeline run`-Commit-Präfix) blieb aus Kompatibilität
unverändert - **"weekly" im Namen ist historisch, nicht mehr wörtlich zu
nehmen.**

**Secrets** (Repo-Settings → Secrets and variables → Actions → *Secrets*):

| Name | Beschreibung |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `ALPACA_API_KEY` | Alpaca Paper API Key |
| `ALPACA_SECRET_KEY` | Alpaca Paper Secret Key |

**Variablen** (optional, unter *Variables* im selben Menü; sonst greifen
die Defaults aus dem Workflow):

`CLAUDE_MODEL`, `ALPACA_BASE_URL`, `DB_PATH`, `PORTFOLIO_NAME`,
`INITIAL_CASH_BALANCE`, `PORTFOLIO_CURRENCY`, `BENCHMARK_SYMBOL`,
`WATCHLIST_PATH`, `RISK_CONFIG_PATH`, `REPORTS_DIR`, `RISK_FREE_RATE_ANNUAL`
(Sharpe-Ratio-Annahme, Anker US-3-Monats-T-Bill - siehe src/metrics.py,
periodisch von Hand aktualisieren).

Die SQLite-Datei (`db/*.db`) und die generierten Reports werden vom
Workflow nach jedem Lauf zurück ins Repository committet, damit der
Portfolio-Zustand zwischen den Läufen erhalten bleibt. Das
Repo braucht dafür `permissions: contents: write` (bereits im Workflow
gesetzt).

## Dokumentationspflicht Short-Stop-Loss

Jede automatische Zwangsschliessung einer Short-Position (Kurs ≥ 20% über
Einstand) wird an zwei Stellen dokumentiert:

- `decisions`-Tabelle: eigener Eintrag mit `forced_action = 1` und
  Freitext-Begründung.
- `positions`-Tabelle: `closure_reason = 'short_stop_loss_forced'` und
  `closure_notes` mit den Details (Symbol, Einstands-/Auslösekurs, Verlust
  in %).

Diese Einträge erscheinen zusätzlich prominent im generierten
Wochenreport.

## Nächste Schritte / bekannte Grenzen

- Kursdaten für strukturierte Produkte werden über den Basiswert
  approximiert (kein Feed für Emittentenkurse/Spreads/Aufgeld enthalten).
- Kein Retry/Backoff für Anthropic- oder Alpaca-API-Fehler; ein
  fehlgeschlagener Lauf beendet sich mit Exit-Code 1 und wird im
  Actions-Log sichtbar.
- Keine Benachrichtigung (E-Mail/Slack) bei Fehlern oder Stop-Loss-
  Triggern - bei Bedarf leicht in `src/pipeline.py` ergänzbar.
