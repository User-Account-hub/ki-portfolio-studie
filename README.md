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
config/                    Watchlist (Anlage-Universum) & Risk-Guardrail-Limiten
db/                        SQLite-Schema + Init-Skript
src/
  config.py                 Lädt .env + YAML-Configs
  data_fetch.py              Marktdaten via yfinance
  fundamentals.py            Weicher Qualitäts-Score als zusätzlicher Prompt-Kontext
  event_calendar.py          Event-Kalender-Hinweis (FOMC/CPI, Options-Verfall, Branchenevents)
  news_feed.py                RSS-News-Aggregation (Kap. 12.2, fünf feste Feeds)
  market_phase.py             Regelbasierter Bull/Bear/Seitwärts-Abgleich je Titel
  momentum_baseline.py        Regelbasierte Momentum-Baseline (Kap. 6.9)
  segment_basket.py           Thematischer Segment-ETF-Korb (Kap. 6.9)
  prompt_builder.py           Baut System-/User-Prompt für Claude
  claude_client.py            Ruft die Anthropic Messages API auf
  order_schema.py             Pydantic-Modelle + JSON-Parsing der Claude-Antwort
  boundary_conditions.py      Kap.-7-Randbedingungs-Tracking (Kursschwellen je Position)
  position_sizing.py          Volatilitätsadjustierte Positionsgrössen-Skalierung
  risk_guardrails.py          Pre-Trade-Risikoprüfung (reines, getestetes Modul)
  correlation.py              Korrelations-Beobachtung neuer Käufe (rein dokumentarisch)
  broker_alpaca.py            Ausführung via Alpaca Paper Trading API
  execution.py                Orchestriert Risk-Check -> Ausführung -> DB-Update
  db.py                       Dünne SQLite-Datenzugriffsschicht über db/schema.sql
  data_quality.py             Datenqualitäts-Checks (u.a. fehlende Kurse bei offener Position)
  deep_reflection_prompt.py   Monatlicher Tiefenreflexions-Prompt (Kap. 6.12.3)
  deep_reflection_schema.py   Pydantic-Modell + Parser für die Tiefenreflexions-Antwort
  stress_test.py              Historischer Stresstest (Kap. 11.2, manuell/on-demand)
  stress_test_prompt.py       Prompt für Claudes Kommentar zum Stresstest
  stress_test_schema.py       Pydantic-Modell + Parser für die Stresstest-Antwort
  metrics.py                  Rekonstruiert NAV-Verlauf & berechnet Kennzahlen
  reporting.py                 Erzeugt Markdown-Report pro Lauf
  pipeline.py                  Einstiegspunkt für einen kompletten Lauf
tests/                      Unit-Tests (deckt alle src/-Module ab, Kernfokus weiterhin risk_guardrails)
.github/workflows/          Werktäglicher Cron-Lauf via GitHub Actions
reports/                    Generierte Markdown-Reports (werden versioniert)
```

**Ablauf eines Laufs** (`src/pipeline.py`):

1. Marktdaten für Watchlist + offene Positionen laden (yfinance).
2. **Pflicht-Sweep:** offene Short-Positionen auf Stop-Loss prüfen und bei
   Bedarf zwangsweise schliessen - unabhängig von Claudes Vorschlag und
   unabhängig vom Tagesverlust-Stop (siehe unten).
3. Prompt bauen, Claude aufrufen, JSON-Antwort parsen & validieren
   (Pydantic).
4. Für jede positionsaufbauende Order (buy/short) wird zuerst die
   Positionsgrösse volatilitätsadjustiert skaliert
   (`position_sizing.compute_scaling_factors`/`scale_order_size`, siehe
   unten), dann durch `risk_guardrails.evaluate_order` geprüft; nur
   freigegebene Orders werden ausgeführt.
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
- **Volatilitätsadjustierte Positionsgrössen-Skalierung (2026-09-19,
  `src/position_sizing.py`):** Die von Claude vorgeschlagene Grösse jeder
  buy/short-Order wird VOR der Kap.-6.8-Guardrail-Prüfung mit
  Universums-Ø-Volatilität / Symbol-Volatilität skaliert (rollierende
  20-Tage-annualisierte Volatilität, dieselbe Grösse wie
  `data_fetch.MarketSnapshot.volatility_20d_annualized` - kein
  zusätzlicher Datenabruf). Der Faktor ist auf
  `volatility_scaling_min_factor`/`_max_factor` in `risk_config.yaml`
  (Default 0.5x-1.5x) geclippt: unterdurchschnittlich volatile Titel werden
  hochskaliert, überdurchschnittlich volatile runterskaliert - eine
  einfache Näherung an inverse-Volatility-Sizing. Wirkt ausdrücklich nur
  INNERHALB der bestehenden Limiten (die skalierte Order durchläuft
  danach dieselbe Guardrail-Prüfung wie jede andere), hebelt sie also
  nicht aus. Sell/Cover-Orders und Symbole ohne berechenbare Volatilität
  (zu kurze Historie) werden nicht skaliert. Der angewendete Faktor wird
  pro Trade im Report dokumentiert.
- **Konviktions-Multiplikator (2026-09-19, `src/position_sizing.py`,
  `order_schema.ConvictionLevel`):** Claude kann optional je Kauf-/Short-
  Order eine Konviktion ("high"/"medium"/"low") angeben; sie skaliert die
  Positionsgrösse zusätzlich zur Volatilitäts-Skalierung (multiplikativ,
  vor derselben Guardrail-Prüfung), aber bewusst NUR LEICHT (×1.15/×1.0/
  ×0.8, siehe `CONVICTION_SCALING_FACTORS`) - deutlich schwächer als das
  0.5x-1.5x-Volatilitätsband. Grund: die Volatilität ist ein tatsächlich
  gemessenes Marktsignal, während "conviction" eine Selbsteinschätzung des
  Sprachmodells über die eigene Handelsidee ist - solche LLM-
  Selbsteinschätzungen sind bekanntermassen schlecht kalibriert
  (Overconfidence-Bias), ein starker Hebel darauf würde also
  Selbstüberschätzung statt zusätzlicher Information einpreisen. Fehlt das
  Feld, wird keine Skalierung angewendet (kein unterstellter Default-
  Level). Auch dieser Faktor kann keines der Kap.-6.8-Limiten aushebeln
  und wird pro Trade im Report dokumentiert.
- **Abgestufter Drawdown-Positionsgrössen-Schutz (2026-09-19,
  `risk_guardrails.drawdown_position_size_factor`):** Erweitert den
  bisherigen harten Circuit-Breaker (der ausschliesslich Hebelpositionen
  komplett stoppt, s.o.) um Zwischenstufen für `check_position_size`
  (`max_position_size_pct_of_portfolio`), basierend auf demselben
  historischen NAV-Höchststand (`ctx.peak_nav`/`db.get_peak_nav`) wie der
  Circuit-Breaker: ab -10% Drawdown wird die max. erlaubte NEUE
  Positionsgrösse auf 75% reduziert (`drawdown_tier1_pct`/
  `_position_size_factor` in `risk_config.yaml`), ab -15% auf 50%
  (`drawdown_tier2_*`), ab -25% komplett gestoppt - dieselbe Schwelle wie
  `circuit_breaker_drawdown_pct`, damit "wie bisher komplett stoppen" am
  selben Drawdown-Punkt greift. Gilt **ausdrücklich nicht für
  Hebelpositionen/strukturierte Produkte** (`evaluate_order` übergibt für
  sie unverändert Faktor 1.0) - deren Verhalten bleibt exklusiv durch den
  bestehenden `check_circuit_breaker` geregelt, um nicht zwei
  unterschiedliche Drawdown-Regimes auf dieselben Positionen anzuwenden.
- **"Kein Margin-Trading"** wird als "keine gehebelte Kaufkraft über 1x
  Cash hinaus" interpretiert (`risk_guardrails.check_no_margin`). Das für
  Shorting technisch nötige Alpaca-Margin-Konto ist davon ausgenommen, da
  Shorting explizit erlaubt ist.
- **News-Feed-Aggregation (Kap. 12.2, `src/news_feed.py`):** erstmalige
  Umsetzung eines seit Beginn vorregistrierten, aber nie gebauten Prinzips -
  eingeführt am 2026-09-21, unmittelbar VOR dem offiziellen Studienstart
  (kein Regimewechsel während der laufenden Studie, die Pilotphase Kap. 6.3
  zählt ohnehin nicht zur Auswertung). Fünf feste, am selben Tag live
  verifizierte RSS-Feeds (CNBC US-Top-News, CNBC Tech, MarketWatch
  Top-Stories, Fed-Monetary-Policy-Pressemitteilungen, Yahoo-Finance-News) -
  ab Studienstart **unveränderlich**, siehe Modul-Docstring für die
  vollständige Begründung inkl. der Feeds, die bewusst NICHT gewählt wurden
  (kommerzielle News-APIs mit Free-Tiers, die laut AGB keinen produktiven
  Cron-Lauf erlauben). Pro Feed nur Titel + Datum der letzten 5-8 Einträge
  (kein Volltext) zu einem festen Textblock aggregiert (Feld `news_context`
  im Prompt) - rein informativ, kein Ausschlusskriterium, kein Guardrail. Ein
  einzelner fehlschlagender Feed blockiert die übrigen nicht; eine
  fehlgeschlagene Aggregation gefährdet den Lauf nicht (liefert dann `null`
  statt Kontext).
- **Markt-Phasen-Abgleich (2026-09-20, `src/market_phase.py`,
  `order_schema.CyclePosition`):** Eine automatische, regelbasierte
  Bull/Bear/Seitwärts-Klassifikation je Titel aus SMA20/SMA50 und der
  rollierenden 20-Tage-Volatilität (`classify_market_phase` - Seitwärts
  innerhalb einer volatilitätsskalierten Bandbreite um SMA50, sonst Bull/Bear
  je nachdem, ob Kurs UND SMA20 gemeinsam über/unter SMA50 liegen; ein
  gemischtes Signal gilt konservativ als Seitwärts). Claude kann optional je
  Order strukturiert dieselbe Zyklus-Position angeben, die SYSTEM_PROMPT-
  Anforderung 1 ohnehin verbal verlangt (Feld `cycle_position`:
  Akkumulation/Aufmerksamkeit/Manie/Crash/Rückkehr zum Mittel, Kap. 3) -
  `check_cycle_position_against_market_phase` vergleicht beide anhand einer
  dokumentierten, bewusst groben Heuristik (z.B. "Manie" ist nur bei
  regelbasiertem Bull plausibel). Ausschliesslich aus bereits vorhandenen
  Kursdaten (denselben `sma20`/`sma50`/`volatility_20d_annualized`-Werten
  wie Prompt/Report), kein zusätzlicher Datenabruf. Ein Widerspruch wird pro
  Trade im Report dokumentiert, löst aber **kein Veto** aus - weder die
  Guardrail-Prüfung noch die Order-Ausführung werden davon beeinflusst; ein
  einfaches 3-Phasen-Regelwerk ist kein Beweis, dass Claudes 5-Phasen-
  Einschätzung falsch liegt.
- **Idempotenz-Sperre (Kap. 12.7 "Datenintegrität", 2026-09-21,
  `db.get_successful_decision_today`, `pipeline._already_decided_today`):**
  ein dauerhafter Guardrail (kein Übergangs-Mechanismus wie der obige) -
  existiert für den heutigen Kalendertag bereits ein abgeschlossener
  Handelsentscheidungs-Eintrag (Claude wurde aufgerufen UND die Antwort
  wurde erfolgreich in eine Order-Liste geparst), bricht ein erneuter Lauf
  am selben Tag sauber ab: kein zweiter Claude-Aufruf, kein Marktdaten-Abruf,
  kein zusätzlicher Trade, aber ein dokumentierender `pipeline_guard`-
  Decision-Eintrag. Ein Börse-geschlossen-Skip oder ein `OrderParsingError`
  zählen bewusst NICHT als abgeschlossene Entscheidung (`proposed_orders`
  bleibt dabei `NULL`) - ein erneuter Versuch nach einem Parsing-Fehler am
  selben Tag muss möglich bleiben. Ausnahme für bewusste manuelle
  Wiederholungen: `force=True` als Parameter an `run()` oder die
  Umgebungsvariable `FORCE_RERUN=true` (im CI über den
  `force_rerun`-Input des `workflow_dispatch`-Triggers, siehe
  `weekly_pipeline.yml`) übergeht die Sperre.
- **Selbstkonsistenz-Prüfung der monatlichen Tiefenreflexion (2026-09-19,
  Kap. 6.12.3, `deep_reflection_schema.check_self_consistency`,
  `pipeline._maybe_run_deep_reflection`):** Claude wird für dieselbe
  Reflexions-Periode zweimal mit IDENTISCHEM Prompt aufgerufen; die
  Kernaussagen (`theses_confirmed`/`theses_falsified_or_overdue`, normalisiert
  verglichen) werden gegenübergestellt. AUSSCHLIESSLICH für die
  Tiefenreflexion, NICHT für den täglichen Handelsablauf - Kostengründe (die
  Reflexion läuft nur alle 4 Wochen, der tägliche Ablauf bei jedem Lauf).
  Bei einer Abweichung gibt es **keine automatische Konfliktlösung**: beide
  vollständigen Antworten werden dokumentiert (DB: `raw_response` =
  1. Aufruf, `risk_check_result` trägt den 2. Aufruf + das Konsistenz-
  Ergebnis; Report: eigener Abschnitt "Selbstkonsistenz-Prüfung", der bei
  Abweichung beide Antworten vollständig zeigt). Der erste Aufruf dient
  unverändert - unabhängig vom Konsistenz-Ergebnis - als Grundlage für den
  nächsten Tagesprompt (`db.get_latest_reflection`). Der Vergleich selbst
  ist ein einfacher, deterministischer Mengenvergleich normalisierter
  These-Strings, kein weiterer LLM-Aufruf zur "semantischen" Prüfung - er
  erkennt daher inhaltlich gleichwertige, aber anders formulierte Thesen
  nicht als übereinstimmend (dokumentierte Grenze, siehe Docstring).
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
- **Information Ratio (2026-09-20, `MetricsResult.information_ratio`):**
  = Alpha (`alpha_pct`, dieselbe kumulierte Rendite-Differenz seit
  `initial_nav` wie in der Report-Zeile "Alpha vs. Benchmark", direkt
  daneben ausgegeben) geteilt durch den Tracking Error (annualisierte
  Standardabweichung der PERIODEN-Renditedifferenz Portfolio minus
  Benchmark, nicht der kumulierten Differenz) - nutzt denselben Phase-1/2-
  Annualisierungs-Split wie Volatilität/Sharpe oben, damit nicht zwei
  unabhängige Checkpoint-Einteilungen für dieselbe Art Kennzahl entstehen.
  `None`, wenn kein Tracking Error berechenbar ist (zu wenig Perioden, oder
  Portfolio bewegt sich exakt wie die Benchmark - Standardabweichung 0,
  keine Division durch Null).
- **Zweiter Vergleichsindex QQQ (Kap. 6.9 Erweiterung, 2026-09-21,
  `metrics.SECONDARY_BENCHMARK_SYMBOL`, `MetricsResult.qqq_total_return_pct`/
  `alpha_vs_qqq_pct`):** ergänzt den bestehenden, konfigurierbaren Haupt-
  Benchmark (`benchmark_symbol`, aktuell SPY) um einen fest kodierten,
  sektorspezifischen Nasdaq-100-Vergleich - ERSETZT SPY nicht, beide Zeilen
  stehen nebeneinander im Report. Bewusst hart kodiert statt konfigurierbar
  (analog zu den News-Feeds/FOMC-CPI-Terminen): das Anlage-Universum
  (Kap. 6.7) ist stark AI-/Halbleiter-lastig, ein reiner S&P-500-Vergleich
  allein unterrepräsentiert das. Dieselbe `initial_nav`-Ankerung und
  Fallback-Logik wie beim Haupt-Benchmark (`_normalize_symbol_to_initial_cash`
  in `metrics.py`, aus der ursprünglich benchmark-spezifischen Inline-Logik
  extrahiert, da jetzt für zwei Symbole gebraucht).
- **Thematischer Segment-ETF-Korb (Kap. 6.9 Erweiterung, 2026-09-21,
  `src/segment_basket.py`, `SEGMENT_BASKET_SYMBOLS`,
  `MetricsResult.segment_basket_total_return_pct`/
  `alpha_vs_segment_basket_pct`):** ein weiterer, rein informativer
  Vergleichspunkt zusätzlich zu SPY/QQQ/Momentum-Baseline - gleichgewichtetes,
  monatlich rebalanciertes Portfolio aus SMH (Halbleiter), URA (Uran) und
  ICLN (Clean Energy), fest kodiert (dieselbe Begründung wie bei QQQ oben:
  eine thematische Referenz für das Anlage-Universum, kein frei wählbarer
  Massstab). Architektonisch identisch zur Momentum-Baseline
  (`src/momentum_baseline.py`) berechnet - reine Kursdaten-Rekonstruktion,
  kein separat gehandeltes Portfolio - nur OHNE deren Auswahl/Rangliste
  (immer alle drei Symbole, immer gleichgewichtet statt "Top-Quintil").
- **`nav_history`-Tabelle (5. Tabelle, seit Kap.-6.8-Guardrails):** eng
  zweckgebunden - pro Pipeline-Lauf genau ein Eintrag mit dem NAV zu
  Lauf-Beginn. Einziger Zweck: `risk_guardrails.check_circuit_breaker` einen
  echten historischen NAV-Höchststand (`db.get_peak_nav`, `MAX(nav)` über
  alle bisherigen Läufe) liefern, statt einer Näherung aus Startkapital und
  aktuellem Stand. Ersetzt nicht die obige Wochenverlauf-Rekonstruktion.
- **Liquiditätslimit (Kap. 6.13, 2026-09-21,
  `risk_guardrails.check_liquidity_limit`,
  `max_order_pct_of_avg_daily_volume` in `risk_config.yaml`):** ein hartes
  Guardrail (Ablehnung wie jede andere Kap.-6.8-Verletzung) - eine
  positionsaufbauende Order (buy/short) darf nicht mehr als einen
  konfigurierbaren Anteil (Default 10%) des Tagesvolumens des Titels
  ausmachen. Schützt vor einem Ausführungs-/Slippage-Risiko, das die
  NAV-basierten Limiten oben nicht abdecken: eine vom NAV her erlaubte
  Positionsgrösse kann das tatsächliche Handelsvolumen eines dünn
  gehandelten Micro-/Small-Cap-Titels trotzdem übersteigen. Nutzt
  `MarketSnapshot.volume` (bereits vorhanden, kein zusätzlicher Datenabruf) -
  **dokumentierte Einschränkung:** das ist das zuletzt bekannte
  EINZELTAGES-Volumen, kein echter mehrtägiger gleitender Durchschnitt
  (yfinance liefert hier keinen). Fehlt das Volumen (Datenausfall), wird
  NICHT blockiert, um einen Datenausfall nicht fälschlich als
  Liquiditätsproblem zu werten.
- **Fehlende Kursdaten bei offener Position (2026-09-21,
  `data_quality.detect_stale_open_positions`):** erkennt eine noch offene
  Position, für die weder ihr eigenes Symbol noch (bei strukturierten
  Produkten) ihr Basiswert einen aktuellen Kurs liefert - typischerweise ein
  Delisting oder eine Übernahme. **Löst ausdrücklich KEINE automatische
  Order aus** (z.B. keine Zwangsschliessung) - zu riskant für einen
  Automatismus. Stattdessen: `log.error` je betroffener Position, ein
  eigener, prominenter Report-Abschnitt "⚠️ Kursdaten fehlen - manuelle
  Prüfung nötig" (analog zu den Stop-Loss-Zwangsschliessungen) UND eine
  Markierung in der Spalte "Status" der Tabelle "Offene Positionen" - beides
  gleichzeitig, damit es nicht übersehen werden kann. Für ein strukturiertes
  Produkt, dessen Basiswert weiterhin gehandelt wird, greift das NICHT (der
  bereits bestehende, gewollte Preis-Proxy-Mechanismus bleibt unverändert).
- **Options-Verfallstage im Event-Kalender (2026-09-21,
  `event_calendar.third_friday_of_month`/`_upcoming_third_fridays`):**
  ergänzt den bestehenden FOMC-/CPI-Hinweis um den dritten Freitag jedes
  Monats ("Hexensabbat"/Options-Quartalsverfall in März/Juni/September/
  Dezember - gleichzeitiger Verfall von Index-Futures, Index-Optionen UND
  Aktienoptionen, die stärkste Ausprägung; regulärer, monatlicher
  Aktienoptionsverfall in den übrigen acht Monaten). Anders als
  `FOMC_DECISION_DATES`/`CPI_RELEASE_DATES` (hardcodiert, jährlich von Hand
  nachzupflegen) ist das eine feste Kalenderregel und wird BERECHNET, nicht
  recherchiert - muss also nie aktualisiert werden. Rein informativ wie der
  übrige Event-Kalender: das Anlage-Universum handelt selbst keine
  Optionen, der Hinweis dokumentiert nur die historisch erhöhte Volatilität
  der zugrundeliegenden Aktien/ETFs an diesen Tagen - kein Guardrail, keine
  automatische Reaktion.
- **Branchenspezifische Grossveranstaltungen im Event-Kalender (2026-09-21,
  `event_calendar.SECTOR_EVENT_DATES`):** kleine, hardcodierte Liste
  wiederkehrender Termine, ausgewählt nach Relevanz für das stark AI-/
  Halbleiter-/Krypto-Mining-lastige Anlage-Universum (Kap. 6.7): CES und
  NVIDIA GTC decken die grossen Halbleiter-/AI-Ankündigungstermine ab (u.a.
  NVDA, AMD, AVGO, ARM, MRVL, TSM, MU, ASML, SMCI im Universum), Mining
  Disrupt die Krypto-Mining-Titel (u.a. RIOT). Je Veranstaltung EIN Datum
  (der markt-/ankündigungsrelevanteste einzelne Tag, i.d.R. Eröffnung/
  Keynote), nicht der volle mehrtägige Zeitraum. Anders als die berechnete
  Options-Verfallsregel oben legt hier der jeweilige Veranstalter das Datum
  jährlich neu fest - muss also wie `FOMC_DECISION_DATES`/
  `CPI_RELEASE_DATES` jährlich von Hand nachgepflegt werden. Rein
  informativ wie der übrige Event-Kalender: kein Guardrail, keine
  automatische Reaktion.
- **Individueller Short-Stop-Loss hat Vorrang vor der globalen Schwelle
  (2026-09-22, `v13`, 17-Punkte-Audit Fund #1, HIGH):**
  `risk_guardrails.evaluate_short_positions_for_stop_loss` liest jetzt den
  individuellen `stop_loss_price` einer Position (von Claude pro Short-Order
  genannt, in `positions.stop_loss_price` gespeichert, im Report angezeigt),
  falls vorhanden - vorher wurde er zwar gespeichert und angezeigt, aber vom
  Pflicht-Sweep selbst nie gelesen; es zählte ausnahmslos der globale
  `risk_config.short_stop_loss_pct`. Fehlt der individuelle Wert für eine
  Position weiterhin, bleibt der bisherige globale Prozent-Fallback
  unverändert. **Bewusst vorgezogen aus der Oktober-Tiefenreflexions-Liste
  (Kap. 6.12.3)** statt bis dahin gesammelt zu warten: die Lücke betraf den
  Stop-Loss-Mechanismus selbst (nicht die Anlagelogik) - Prompt und Report
  suggerierten Claude/dem Leser ein individuelles Sicherheitsnetz, das
  faktisch nie griff. Kein neuer Mechanismus, keine Änderung an
  Anlagekriterien - reine Reparatur eines bereits bestehenden, dokumentierten
  Pflicht-Guardrails. Siehe `evaluate_short_positions_for_stop_loss`-
  Docstring/Kommentar für die vollständige Begründung.
- **Watchlist entscheidet über den Ausführungsweg (echt/simuliert), nicht
  Claudes eigene Angabe (2026-09-22, `v14`, 17-Punkte-Audit Fund #2, HIGH):**
  `execution.execute_proposed_orders` ermittelt vor dem `_route_fill`-Aufruf
  jetzt das in der Watchlist (`config/watchlist.yaml`) hinterlegte
  `instrument_type` für das Order-Symbol und nutzt DIESES für die
  Real/Simuliert-Entscheidung - vorher entschied ausschliesslich Claudes
  eigene `instrument_type`-Angabe im JSON, ungeprüft gegen die Watchlist.
  Eine abweichende Angabe für ein real gelistetes Symbol (z.B. `NVDA` mit
  `instrument_type: mini_future` statt `equity`) hätte die Order unbemerkt
  rein simuliert gebucht, während DB-Zustand und echtes Alpaca-Konto
  dauerhaft auseinanderdriften. Eine Abweichung wird jetzt als `log.warning`
  dokumentiert statt still übernommen zu werden; fehlt der Symbol-Eintrag in
  der Watchlist (sollte die LOW-3-Universumsprüfung in `evaluate_order`
  ohnehin bereits verhindern), bleibt Claudes Angabe unverändert der
  Fallback. **Bewusst vorgezogen aus der Oktober-Tiefenreflexions-Liste**
  (zweiter offener HIGH-Punkt aus dem Audit) - inhaltlich die Kehrseite des
  ungeklärten 8.9.-Vorfalls (`INCIDENT_2026-09-08.md`, `SECURITY-REVIEW.md`):
  dort ging es um unautorisierte Trades OHNE Pipeline-Ursprung, hier um das
  Risiko, dass die Pipeline selbst - bei einer falschen Selbstauskunft
  Claudes - unbemerkt am echten Broker vorbei- oder gegen ihn ausführt.
  Betrifft ausschliesslich die Real/Simuliert-Weiche selbst, keine Änderung
  an Risikoprüfung, Anlagekriterien oder den in DB/Report gespeicherten
  `instrument_type`-Werten.
- **Aufräumarbeiten ohne Verhaltensänderung aus dem 17-Punkte-Audit (2026-09-22,
  `v15`, Sammel-Ausnahme):** fünf Audit-Funde ohne akute Dringlichkeit wurden
  in EINEM Durchgang statt einzeln vorgezogen, weil sie alle derselben
  Kategorie angehören - reine Test-/Dokumentations-/Tote-Code-Aufräumarbeit,
  ohne jede Änderung an Anlagelogik, Guardrails oder Ausführungsverhalten
  (im Unterschied zu v13/v14 oben, die beide echtes Verhalten korrigiert
  haben). Jeder Punkt ist trotzdem ein eigener, für sich lauffähiger Commit:
  - **Fund #5:** toten Code entfernt (`PortfolioContext.
    structured_products_notional`, Vorgänger-Logik von `leveraged_notional`,
    ohne verbleibende Aufrufer).
  - **Fund #8:** README-Architekturabschnitt um bis dahin fehlende Module
    ergänzt (`boundary_conditions`, `correlation`, `data_quality`, `db`,
    `deep_reflection_prompt`/`_schema`, `event_calendar`, `fundamentals`,
    `market_phase`, `momentum_baseline`, `news_feed`, `segment_basket`,
    `stress_test`-Trio) sowie den veralteten "wöchentlich"-Cron-Hinweis
    korrigiert.
  - **Fund #9:** Regressionstests für `pipeline._run_data_quality_checks`
    und `pipeline._check_boundary_conditions` ergänzt (`_maybe_run_deep_
    reflection` hatte bereits Abdeckung).
  - **Fund #10:** Tests für `config.RiskConfig.from_yaml`/`config.AppConfig.
    load` über den echten Datei-/Env-Ladepfad ergänzt (vorher nur indirekt
    über ein lokal vorhandenes `.env`/`risk_config.yaml` getestet).
  - **Fund #14:** Tests für `data_fetch.py`s Rechenhelfer (`_pct_change`,
    `_sma`, `_annualized_volatility`) ergänzt, inkl. Grenzfällen (exakte
    Mindestlänge, konstante Rendite -> Volatilität 0).

  Diese Sammel-Ausnahme gilt AUSSCHLIESSLICH für diese fünf, explizit
  genannten, verhaltensneutralen Punkte - **keine allgemeine
  "Aufräum-Ausnahme"-Regel** für künftige Funde; jeder Fund mit
  Verhaltensänderung (wie v13/v14) bekommt weiterhin seine eigene,
  individuell begründete Versionsnummer.
- **Hebel-Cap bewertet strukturierte Produkte jetzt mit Basiswert-Kurs-
  Fallback (2026-09-22, `v16`, 17-Punkte-Audit Fund #6):**
  `PortfolioContext.leveraged_notional` (genutzt von
  `check_structured_products_cap` und `check_circuit_breaker`) nutzt bei
  fehlendem aktuellem Kurs für ein strukturiertes Produkt jetzt denselben
  Fallback wie `execution.resolve_price`: zuerst der Kurs des Symbols
  selbst, sonst der seines Basiswerts (`OpenPosition.underlying_symbol`,
  neu aus `positions.underlying_symbol` durchgereicht), erst danach der
  Einstandskurs als letzter Fallback. Vorher fiel diese Bewertung bei
  fehlendem eigenen Kurs SOFORT auf den permanent unveränderlichen
  Einstandskurs zurück - der Hebel-Cap/Circuit-Breaker sah damit dauerhaft
  denselben Wert, selbst wenn sich der (über den Basiswert-Proxy
  eigentlich bekannte) tatsächliche Kurs des Produkts längst bewegt hatte.
  Betrifft ausschliesslich die Hebel-spezifischen Checks - `compute_nav`
  und die übrigen NAV-basierten Guardrails (Segmentgewicht, Top-3-
  Konzentration etc.) nutzen weiterhin ihren bisherigen, unveränderten
  Kurs-Fallback (derselbe grundsätzliche Gap besteht dort potenziell auch,
  ist aber nicht Teil dieses Funds - bewusst minimal gehalten).
- **Segment-/Konzentrations-Checks rechnen Short-Exposure jetzt gegen statt
  brutto zu addieren (2026-09-22, `v17`, 17-Punkte-Audit Fund #7):**
  `check_segment_weight`, `check_correlated_segment_exposure`,
  `check_micro_cap_exposure` und `check_top3_concentration` summierten
  Long- und Short-Positionen bisher UNGERICHTET (`+quantity*price`
  unabhängig von der Positionsseite) - inkonsistent zu `compute_nav`, das
  eine Short-Position korrekt gegenrechnet (`-quantity*price`). Neue
  Helfer `_signed_position_value`/`_signed_order_notional` stellen
  dieselbe Vorzeichen-Logik wie `compute_nav` her; der Limit-Vergleich
  nutzt danach bewusst `abs()` des resultierenden Netto-Werts statt des
  rohen Netto-Werts - ein reiner Netto-Vergleich würde die Leitplanke für
  eine grosse NETTO-SHORT-Position stillschweigend wirkungslos machen
  (eine negative Zahl ist nie `> limit`). Am direktesten real auslösbar
  über `check_top3_concentration` mit einer SHORT-Position in einem
  Titel wie TSDD (2x-Short-TSLA-ETF, `leveraged: true`, kein
  Segment/CapTier in der Watchlist) - die anderen drei Checks greifen für
  ein solches Symbol gar nicht erst, da sie ein bekanntes Segment/CapTier
  voraussetzen. **Bewusst NICHT Teil dieses Funds:**
  `PortfolioContext.leveraged_notional` (Hebel-Cap/Circuit-Breaker) hat
  denselben ungerichteten Summierungs-Fehler, ist aber nicht in Fund #7
  benannt (siehe v16/Fund #6, der denselben Denominator nur beim
  Kurs-Fallback angefasst hat) - bleibt als mögliches künftiges
  Audit-Thema offen.
- **Letzte 6 unkritische Audit-Punkte (2026-09-22, `v18`, Sammel-Ausnahme):**
  alle ohne Risikorelevanz - reine Robustheits-/Dokumentations-/Test-
  Verbesserungen, zwei davon bewusst NICHT geändert (siehe unten). Wie bei
  `v15` ein einziger Durchgang, aber jeder Punkt ein eigener Commit.
  - **Fund #4:** ein unerwarteter Fehler IM Short-Stop-Loss-SWEEP SELBST
    (nicht nur bei einem einzelnen Pflicht-Cover, der bereits abgefangen
    wird) liess vorher den GESAMTEN Lauf ungefangen abstürzen, bevor
    überhaupt ein Report geschrieben wurde - der dokumentationspflichtige
    Sweep fehlte dann komplett statt sichtbar zu sein. Neuer, unit-
    testbarer Helper `pipeline._run_short_stop_loss_sweep_safely` (analog
    zu `_run_data_quality_checks`/`_check_boundary_conditions`) fängt den
    Fehler ab; `reporting.generate_report` bekommt einen neuen
    `stop_loss_sweep_error`-Parameter und zeigt ihn als eigenen,
    prominenten Abschnitt.
  - **Fund #12:** `boundary_conditions.evaluate_boundary_conditions` prüfte
    bei strukturierten Produkten AUSSCHLIESSLICH den (nie vorhandenen)
    Kurs des Produkts selbst - eine price_above/price_below-Randbedingung
    blieb dadurch permanent offen, unabhängig vom tatsächlichen
    Kursverlauf des Basiswerts. Jetzt derselbe Basiswert-Kurs-Fallback wie
    bei `execution.resolve_price`/`risk_guardrails.leveraged_notional`
    (v16); `db.get_open_boundary_conditions` liefert dafür
    `underlying_symbol` per JOIN gegen `positions` mit (kein
    Schema-Wechsel nötig). Der Report zeigt jetzt zusätzlich, wenn eine
    Randbedingung über den Basiswert approximiert wurde.
  - **Fund #13:** README korrigiert - der Short-Stop-Loss "20%" war als
    fixer Fakt dargestellt, ist aber der konfigurierbare
    `short_stop_loss_pct` in `risk_config.yaml` (mit `v13`-Hinweis auf den
    individuellen Override).
  - **Fund #15:** `tests/test_stress_test_prompt.py` ergänzt (analog zu
    `test_deep_reflection_prompt.py`) - `build_stress_test_user_prompt`
    war die einzige Prompt-Bau-Funktion ohne eigene Testdatei.
  - **Fund #16 (bewusst NICHT geändert):** die drei yfinance-Downloads pro
    Lauf (`data_fetch.fetch_market_snapshots`/`fetch_price_histories` in
    `_run_data_quality_checks`, beide 90-Tage-Fenster über teils
    überlappende Symbole, sowie `metrics.py`s eigener Download für die
    NAV-Rekonstruktion) liessen sich bei genauerer Prüfung NICHT risikofrei
    zusammenführen: `fetch_price_histories`s eigener Docstring markiert die
    Trennung von `fetch_market_snapshots` bereits als BEWUSSTE
    Entscheidung ("already-incident-prone" - yfinances Multi-Ticker-
    DataFrame-Form ist eine bekannte Fehlerquelle) statt versehentlicher
    Duplikation; ein Zusammenführen würde ausserdem eine subtile
    Divergenz einführen (`fetch_market_snapshots` verwirft ein Symbol
    komplett, wenn dessen "Volume"-Spalte fehlt/einen KeyError wirft -
    `fetch_price_histories` prüft nur "Close" und wäre davon nicht
    betroffen), was welche Symbole im Datenqualitäts-Report auftauchen
    unbemerkt ändern könnte. `metrics.py`s Download deckt ausserdem einen
    fundamental anderen Zeitraum ab (seit Projektstart, nicht 90 Tage
    rollierend) und ist damit architektonisch nicht sinnvoll
    zusammenführbar. Bleibt wie dokumentiert bestehen statt einer
    unsicheren Änderung.
  - **Fund #17 (bewusst NICHT geändert):** `execution.execute_proposed_orders`
    fragt `db.get_open_positions`/`portfolio_row` (Cash) pro Order neu ab
    (Zeilen rund um `build_context`/`db.get_portfolio` im Order-Loop) -
    das ist KEINE Ineffizienz, sondern korrektheitskritisch: jede Order
    innerhalb desselben Laufs muss die Kap.-6.8-Guardrails auf Basis des
    kumulierten Cash-/Positionsstands ALLER vorherigen Orders DESSELBEN
    Laufs prüfen (siehe Korrelations-Beobachtungs-Kommentar direkt im
    Code). Eine In-Memory-Optimierung müsste die exakte
    Positions-/Cash-Mutationslogik von `db.upsert_open_position`/
    `reduce_or_close_position`/`update_cash_balance` (gewichteter
    Durchschnitts-Einstandspreis, Teil-/Vollschliessung, Transaktionskosten)
    verlustfrei in Python nachbauen - zwei parallele Implementierungen
    derselben Mutationslogik, die bei jeder künftigen Änderung an einer
    Stelle auseinanderlaufen können. Bei einer lokalen SQLite-DB mit
    wenigen Positionen und typischerweise einstelliger Order-Anzahl pro
    Lauf ist der tatsächliche Performance-Gewinn vernachlässigbar - das
    Risiko einer Cash-/Positions-Divergenz überwiegt den Nutzen klar.
    Bleibt wie dokumentiert bestehen statt einer unsicheren Änderung.
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

Versionen in `requirements.txt` sind exakt gepinnt (2026-09-18, Repo-Härtung)
statt offener `>=`-Ranges - siehe Kommentar dort für die Update-Prozedur.

**Optional, für Beiträge/Commits (Repo-Hygiene):**

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Richtet einen lokalen Git-Hook ein, der versehentliche Secret-Commits
(API-Keys, Tokens) via `detect-secrets` blockiert (`.pre-commit-config.yaml`).
Läuft zusätzlich bei jedem Push/PR auch in CI
(`.github/workflows/pre-commit.yml`) - der lokale Hook kann übersprungen
werden, die CI-Prüfung nicht. Ein echter Fund darf nicht per `--no-verify`
umgangen werden; ein bekannter False-Positive wird stattdessen per
`detect-secrets scan --baseline .secrets.baseline` neu erzeugt und
mitcommittet.

Dependabot (`.github/dependabot.yml`) öffnet wöchentlich automatische
Update-PRs für Python-Abhängigkeiten und die GitHub-Actions-Workflows.
Für automatische **Security**-Updates zusätzlich "Dependabot alerts" und
"Dependabot security updates" unter *Settings → Code security* aktivieren
(Repo-Einstellung, nicht per Datei setzbar).

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

Jede automatische Zwangsschliessung einer Short-Position (Kurs auf oder
über der Stop-Loss-Schwelle) wird an zwei Stellen dokumentiert. Die
Schwelle ist **kein fixer Wert**, sondern der konfigurierbare
`short_stop_loss_pct` in `config/risk_config.yaml` (aktuell -0.20, siehe
Kommentar dort) - änderbar ohne Code-Änderung. Seit `v13` (17-Punkte-Audit
Fund #1) hat ausserdem ein individueller, pro Position gespeicherter
`stop_loss_price` (z.B. von Claude bei der Short-Order genannt) Vorrang
vor diesem globalen Prozentwert, falls einer gesetzt ist - siehe
`risk_guardrails.evaluate_short_positions_for_stop_loss`.

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
- **TEMPORÄR, nach dem Portfolio-Reset zu entfernen (2026-09-21):**
  `pipeline._pilot_phase_positions_still_open` + ihr Aufruf in `run()` sind
  eine einmalige Übergangs-Sicherheitsprüfung Pilotphase → offizielle Studie
  (siehe `RESET_2026-09-21.md`) - solange der geplante Reset (alle
  Pilotphase-Positionen bei Alpaca UND lokal schliessen) noch nicht
  durchgeführt wurde, bricht ein Lauf sauber ab (kein Marktdaten-Abruf, kein
  Claude-Aufruf), statt auf dem verschmutzten Zustand als "Tag 1" zu starten.
  Kein dauerhafter Guardrail - nach erfolgreichem Reset wird die Prüfung für
  immer wirkungslos (siehe Docstring dort) und kann komplett entfernt werden.
