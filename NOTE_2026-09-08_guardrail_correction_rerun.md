# Notiz: Einmaliger Korrektur-Lauf am 2026-09-08

**Status:** Kein neuer Rhythmus. Regulärer Takt bleibt Montag + Donnerstag,
15:00 UTC.

## Hintergrund

Am 2026-09-08 fanden **zwei** Pipeline-Läufe statt:

1. **`report_2026-09-08_165019.md` (16:50 UTC)** - der reguläre Lauf des
   Tages. Wegen eines zu diesem Zeitpunkt noch nicht behobenen Bugs in
   `check_trade_notional` (prüfte gegen das mit jedem Fill schrumpfende
   `ctx.cash` statt gegen eine für den Lauf feste NAV-Bezugsgrösse) wurden
   nur **2 von 16** von Claude sinnvoll vorgeschlagenen Orders ausgeführt
   (NVDA, AMD) - alle übrigen 14 wurden fälschlich mit "Trade-Notional
   übersteigt Limit von 5% des Cash" abgelehnt, obwohl sie das eigentlich
   beabsichtigte 5%-NAV-Limit eingehalten hätten.
2. **`report_2026-09-08_192332.md` (19:23 UTC)** - Korrektur-Lauf, nachdem
   der Bug behoben war (Commit `78f5dd3`, `max_trade_notional_pct_of_cash`
   → `max_trade_notional_pct_of_nav`, feste Bezugsgrösse `start_of_run_nav`).
   Claude schlug 15 neue Orders vor (aufbauend auf den bereits bestehenden
   NVDA/AMD-Positionen aus Lauf 1); **alle 15 wurden ausgeführt**, keine
   einzige Ablehnung.

## Warum das kein zusätzlicher Termin ist

- Der 2026-09-07 (Labor Day) war ein Börsenfeiertag - die Pipeline hat an
  diesem Tag korrekt keinen Handelsversuch unternommen (siehe
  `is_trading_day`-Check). Der 8.9. ist damit ohnehin der nachgeholte erste
  reguläre Termin dieser Kalenderwoche.
- Lauf 2 am 8.9. **ersetzt/vervollständigt** den durch den Bug verkrüppelten
  Lauf 1 desselben Tages - er tritt nicht zusätzlich zum regulären
  Montag/Donnerstag-Rhythmus hinzu.
- Der Cron-Zeitplan in `.github/workflows/weekly_pipeline.yml` wurde **nicht**
  verändert und bleibt `0 15 * * 1` (Montag) + `0 15 * * 4` (Donnerstag).

## Was NICHT rückgängig gemacht wurde

Die beiden aus Lauf 1 stammenden Positionen (NVDA, AMD) waren durch die
Guardrails korrekt freigegeben und wurden **nicht** storniert oder
glattgestellt - nur die fälschlich abgelehnten übrigen 14 Vorschläge wurden
in Lauf 2 nachgeholt (mit einer für diesen Zeitpunkt aktuellen, nicht
identischen Neuentscheidung von Claude, da sich Marktdaten zwischen 16:50
und 19:23 UTC leicht verändert haben).
