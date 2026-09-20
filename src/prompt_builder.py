"""Builds the system/user prompt sent to Claude for a trading decision."""
from __future__ import annotations

import json

from src.config import RiskConfig, Watchlist
from src.data_fetch import MarketSnapshot

# Prompt-Versionswechsel v2 (2026-09-10, Thesis Kap. 6.2): expliziter
# Renditemaximierungs-Auftrag ("Dein Ziel ist die Maximierung der
# Portfolio-Rendite.") ergaenzt, direkt nach der Einleitung und vor den
# Rahmenbedingungen. Vorher enthielt SYSTEM_PROMPT keine explizite
# Zielformulierung, nur Rolle + Rahmenbedingungen.
#
# Prompt-Versionswechsel v7 (2026-09-17, methodisch wichtigste Aenderung vor
# dem offiziellen Studienstart am 21.09.2026): der pauschale v2-Renditemaxi-
# mierungs-Auftrag wird durch einen sechsteiligen Anforderungskatalog ersetzt,
# der jede Empfehlung an eine nachvollziehbare, falsifizierbare Anlagethese
# bindet statt an reines Kurs-Momentum (Zyklus-Position Kap. 3, Moat-Qualitaet
# Kap. 13, Edge-Typ-Zuordnung, Wirkungsmechanismus + Zeitfenster Kap. 4,
# Portfolio-weite Konsistenz, explizite Abgrenzung von reinem SMA20/50-
# Pattern-Matching). Grund: die bisherigen Reports begruenden Kaeufe fast
# durchgehend allein mit "Momentum"/SMA20/50 (siehe reports/*.md) - das ist
# nicht die intendierte analytische Tiefe der Fallstudie.
#
# Prompt-Versionswechsel v8 (2026-09-17, Thesis Kap. 7): siebte Anforderung
# "RANDBEDINGUNGEN" ergaenzt - fuer jede Kauf-/Short-Empfehlung soll Claude
# mindestens eine konkrete Randbedingung nennen, deren Eintreten die These
# bestaetigen oder entkraeften wuerde, wo moeglich als pruefbare Kursschwelle
# statt nur als Fliesstext. Optional/best-effort (kein Pflichtfeld, siehe
# order_schema.BoundaryCondition) - keine Order wird allein deswegen
# abgelehnt. Kursschwellen-Randbedingungen werden bei jedem Lauf mechanisch
# geprueft (boundary_conditions.py), solange die zugehoerige Position offen
# ist; qualitative (nicht kursbasierte) Randbedingungen werden dokumentiert,
# aber nicht automatisch verifiziert - siehe dortigen Modul-Docstring fuer
# die Begruendung, warum eine vorgetaeuschte Automatisierung vermieden wird.
SYSTEM_PROMPT = """\
Du bist der Portfolio-Analyst einer KI-gestützten Portfolio-Fallstudie im Paper-Trading-Modus \
(kein echtes Geld). Du erhältst den aktuellen Portfolio-Zustand und Marktdaten für ein festes \
Anlage-Universum und schlägst darauf basierend Handelsentscheidungen vor.

Deine Analyse und jede Kaufempfehlung müssen auf einer nachvollziehbaren, testbaren \
Anlagethese basieren - nicht auf reinem Kurs-Momentum oder einer unspezifizierten \
Renditemaximierung. Es gelten sieben Anforderungen:

1. ZYKLUS-POSITION
   Ordne jeden vorgeschlagenen Titel explizit in eine der historischen Zyklusphasen \
ein (Akkumulation / Aufmerksamkeit / Manie / Crash / Rückkehr zum Mittel, Kap. 3). \
Nenne mindestens einen konkreten Indikator, der diese Einordnung stützt. Bevorzuge \
Titel in früher bis mittlerer Zyklusphase. Falls du einen Titel in später \
Manie-Phase empfiehlst, ist eine gesonderte, explizite Begründung zwingend. Gib diese \
Einordnung zusätzlich strukturiert im Feld "cycle_position" an (siehe JSON-Schema) - \
sie wird serverseitig automatisiert gegen eine regelbasierte SMA/Volatilitäts-\
Klassifikation verglichen (rein dokumentarisch, kein Ausschlusskriterium).

2. MOAT-QUALITÄT UND POSITIONSGRÖSSE
   Nutze die Moat-Klassifikation (Wide/Narrow/No Moat, Kap. 13) als Grundlage für \
die Positionsgrösse: Wide-Moat-Titel rechtfertigen tendenziell höhere Konviktion \
als No-Moat-Titel. Bei Micro-/Small-Cap-Titeln ohne belegten Wettbewerbsvorteil \
ist besondere Zurückhaltung geboten - explizites Gegenrisiko benennen.

3. STRUKTURELLER VORTEIL (EDGE) - MIT TYP-ZUORDNUNG
   Benenne für jede Empfehlung den Typ deines analytischen Vorteils: (a) \
Informationsvorteil, (b) Zeithorizont-Vorteil, oder (c) Verhaltensvorteil \
(vermiedene kognitive Verzerrung des Marktes). Eine Empfehlung ohne zuordenbaren \
Vorteil gegenüber Buy-and-Hold ist nicht ausreichend.

4. WIRKUNGSMECHANISMUS UND ZEITFENSTER
   Verknüpfe jede These mit einem Wirkungsmechanismus aus Kap. 4 und gib ein \
realistisches Zeitfenster an, in dem sich die These bestätigen oder widerlegen \
sollte. Eine nicht-falsifizierbare These ist unzureichend.

5. PORTFOLIO-WEITE KONSISTENZPFLICHT
   Deine Einzelempfehlungen müssen mit einer kohärenten Gesamthaltung \
(offensiv/zyklusfrüh vs. defensiv/zyklusspät) vereinbar sein. Widersprüchliche \
Positionen ohne übergeordnete Logik vermeiden.

6. ABGRENZUNG VON REINEM PATTERN-MATCHING
   Eine Begründung, die sich ausschliesslich auf SMA20/50 stützt, erfüllt diese \
Anforderungen NICHT. Technische Indikatoren nur unterstützend.

7. RANDBEDINGUNGEN (KAP. 7)
   Nenne für jede Kauf-/Short-Empfehlung mindestens eine konkrete Randbedingung, \
deren Eintreten deine These bestätigen oder entkräften würde. Formuliere sie, wo \
möglich, als prüfbare Kursschwelle (Feld "check_type": "price_above"/"price_below" \
mit "threshold_price") statt als reinen Fliesstext - das ermöglicht eine \
automatische Prüfung bei jedem Lauf. Ist die Randbedingung nicht sinnvoll in eine \
Kursschwelle übersetzbar (z.B. ein Makro- oder Earnings-Ereignis), gib \
"check_type": "qualitative" an - sie wird dann dokumentiert und dir bei künftigen \
Läufen erneut vorgelegt, aber nicht automatisch verifiziert.

Wichtige Rahmenbedingungen:
- Du darfst NUR Symbole aus dem gelieferten Universum vorschlagen.
- Long- und Short-Positionen auf Aktien/ETFs sind erlaubt, ebenso strukturierte Produkte \
(Hebelzertifikate, Mini-Futures, Optionsscheine) auf Titel des Universums.
- Kein Margin-Trading: Order-Notionals dürfen das verfügbare Cash nicht überschreiten.
- Wo verfügbar, erhältst du je Titel im Universum grobe Fundamentaldaten (Umsatzwachstum, \
Verschuldungsgrad, Free Cashflow - Feld "fundamentals" im JSON-Kontext) - das ist \
ergänzende Information für deine eigene Bewertung, KEIN Ausschlusskriterium. Das \
Universum ist bewusst spekulativ; ein unprofitabler oder hoch verschuldeter Titel \
bleibt uneingeschränkt handelbar.
- Feld "news_context" im JSON-Kontext: ein fester, bei jedem Lauf neu aggregierter \
Textblock aus fünf definierten Nachrichten-Feeds (Titel + Datum der jüngsten Einträge, \
kein Volltext) - rein ergänzender Kontext, KEIN Ausschlusskriterium und keine Garantie \
auf Vollständigkeit oder Relevanz für ein bestimmtes Symbol. Kann fehlen (null), wenn \
die Aggregation an einem Tag fehlschlägt.
- Die tatsächliche Durchsetzung aller Risikolimiten (Positionsgrössen, Tagesverlust-Stop, \
strukturierte-Produkte-Obergrenze, Short-Stop-Loss, max. 1 Trade/Symbol/Tag, max. \
Segmentgewichtung, korrelierte Krypto/Mining-Exposure, Micro-Cap-Sublimit, \
Top-3-Konzentration, Mindest-Cash-Quote, Drawdown-Circuit-Breaker für Hebelpositionen, \
Liquiditätslimit relativ zum Tagesvolumen) erfolgt serverseitig nach deiner Antwort - \
Vorschläge ausserhalb der Limiten werden automatisch abgelehnt und nicht ausgeführt. \
Halte dich trotzdem an die unten genannten Limiten (inkl. "segment"/"cap_tier" je Titel \
im Universum und "volume" je Titel im Marktdaten-Kontext), um unnötige Ablehnungen zu \
vermeiden.
- Antworte AUSSCHLIESSLICH mit einem einzigen validen JSON-Objekt, ohne Markdown-Fences, \
ohne Fliesstext davor oder danach.
- Optional kannst du je Kauf-/Short-Order eine Konviktions-Einschätzung angeben (Feld \
"conviction": "high"/"medium"/"low"). Sie skaliert die Positionsgrösse serverseitig NUR \
LEICHT (high ×1.15, medium ×1.0, low ×0.8) - bewusst schwach, da eine Selbsteinschätzung \
deiner eigenen Sicherheit kein verlässliches, kalibriertes Signal ist. Kein Pflichtfeld; \
ohne Angabe wird wie "medium" (kein Effekt) behandelt.

JSON-Ausgabeschema:
{
  "orders": [
    {
      "symbol": "string (muss im Universum sein)",
      "instrument_type": "equity|etf|leverage_certificate|mini_future|warrant",
      "side": "buy|sell|short|cover",
      "quantity": number,            // ODER "notional", nicht beides zwingend, aber mind. eines
      "notional": number,
      "order_type": "market|limit",
      "limit_price": number,          // Pflicht wenn order_type = limit
      "underlying_symbol": "string",  // Pflicht bei strukturierten Produkten
      "stop_loss_price": number,      // empfohlen bei "short"
      "boundary_conditions": [         // optional, mind. eine bei buy/short empfohlen (Kap. 7)
        {
          "description": "string - kurz, was genau eintreten müsste",
          "check_type": "price_above|price_below|qualitative",
          "threshold_price": number    // Pflicht bei price_above/price_below
        }
      ],
      "conviction": "high|medium|low", // optional, siehe oben - nur leichter Effekt
      "cycle_position": "accumulation|attention|mania|crash|reversion_to_mean", // optional, siehe Anforderung 1
      "rationale": "string - kurze Begründung"
    }
  ],
  "portfolio_commentary": "string - kurze Gesamteinschätzung"
}

Wenn du aktuell keine Handlung empfiehlst, gib "orders": [] zurück und begründe dies in \
"portfolio_commentary".
"""


def build_user_prompt(
    portfolio_row,
    open_positions: list,
    watchlist: Watchlist,
    snapshots: dict[str, MarketSnapshot],
    risk_config: RiskConfig,
    latest_reflection=None,
    triggered_boundary_conditions: list | None = None,
    still_open_boundary_conditions: list | None = None,
    earnings_warnings: list | None = None,
    macro_events: list | None = None,
    fundamentals: dict | None = None,
    news_text_block: str | None = None,
) -> str:
    portfolio_state = {
        "name": portfolio_row["name"],
        "currency": portfolio_row["currency"],
        "cash_balance": portfolio_row["cash_balance"],
        "benchmark_symbol": portfolio_row["benchmark_symbol"],
    }

    positions_state = [
        {
            "symbol": p["symbol"],
            "instrument_type": p["instrument_type"],
            "side": p["side"],
            "quantity": p["quantity"],
            "avg_entry_price": p["avg_entry_price"],
            "stop_loss_price": p["stop_loss_price"],
        }
        for p in open_positions
    ]

    universe = [
        {
            "symbol": s.symbol,
            "instrument_type": s.instrument_type,
            "underlying_symbol": s.underlying_symbol,
            "segment": s.segment,
            "cap_tier": s.cap_tier,
        }
        for s in watchlist.symbols
    ]

    market_data = {
        symbol: {
            "last_price": snap.last_price,
            "change_1d_pct": snap.change_1d_pct,
            "sma20": snap.sma20,
            "sma50": snap.sma50,
            "volatility_20d_annualized": snap.volatility_20d_annualized,
            # Kap. 6.13 Liquiditätslimit (2026-09-21) - zuletzt bekanntes
            # Tagesvolumen, siehe risk_guardrails.check_liquidity_limit.
            "volume": snap.volume,
        }
        for symbol, snap in snapshots.items()
    }

    limits = {
        "max_position_size_pct_of_portfolio": risk_config.max_position_size_pct_of_portfolio,
        "max_trade_notional_pct_of_nav": risk_config.max_trade_notional_pct_of_nav,
        "daily_loss_stop_pct": risk_config.daily_loss_stop_pct,
        "max_trades_per_symbol_per_day": risk_config.max_trades_per_symbol_per_day,
        "structured_products_max_notional_pct_of_nav": risk_config.structured_products_max_notional_pct_of_nav,
        "short_stop_loss_pct": risk_config.short_stop_loss_pct,
        "max_segment_weight_pct_of_nav": risk_config.max_segment_weight_pct_of_nav,
        "correlated_crypto_mining_segments": risk_config.correlated_crypto_mining_segments,
        "max_correlated_crypto_mining_pct_of_nav": risk_config.max_correlated_crypto_mining_pct_of_nav,
        "max_micro_cap_pct_of_nav": risk_config.max_micro_cap_pct_of_nav,
        "max_top3_concentration_pct_of_nav": risk_config.max_top3_concentration_pct_of_nav,
        "min_cash_pct_of_nav": risk_config.min_cash_pct_of_nav,
        "circuit_breaker_drawdown_pct": risk_config.circuit_breaker_drawdown_pct,
        # Abgestufter Drawdown-Schutz (2026-09-19, siehe risk_guardrails.
        # drawdown_position_size_factor) - reduziert max_position_size_pct_
        # of_portfolio je nach aktuellem Drawdown seit dem historischen NAV-
        # Hoechststand; gilt NICHT fuer Hebelpositionen (dafuer weiterhin nur
        # circuit_breaker_drawdown_pct oben).
        "drawdown_tier1_pct": risk_config.drawdown_tier1_pct,
        "drawdown_tier1_position_size_factor": risk_config.drawdown_tier1_position_size_factor,
        "drawdown_tier2_pct": risk_config.drawdown_tier2_pct,
        "drawdown_tier2_position_size_factor": risk_config.drawdown_tier2_position_size_factor,
        # Kap. 6.13 (2026-09-21, siehe risk_guardrails.check_liquidity_limit) -
        # max. Anteil der Order-Menge am Tagesvolumen ("volume" je Titel oben).
        "max_order_pct_of_avg_daily_volume": risk_config.max_order_pct_of_avg_daily_volume,
    }

    payload = {
        "portfolio": portfolio_state,
        "open_positions": positions_state,
        "universe": universe,
        "market_data": market_data,
        "risk_limits": limits,
        # Kap. 6.12.3: Nachwirkung der letzten monatlichen Tiefenreflexion
        # (siehe deep_reflection_prompt.py) - None vor der ersten Reflexion
        # (Woche 5) oder falls die letzte Reflexion nicht geparst werden
        # konnte. Bleibt bis zur naechsten Reflexion unveraendert im
        # taeglichen Prompt, damit ihre Erkenntnisse tatsaechlich nachwirken
        # statt nur im Report zu stehen.
        "latest_deep_reflection": _summarize_reflection_for_prompt(latest_reflection),
        # Kap. 7: Randbedingungs-Tracking (siehe boundary_conditions.py) -
        # "triggered" sind die in DIESEM Lauf ausgeloesten (Kurs hat die
        # genannte Schwelle erreicht), "still_open" alle weiterhin
        # unausgeloesten (inkl. qualitativer, nie automatisch geprueften)
        # Randbedingungen zu noch offenen Positionen.
        "boundary_conditions": {
            "triggered_this_run": [_summarize_boundary_condition(c) for c in (triggered_boundary_conditions or [])],
            "still_open": [_summarize_boundary_condition(c) for c in (still_open_boundary_conditions or [])],
        },
        # Event-Kalender-Hinweis (yfinance-Earnings + hardcodierte FOMC-/
        # CPI-Termine, siehe event_calendar.py) - rein informativ, KEIN
        # Verbot. Du entscheidest selbst, ob/wie du das erhoehte
        # Ereignisrisiko in Positionsgroesse oder Timing einbeziehst.
        "upcoming_events": {
            "earnings_within_3_trading_days": [
                _summarize_earnings_warning(w) for w in (earnings_warnings or [])
            ],
            "macro_events_within_3_trading_days": [
                _summarize_macro_event(m) for m in (macro_events or [])
            ],
        },
        # Weicher Qualitäts-Score (siehe fundamentals.py) - grobe
        # Fundamentaldaten je Titel, wo verfügbar. AUSDRÜCKLICH kein
        # Ausschlusskriterium (siehe Rahmenbedingungen oben) - nur
        # ergänzender Kontext für deine eigene Bewertung. Symbole ohne
        # verfügbare Daten (ETFs, strukturierte Produkte, sehr junge
        # Börsengänge) fehlen hier einfach, statt mit Platzhalterwerten
        # aufzutauchen.
        "fundamentals": {
            symbol: {
                "revenue_growth": snap.revenue_growth,
                "debt_to_equity": snap.debt_to_equity,
                "free_cash_flow": snap.free_cash_flow,
            }
            for symbol, snap in (fundamentals or {}).items()
        },
        # Kap. 12.2 (2026-09-21, siehe src/news_feed.py fuer die vollstaendige
        # Begruendung und die - ab Studienstart feste - Feed-Liste): fester,
        # taeglich neu aggregierter Textblock aus fuenf definierten RSS-Feeds
        # (Titel + Datum der juengsten Eintraege, kein Volltext). Rein
        # informativ, KEIN Ausschlusskriterium. None/leer, falls die
        # Aggregation fehlgeschlagen ist (siehe pipeline._fetch_news_context) -
        # dann einfach ohne diesen Kontext entscheiden, kein Grund zum Abbruch.
        "news_context": news_text_block or None,
    }
    return (
        "Aktueller Portfolio-Zustand, Marktdaten und Risikolimiten (JSON):\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Erstelle deine Handelsentscheidung gemäss dem im System-Prompt definierten JSON-Schema."
    )


def _summarize_reflection_for_prompt(latest_reflection) -> dict | None:
    """`latest_reflection` ist eine 'decisions'-Zeile mit model='deep_reflection'
    (siehe db.get_latest_reflection) oder None. Extrahiert nur die fuer die
    Tagesentscheidung relevanten Felder aus ihrem raw_response-JSON - nicht
    den vollen Reflexions-Prompt/-Output, um den Tages-Prompt schlank zu
    halten. Robust gegen fehlende/kaputte Daten (z.B. eine Reflexion, deren
    raw_response aus irgendeinem Grund nicht mehr valide JSON ist) - liefert
    dann None statt den gesamten Tages-Prompt-Aufbau zum Absturz zu bringen."""
    if latest_reflection is None:
        return None
    try:
        data = json.loads(latest_reflection["raw_response"])
    except (TypeError, ValueError):
        return None
    return {
        "created_at": latest_reflection["created_at"],
        "theses_falsified_or_overdue": data.get("theses_falsified_or_overdue", []),
        "pattern_matching_concerns": data.get("pattern_matching_concerns"),
        "portfolio_stance_assessment": data.get("portfolio_stance_assessment"),
    }


def _summarize_boundary_condition(condition) -> dict:
    """`condition` ist ein boundary_conditions.BoundaryConditionCheck -
    reicht nur die fuer die Tagesentscheidung relevanten Felder durch (nicht
    die interne DB-`id`/`position_id`)."""
    return {
        "symbol": condition.symbol,
        "description": condition.description,
        "check_type": condition.check_type,
        "threshold_price": condition.threshold_price,
    }


def _summarize_earnings_warning(warning) -> dict:
    """`warning` ist ein event_calendar.EarningsWarning."""
    return {
        "symbol": warning.symbol,
        "earnings_date": warning.earnings_date.isoformat(),
        "trading_days_until": warning.trading_days_until,
        "note": f"{warning.symbol} berichtet in {warning.trading_days_until} Handelstag(en) - erhöhtes Ereignisrisiko.",
    }


def _summarize_macro_event(event) -> dict:
    """`event` ist ein event_calendar.MacroEvent."""
    return {
        "name": event.name,
        "event_date": event.event_date.isoformat(),
        "trading_days_until": event.trading_days_until,
        "note": (
            f"{event.name}-Termin in {event.trading_days_until} Handelstag(en) - "
            "erhöhtes Ereignisrisiko fürs gesamte Portfolio."
        ),
    }
