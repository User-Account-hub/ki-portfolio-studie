-- SQLite-Schema für die KI-gestützte Portfolio-Fallstudie.
-- Vier Tabellen: portfolios, decisions, positions, trades.
-- (decisions steht vor positions/trades im Skript, da trades per FK
--  darauf verweist; SQLite prüft FK-Ziele erst zur DML-Zeit, die
--  Reihenfolge dient hier nur der Lesbarkeit.)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS portfolios (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT NOT NULL UNIQUE,
    currency                TEXT NOT NULL DEFAULT 'USD',
    initial_cash_balance    REAL NOT NULL,
    cash_balance            REAL NOT NULL,
    benchmark_symbol        TEXT NOT NULL DEFAULT 'SPY',
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Jeder Pipeline-Lauf erzeugt (mindestens) einen Decision-Eintrag:
-- den Prompt, Claudes Rohantwort, das Ergebnis der Risikoprüfung und
-- ob/wie ausgeführt wurde. Erzwungene Aktionen (z.B. Short-Stop-Loss)
-- werden ebenfalls als Decision protokolliert (forced_action = 1),
-- damit sie dokumentationspflichtig nachvollziehbar sind.
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id),
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    model               TEXT NOT NULL,
    prompt              TEXT NOT NULL,
    raw_response        TEXT,
    proposed_orders     TEXT,          -- JSON-Array der von Claude vorgeschlagenen Orders
    risk_check_result   TEXT,          -- JSON: pro Order Freigabe/Ablehnung + Begründung
    rationale           TEXT,          -- Klartext-Begründung (Claude oder Guardrail)
    forced_action       INTEGER NOT NULL DEFAULT 0,  -- 1 = z.B. Stop-Loss-Zwangsschliessung
    approved            INTEGER NOT NULL DEFAULT 0,
    executed            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id),
    symbol              TEXT NOT NULL,
    instrument_type     TEXT NOT NULL CHECK (instrument_type IN
                            ('equity', 'etf', 'leverage_certificate', 'mini_future', 'warrant')),
    underlying_symbol   TEXT,          -- Pflicht bei strukturierten Produkten
    side                TEXT NOT NULL CHECK (side IN ('long', 'short')),
    quantity            REAL NOT NULL CHECK (quantity >= 0),  -- 0 bei geschlossener Position
    avg_entry_price     REAL NOT NULL,
    stop_loss_price     REAL,          -- gesetzt bei Short-Positionen
    status              TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    realized_pnl        REAL,          -- gesetzt beim Schliessen
    opened_at           TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at           TEXT,
    closure_reason      TEXT,          -- z.B. 'claude_decision', 'short_stop_loss_forced'
    closure_notes       TEXT,          -- Dokumentationspflicht bei Stop-Loss-Trigger
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id),
    position_id         INTEGER REFERENCES positions(id),
    decision_id         INTEGER REFERENCES decisions(id),
    symbol              TEXT NOT NULL,
    instrument_type     TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('buy', 'sell', 'short', 'cover')),
    quantity            REAL NOT NULL CHECK (quantity > 0),
    price               REAL NOT NULL,
    notional            REAL NOT NULL,
    -- Feste Spread/Slippage-Pauschale (transaction_cost_pct_of_notional in
    -- risk_config.yaml), tatsaechlich vom cash_balance abgezogen (siehe
    -- execution.py) - NICHT nur als Limitation dokumentiert. DEFAULT 0
    -- haelt Trades von vor dieser Aenderung unveraendert (keine rueckwirkend
    -- erfundenen Kosten).
    transaction_cost    REAL NOT NULL DEFAULT 0,
    order_type          TEXT NOT NULL DEFAULT 'market',
    broker_order_id     TEXT,          -- Alpaca Order-ID, NULL bei simulierten Trades
    source              TEXT NOT NULL CHECK (source IN ('alpaca', 'manual_simulation')),
    status              TEXT NOT NULL CHECK (status IN ('filled', 'simulated', 'rejected', 'pending')),
    rejection_reason    TEXT,          -- gesetzt, wenn status = 'rejected'
    executed_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ein Eintrag pro Pipeline-Lauf: NAV zu Lauf-Beginn (vor Stop-Loss-Sweep/
-- Claude-Entscheidung). Einziger Zweck: dem Kap.-6.8-Circuit-Breaker
-- (risk_guardrails.check_circuit_breaker) den tatsaechlichen historischen
-- NAV-Hoechststand liefern (echtes MAX ueber alle bisherigen Laeufe, nicht
-- nur eine Naeherung aus Startkapital/aktuellem Stand). Ersetzt NICHT die
-- Wochenverlauf-Rekonstruktion in metrics.py (die bleibt Trade-Replay-basiert
-- fuer den vollen Report-Chart inkl. Sharpe/Drawdown) - siehe README.
CREATE TABLE IF NOT EXISTS nav_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id),
    recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
    nav             REAL NOT NULL
);

-- Randbedingungs-Tracking (Thesis Kap. 7, 2026-09-17): jede von Claude bei
-- einer Kauf-/Short-Empfehlung genannte Randbedingung (siehe
-- order_schema.BoundaryCondition) wird hier gespeichert und bei jedem Lauf
-- gegen die aktuellen Kurse geprueft, solange die zugehoerige Position
-- offen ist (siehe boundary_conditions.py). Rein dokumentarisch - ein
-- Trigger loest KEINE automatische Order aus.
CREATE TABLE IF NOT EXISTS boundary_conditions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id),
    position_id         INTEGER NOT NULL REFERENCES positions(id),
    decision_id         INTEGER REFERENCES decisions(id),
    symbol              TEXT NOT NULL,
    description         TEXT NOT NULL,
    check_type          TEXT NOT NULL CHECK (check_type IN ('price_above', 'price_below', 'qualitative')),
    threshold_price     REAL,          -- Pflicht bei price_above/price_below, NULL bei qualitativ
    status              TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'triggered', 'closed_with_position')),
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    triggered_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status ON positions(portfolio_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_portfolio_symbol_date ON trades(portfolio_id, symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_decisions_portfolio_date ON decisions(portfolio_id, created_at);
CREATE INDEX IF NOT EXISTS idx_nav_history_portfolio_recorded ON nav_history(portfolio_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_boundary_conditions_portfolio_status ON boundary_conditions(portfolio_id, status);
CREATE INDEX IF NOT EXISTS idx_boundary_conditions_position ON boundary_conditions(position_id);
