"""Market data retrieval via yfinance."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

_YFINANCE_LOGGER_NAME = "yfinance"


class _DowngradeKnownUnresolvableSymbols(logging.Filter):
    """Downgrades yfinance's ERROR logs to INFO for symbols we already know it
    can't resolve (our watchlist's structured-product placeholders, e.g.
    MINI-NVDA-LONG-1) - yfinance logs "No data found" at ERROR for every
    ticker it can't fetch, but for these that's expected, not a real failure.
    Anything about a symbol NOT in this set is left untouched at ERROR, so
    genuine data-fetch problems still stand out.
    """

    def __init__(self, known_unresolvable_symbols: set[str]) -> None:
        super().__init__()
        self._symbols = known_unresolvable_symbols

    def filter(self, record: logging.LogRecord) -> bool:
        if self._symbols and record.levelno >= logging.ERROR:
            message = record.getMessage()
            if any(symbol in message for symbol in self._symbols):
                record.levelno = logging.INFO
                record.levelname = "INFO"
        return True


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    last_price: float
    change_1d_pct: float | None
    sma20: float | None
    sma50: float | None
    volatility_20d_annualized: float | None
    volume: int | None


def fetch_market_snapshots(
    symbols: list[str],
    lookback_days: int = 90,
    known_unresolvable_symbols: set[str] | None = None,
) -> dict[str, MarketSnapshot]:
    """Fetches recent daily history for `symbols` and derives simple stats.

    Uses a single batched yfinance download to minimise API calls. Symbols
    that yfinance cannot resolve (e.g. placeholder structured-product
    tickers from the watchlist) are silently skipped - the caller falls
    back to the underlying symbol's price for those (see prompt_builder).

    `known_unresolvable_symbols` (typically the watchlist's structured-product
    symbols) downgrades yfinance's ERROR-level "No data found" log noise for
    exactly those symbols to INFO, since that failure is expected for them.
    """
    if not symbols:
        return {}

    yf_logger = logging.getLogger(_YFINANCE_LOGGER_NAME)
    downgrade_filter = _DowngradeKnownUnresolvableSymbols(known_unresolvable_symbols or set())
    yf_logger.addFilter(downgrade_filter)
    try:
        data = yf.download(
            tickers=symbols,
            period=f"{lookback_days}d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    finally:
        yf_logger.removeFilter(downgrade_filter)

    snapshots: dict[str, MarketSnapshot] = {}
    for symbol in symbols:
        try:
            series = data[symbol] if len(symbols) > 1 else data
            close = series["Close"].dropna()
            if close.empty:
                continue
            volume_series = series["Volume"].dropna()
            snapshots[symbol] = MarketSnapshot(
                symbol=symbol,
                last_price=float(close.iloc[-1]),
                change_1d_pct=_pct_change(close, 1),
                sma20=_sma(close, 20),
                sma50=_sma(close, 50),
                volatility_20d_annualized=_annualized_volatility(close, 20),
                volume=int(volume_series.iloc[-1]) if not volume_series.empty else None,
            )
        except (KeyError, IndexError):
            continue  # Symbol nicht auf yfinance auflösbar (z.B. strukturiertes Produkt)
    return snapshots


def fetch_latest_prices(symbols: list[str]) -> dict[str, float]:
    """Lightweight helper returning just last close prices, used by risk checks."""
    snapshots = fetch_market_snapshots(symbols, lookback_days=5)
    return {s: snap.last_price for s, snap in snapshots.items()}


def _pct_change(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - periods] - 1)


def _sma(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return float(close.rolling(window).mean().iloc[-1])


def _annualized_volatility(close: pd.Series, window: int) -> float | None:
    if len(close) < window + 1:
        return None
    returns = close.pct_change().dropna().iloc[-window:]
    return float(returns.std() * np.sqrt(252))
