"""Market data retrieval via yfinance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    last_price: float
    change_1d_pct: float | None
    sma20: float | None
    sma50: float | None
    volatility_20d_annualized: float | None
    volume: int | None


def fetch_market_snapshots(symbols: list[str], lookback_days: int = 90) -> dict[str, MarketSnapshot]:
    """Fetches recent daily history for `symbols` and derives simple stats.

    Uses a single batched yfinance download to minimise API calls. Symbols
    that yfinance cannot resolve (e.g. placeholder structured-product
    tickers from the watchlist) are silently skipped - the caller falls
    back to the underlying symbol's price for those (see prompt_builder).
    """
    if not symbols:
        return {}

    data = yf.download(
        tickers=symbols,
        period=f"{lookback_days}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

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
