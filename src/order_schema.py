"""Pydantic models for the JSON order format Claude must respond with,
plus the parser that turns Claude's raw text into validated objects."""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

STRUCTURED_INSTRUMENT_TYPES = {"leverage_certificate", "mini_future", "warrant"}
TRADABLE_INSTRUMENT_TYPES = {"equity", "etf"}


class InstrumentType(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    LEVERAGE_CERTIFICATE = "leverage_certificate"
    MINI_FUTURE = "mini_future"
    WARRANT = "warrant"


class OrderSide(str, Enum):
    BUY = "buy"      # long eröffnen/aufstocken
    SELL = "sell"     # long reduzieren/schliessen
    SHORT = "short"    # short eröffnen/aufstocken
    COVER = "cover"    # short reduzieren/schliessen


class ProposedOrder(BaseModel):
    symbol: str
    instrument_type: InstrumentType
    side: OrderSide
    quantity: Optional[float] = Field(default=None, gt=0)
    notional: Optional[float] = Field(default=None, gt=0)
    order_type: str = Field(default="market", pattern="^(market|limit)$")
    limit_price: Optional[float] = Field(default=None, gt=0)
    underlying_symbol: Optional[str] = None
    stop_loss_price: Optional[float] = Field(default=None, gt=0)
    rationale: str

    @model_validator(mode="after")
    def _validate_combination(self) -> "ProposedOrder":
        if self.quantity is None and self.notional is None:
            raise ValueError("Order muss entweder 'quantity' oder 'notional' angeben.")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("Limit-Order benötigt 'limit_price'.")
        if self.instrument_type.value in STRUCTURED_INSTRUMENT_TYPES and not self.underlying_symbol:
            raise ValueError(
                f"Strukturiertes Produkt '{self.symbol}' benötigt 'underlying_symbol'."
            )
        if self.side == OrderSide.SHORT and self.stop_loss_price is None:
            # Wird von risk_guardrails ohnehin auf Basis von short_stop_loss_pct gesetzt,
            # falls Claude keinen expliziten Wert liefert - siehe execution.py.
            pass
        return self


class ClaudeDecisionOutput(BaseModel):
    orders: list[ProposedOrder] = Field(default_factory=list)
    portfolio_commentary: str = ""


class OrderParsingError(Exception):
    """Raised when Claude's response cannot be parsed into valid orders."""


def parse_orders_from_json(raw_text: str) -> ClaudeDecisionOutput:
    """Extracts the first top-level JSON object from raw_text and validates it.

    Claude is instructed to answer with pure JSON, but this tolerates
    surrounding prose/markdown fences defensively.
    """
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        raise OrderParsingError("Keine JSON-Struktur in der Antwort gefunden.")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise OrderParsingError(f"Ungültiges JSON: {exc}") from exc
    try:
        return ClaudeDecisionOutput.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise OrderParsingError(f"Schema-Validierung fehlgeschlagen: {exc}") from exc


def _extract_json_object(text: str) -> Optional[str]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
