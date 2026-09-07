"""Tests for src/broker_alpaca.py's post-submit fill polling.

Uses a fake TradingClient (no network) so these run fast and
deterministically; the real Alpaca paper account is exercised separately
and manually, not via automated tests.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.broker_alpaca import submit_equity_order


def make_order(order_id="order-1", status="new", filled_qty=None, filled_avg_price=None):
    return SimpleNamespace(
        id=order_id,
        status=SimpleNamespace(value=status),
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )


class FakeTradingClient:
    """Stands in for alpaca.trading.client.TradingClient. `submit_response`
    is returned once from submit_order(); `poll_sequence` is returned, one
    entry per call, from subsequent get_order_by_id() calls - simulating the
    asynchronous state transitions a real Alpaca order goes through."""

    def __init__(self, submit_response, poll_sequence):
        self.submit_response = submit_response
        self._poll_sequence = list(poll_sequence)
        self.get_order_calls = 0
        self.submitted_requests = []
        self.cancel_calls: list = []

    def submit_order(self, request):
        self.submitted_requests.append(request)
        return self.submit_response

    def get_order_by_id(self, order_id):
        self.get_order_calls += 1
        return self._poll_sequence.pop(0)

    def cancel_order_by_id(self, order_id):
        self.cancel_calls.append(order_id)


def test_submit_equity_order_fills_on_first_poll():
    """The common case: submit_order() returns before the fill lands, but
    it's there one poll later - must be reported as filled, not skipped."""
    client = FakeTradingClient(
        submit_response=make_order(status="new"),
        poll_sequence=[make_order(status="filled", filled_qty="10", filled_avg_price="150.25")],
    )
    fill = submit_equity_order(
        client, "AAPL", 10, "buy", poll_interval_seconds=0, max_poll_attempts=5
    )
    assert fill.status == "filled"
    assert fill.filled_qty == 10.0
    assert fill.filled_price == 150.25
    assert client.get_order_calls == 1


def test_submit_equity_order_fills_after_multiple_polls():
    client = FakeTradingClient(
        submit_response=make_order(status="new"),
        poll_sequence=[
            make_order(status="pending_new"),
            make_order(status="accepted"),
            make_order(status="filled", filled_qty="5", filled_avg_price="99.5"),
        ],
    )
    fill = submit_equity_order(
        client, "MSFT", 5, "buy", poll_interval_seconds=0, max_poll_attempts=5
    )
    assert fill.status == "filled"
    assert fill.filled_qty == 5.0
    assert client.get_order_calls == 3


def test_submit_equity_order_already_filled_on_submit_needs_no_poll():
    """Sometimes the fill IS already reflected in submit_order()'s response -
    must not poll unnecessarily in that case."""
    client = FakeTradingClient(
        submit_response=make_order(status="filled", filled_qty="1", filled_avg_price="500.0"),
        poll_sequence=[],
    )
    fill = submit_equity_order(
        client, "SPY", 1, "buy", poll_interval_seconds=0, max_poll_attempts=5
    )
    assert fill.status == "filled"
    assert client.get_order_calls == 0
    assert client.cancel_calls == []


def test_submit_equity_order_times_out_and_cancels_the_still_open_broker_order():
    """The critical safety property: if we give up waiting for a fill, the
    order must not be left dangling open on the broker - otherwise it can
    fill later (e.g. at the next market open) with zero record of it in the
    local ledger. Reproduces a real incident: 19 such DAY orders from
    earlier pipeline runs (before this fix) were found still 'accepted' a
    day later, primed to silently fill."""
    client = FakeTradingClient(
        submit_response=make_order(order_id="o1", status="new"),
        poll_sequence=[
            make_order(order_id="o1", status="accepted"),
            make_order(order_id="o1", status="accepted"),
            make_order(order_id="o1", status="canceled"),  # Refetch nach dem Cancel-Aufruf
        ],
    )
    fill = submit_equity_order(
        client, "NVDA", 100, "buy", poll_interval_seconds=0, max_poll_attempts=2
    )
    assert fill.status == "pending"
    assert fill.filled_qty == 0.0
    assert client.cancel_calls == ["o1"]


def test_submit_equity_order_stops_polling_immediately_on_rejection():
    """A rejected order can never transition to filled - polling must stop
    at the first terminal-unfilled status instead of burning through the
    full timeout budget, and no cancel is needed for an already-dead order."""
    client = FakeTradingClient(
        submit_response=make_order(status="new"),
        poll_sequence=[make_order(status="rejected")],
    )
    fill = submit_equity_order(
        client, "TSLA", 1, "buy", poll_interval_seconds=0, max_poll_attempts=10
    )
    assert fill.status == "pending"
    assert fill.filled_qty == 0.0
    assert client.get_order_calls == 1  # nicht 10 - Polling endet sofort bei 'rejected'
    assert client.cancel_calls == []  # bereits tot, kein Cancel noetig


def test_submit_equity_order_cancel_failure_does_not_crash():
    """If even the cleanup cancel call fails (e.g. transient broker error),
    the already-successful order submission must still return normally -
    not raise - since the pipeline should never crash over a best-effort
    cleanup step."""

    class FailingCancelClient(FakeTradingClient):
        def cancel_order_by_id(self, order_id):
            self.cancel_calls.append(order_id)
            raise RuntimeError("broker unavailable")

    client = FailingCancelClient(
        submit_response=make_order(order_id="o1", status="new"),
        poll_sequence=[make_order(order_id="o1", status="accepted")],
    )
    fill = submit_equity_order(
        client, "NVDA", 100, "buy", poll_interval_seconds=0, max_poll_attempts=1
    )
    assert fill.status == "pending"
    assert fill.filled_qty == 0.0
    assert client.cancel_calls == ["o1"]
