"""LangChain tool wrappers around the deterministic customer support tools.

The wrappers adapt signatures, normalize order IDs to digits only (reusing the
tools' own normalization), and serialize the Pydantic results to JSON so an LLM
can consume them. They must not duplicate or change any business rule: every
call is delegated to ``agent.tools.customer_support_tools``.
"""

from __future__ import annotations

from datetime import date

from langchain_core.tools import BaseTool, tool

from .tools import customer_support_tools as cst
from .tools.customer_support_tools import EscalationRegistry, _normalize_order_id
from .tools.store import OrderStore


def build_customer_support_tools(
    *,
    store: OrderStore | None = None,
    registry: EscalationRegistry | None = None,
    reference_date: date | None = None,
) -> list[BaseTool]:
    """Wrap the five tools, binding an optional store, registry, and date.

    ``store``, ``registry``, and ``reference_date`` are captured by closure and
    injected into the existing tool functions through their keyword-only
    dependency-injection parameters. When ``store`` or ``registry`` is ``None``
    the tools fall back to their process-wide defaults.
    """

    @tool
    def get_order_status(order_id: str) -> str:
        """Return the current status of an order: processing, shipped, delivered, or cancelled. This returns only the status and does not include a delivery date."""
        return cst.get_order_status(
            _normalize_order_id(order_id) or "", store=store
        ).model_dump_json()

    @tool
    def get_order_details(order_id: str) -> str:
        """Return an order's items, quantities, unit prices, order date, delivery date, total, and currency."""
        return cst.get_order_details(
            _normalize_order_id(order_id) or "", store=store
        ).model_dump_json()

    @tool
    def check_refund_eligibility(order_id: str) -> str:
        """Check whether an order is eligible for a refund. Reports eligibility only."""
        return cst.check_refund_eligibility(
            _normalize_order_id(order_id) or "",
            reference_date=reference_date,
            store=store,
        ).model_dump_json()

    @tool
    def cancel_order(order_id: str) -> str:
        """Cancel an order. Only orders with status 'processing' can be cancelled; call get_order_status first to confirm. Do not attempt to cancel shipped, delivered, or already-cancelled orders."""
        return cst.cancel_order(
            _normalize_order_id(order_id) or "", store=store
        ).model_dump_json()

    @tool
    def escalate_to_human(reason: str, order_id: str | None = None) -> str:
        """Escalate an issue to a human support agent.

        Call this (do not merely offer it or ask for permission) when any of these apply:
        the order cannot be found after checking the customer's order ID; the customer reports a
        delivery delay with explicit urgency or an unmet deadline; an item is damaged, missing,
        tampered with, or cannot be verified; the customer asks for a human representative; or the
        complaint is repeatedly unresolved. Provide a short reason; the order ID is optional.
        Do not escalate when the customer simply has not provided an order ID; ask for it first.
        """
        return cst.escalate_to_human(
            _normalize_order_id(order_id), reason, registry=registry
        ).model_dump_json()

    return [
        get_order_status,
        get_order_details,
        check_refund_eligibility,
        cancel_order,
        escalate_to_human,
    ]
