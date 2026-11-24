"""Customer support tools backed by the in-memory mock store."""

from __future__ import annotations

from datetime import UTC, date, datetime

from .models import (
    CancellationResult,
    EscalationRecord,
    EscalationRequest,
    EscalationResult,
    OrderDetailsResult,
    OrderStatusResult,
    RefundEligibilityResult,
)
from .store import OrderStore, get_default_store

RETURN_WINDOW_DAYS = 30
HUMAN_SUPPORT_NOTE_KEYWORDS = ("missing", "tampered")


def _normalize_order_id(order_id: str | int | None) -> str | None:
    if order_id is None:
        return None
    normalized = str(order_id).strip().lstrip("#").strip()
    return normalized or None


def _not_found_status(order_id: str | None) -> OrderStatusResult:
    return OrderStatusResult(
        success=False,
        order_id=order_id,
        message=f"Order '{order_id}' was not found.",
    )


def get_order_status(
    order_id: str,
    *,
    store: OrderStore | None = None,
) -> OrderStatusResult:
    """Return the current status of an order."""
    store = store or get_default_store()
    normalized_id = _normalize_order_id(order_id)
    if normalized_id is None:
        return OrderStatusResult(
            success=False,
            message="A non-empty order ID is required.",
        )

    order = store.get_order(normalized_id)
    if order is None:
        return _not_found_status(normalized_id)

    return OrderStatusResult(
        success=True,
        order_id=order.order_id,
        status=order.status,
        message=f"Order '{order.order_id}' is currently '{order.status}'.",
    )


def get_order_details(
    order_id: str,
    *,
    store: OrderStore | None = None,
) -> OrderDetailsResult:
    """Return the items, dates, and total for an order."""
    store = store or get_default_store()
    normalized_id = _normalize_order_id(order_id)
    if normalized_id is None:
        return OrderDetailsResult(
            success=False,
            message="A non-empty order ID is required.",
        )

    order = store.get_order(normalized_id)
    if order is None:
        return OrderDetailsResult(
            success=False,
            order_id=normalized_id,
            message=f"Order '{normalized_id}' was not found.",
        )

    return OrderDetailsResult(
        success=True,
        order_id=order.order_id,
        items=order.items,
        order_date=order.order_date,
        delivery_date=order.delivery_date,
        total=order.total,
        currency=store.currency,
        message=f"Found {len(order.items)} item(s) for order '{order.order_id}'.",
    )


def _human_support_reason(order_notes: str | None, item_condition: str) -> str | None:
    condition = item_condition.lower()
    if condition == "damaged":
        return (
            "The item is reported as damaged. Damaged-item claims must be "
            "reviewed by human support and cannot be approved as an ordinary return."
        )

    notes = (order_notes or "").lower()
    if "missing" in notes:
        return (
            "The order notes report a missing item. Missing-item claims must be "
            "reviewed by human support and cannot be approved as an ordinary return."
        )
    if "tampered" in notes:
        return (
            "The order notes report possible tampering. Tampering claims must be "
            "reviewed by human support and cannot be approved as an ordinary return."
        )

    if condition == "unknown":
        return (
            "The item's condition could not be verified. A human support agent "
            "must review this return request."
        )

    return None


def check_refund_eligibility(
    order_id: str,
    *,
    reference_date: date | None = None,
    store: OrderStore | None = None,
) -> RefundEligibilityResult:
    """Check whether an order qualifies for the standard return process.

    This tool only reports eligibility. It never processes or promises a
    refund. ``reference_date`` defaults to today and can be supplied for
    deterministic checks.
    """
    store = store or get_default_store()
    normalized_id = _normalize_order_id(order_id)
    if normalized_id is None:
        return RefundEligibilityResult(
            success=False,
            message="A non-empty order ID is required.",
            reason="A non-empty order ID is required.",
            return_window_days=RETURN_WINDOW_DAYS,
        )

    order = store.get_order(normalized_id)
    if order is None:
        return RefundEligibilityResult(
            success=False,
            order_id=normalized_id,
            message=f"Order '{normalized_id}' was not found.",
            reason=f"Order '{normalized_id}' was not found.",
            return_window_days=RETURN_WINDOW_DAYS,
        )

    today = reference_date or date.today()
    base_result = {
        "success": True,
        "order_id": order.order_id,
        "order_status": order.status,
        "delivery_date": order.delivery_date,
        "return_window_days": RETURN_WINDOW_DAYS,
        "item_condition": order.item_condition,
        "original_packaging": order.original_packaging,
        "total": order.total,
        "currency": store.currency,
    }

    if order.status == "cancelled":
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            reason="The order was cancelled, so there is no delivery to return.",
            message=f"Order '{order.order_id}' is not eligible for a return.",
        )

    if order.status != "delivered" or order.delivery_date is None:
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            reason="The order has not been delivered yet, so it is not eligible for a return.",
            message=f"Order '{order.order_id}' is not eligible for a return.",
        )

    days_since_delivery = (today - order.delivery_date).days

    if days_since_delivery < 0:
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            days_since_delivery=days_since_delivery,
            reason="The delivery date is in the future relative to the check date, "
            "so return eligibility cannot be confirmed.",
            message=f"Order '{order.order_id}' is not eligible for a return.",
        )

    human_support_reason = _human_support_reason(order.notes, order.item_condition)
    if human_support_reason is not None:
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            requires_human_support=True,
            days_since_delivery=days_since_delivery,
            reason=human_support_reason,
            message=(
                f"Order '{order.order_id}' cannot be handled through the standard "
                f"return process; human support is required. {human_support_reason}"
            ),
        )

    if days_since_delivery > RETURN_WINDOW_DAYS:
        reason = (
            f"Delivery was {days_since_delivery} days ago, which is outside the "
            f"{RETURN_WINDOW_DAYS}-day return window."
        )
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            days_since_delivery=days_since_delivery,
            reason=reason,
            message=f"Order '{order.order_id}' is not eligible for a return: {reason}",
        )

    if order.item_condition.lower() != "unused":
        reason = (
            f"Item condition is '{order.item_condition}'. The item must be unused "
            "to qualify for a return."
        )
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            days_since_delivery=days_since_delivery,
            reason=reason,
            message=f"Order '{order.order_id}' is not eligible for a return: {reason}",
        )

    if not order.original_packaging:
        reason = "The item is not in its original packaging."
        return RefundEligibilityResult(
            **base_result,
            eligible=False,
            days_since_delivery=days_since_delivery,
            reason=reason,
            message=f"Order '{order.order_id}' is not eligible for a return: {reason}",
        )

    reason = (
        f"The order was delivered {days_since_delivery} day(s) ago, the item is "
        "unused, and it is in its original packaging."
    )
    return RefundEligibilityResult(
        **base_result,
        eligible=True,
        days_since_delivery=days_since_delivery,
        reason=reason,
        message=(
            f"Order '{order.order_id}' is eligible for a return. This is an "
            "eligibility check only; no refund has been processed."
        ),
    )


def cancel_order(
    order_id: str,
    *,
    store: OrderStore | None = None,
) -> CancellationResult:
    """Cancel a processing order and update its in-memory status."""
    store = store or get_default_store()
    normalized_id = _normalize_order_id(order_id)
    if normalized_id is None:
        return CancellationResult(
            success=False,
            message="A non-empty order ID is required.",
        )

    order = store.get_order(normalized_id)
    if order is None:
        return CancellationResult(
            success=False,
            order_id=normalized_id,
            message=f"Order '{normalized_id}' was not found.",
        )

    if order.status != "processing":
        return CancellationResult(
            success=False,
            order_id=order.order_id,
            previous_status=order.status,
            status=order.status,
            message=(
                f"Order '{order.order_id}' cannot be cancelled because its status "
                f"is '{order.status}'. Only processing orders can be cancelled."
            ),
        )

    try:
        updated_order = store.set_order_status(order.order_id, "cancelled")
    except (KeyError, ValueError) as exc:
        return CancellationResult(
            success=False,
            order_id=order.order_id,
            previous_status=order.status,
            status=order.status,
            message=f"Order '{order.order_id}' could not be cancelled: {exc}",
        )

    stored_order = store.get_order(order.order_id)
    if updated_order.status != "cancelled" or stored_order is None:
        return CancellationResult(
            success=False,
            order_id=order.order_id,
            previous_status=order.status,
            status=order.status,
            message=(
                f"Order '{order.order_id}' could not be cancelled because the "
                "status update did not complete."
            ),
        )

    return CancellationResult(
        success=True,
        order_id=order.order_id,
        previous_status=order.status,
        status=stored_order.status,
        cancelled=True,
        message=f"Order '{order.order_id}' was successfully cancelled.",
    )


class EscalationRegistry:
    """In-memory record of escalation requests."""

    def __init__(self) -> None:
        self._counter = 0
        self._records: list[EscalationRecord] = []

    def create(
        self,
        request: EscalationRequest,
        *,
        created_at: datetime | None = None,
    ) -> EscalationRecord:
        self._counter += 1
        record = EscalationRecord(
            escalation_id=f"ESC-{self._counter:06d}",
            order_id=request.order_id,
            reason=request.reason,
            created_at=created_at or datetime.now(UTC),
        )
        self._records.append(record)
        return record

    def list_escalations(self) -> list[EscalationRecord]:
        return list(self._records)


_default_escalation_registry = EscalationRegistry()


def escalate_to_human(
    order_id: str | None,
    reason: str,
    *,
    registry: EscalationRegistry | None = None,
) -> EscalationResult:
    """Record an escalation request, with or without an order ID.

    The order is deliberately not validated against the store so that a
    customer reporting an unverifiable order can still be escalated.
    """
    normalized_id = _normalize_order_id(order_id)
    normalized_reason = reason.strip() if isinstance(reason, str) else ""

    if not normalized_reason:
        return EscalationResult(
            success=False,
            order_id=normalized_id,
            message="An escalation reason is required.",
        )

    request = EscalationRequest(order_id=normalized_id, reason=normalized_reason)
    registry = registry or _default_escalation_registry
    record = registry.create(request)

    return EscalationResult(
        success=True,
        escalation_id=record.escalation_id,
        order_id=record.order_id,
        reason=record.reason,
        message=(
            f"Escalation request {record.escalation_id} was created successfully."
        ),
    )
