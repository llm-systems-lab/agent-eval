from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from agent.tools import (
    DEFAULT_DATA_PATH,
    EscalationRegistry,
    OrderStore,
    cancel_order,
    check_refund_eligibility,
    escalate_to_human,
    get_order_details,
    get_order_status,
)

RAW_STORE = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
REFERENCE_DATE = date(2026, 11, 24)


def make_store(overrides: dict | None = None) -> OrderStore:
    data = copy.deepcopy(RAW_STORE)
    if overrides:
        for order in data["orders"]:
            if order["order_id"] in overrides:
                order.update(overrides[order["order_id"]])
    return OrderStore(data=data)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    path = tmp_path / "mock_store.json"
    path.write_text(DEFAULT_DATA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.fixture
def store(store_path: Path) -> OrderStore:
    return OrderStore(data_path=store_path)


@pytest.fixture
def registry() -> EscalationRegistry:
    return EscalationRegistry()


class TestGetOrderStatus:
    @pytest.mark.parametrize(
        ("order_id", "expected_status"),
        [
            ("1042", "processing"),
            ("1043", "shipped"),
            ("1046", "delivered"),
        ],
    )
    def test_returns_each_existing_status(
        self, store: OrderStore, order_id: str, expected_status: str
    ) -> None:
        result = get_order_status(order_id, store=store)

        assert result.success is True
        assert result.order_id == order_id
        assert result.status == expected_status

    def test_normalizes_order_id_with_hash_and_spaces(self, store: OrderStore) -> None:
        result = get_order_status("  #1055  ", store=store)

        assert result.success is True
        assert result.order_id == "1055"
        assert result.status == "processing"

    def test_unknown_order_returns_not_found(self, store: OrderStore) -> None:
        result = get_order_status("9999", store=store)

        assert result.success is False
        assert result.order_id == "9999"
        assert result.status is None
        assert "not found" in result.message.lower()

    @pytest.mark.parametrize("order_id", ["", "   ", "#"])
    def test_empty_order_id_is_rejected(self, store: OrderStore, order_id: str) -> None:
        result = get_order_status(order_id, store=store)

        assert result.success is False
        assert "order id" in result.message.lower()

    def test_cancelled_status_is_reported_after_cancellation(
        self, store: OrderStore
    ) -> None:
        cancel_order("1055", store=store)

        result = get_order_status("1055", store=store)

        assert result.success is True
        assert result.status == "cancelled"


class TestGetOrderDetails:
    def test_returns_items_dates_and_total(self, store: OrderStore) -> None:
        result = get_order_details("1046", store=store)

        assert result.success is True
        assert result.order_id == "1046"
        assert result.order_date == date(2026, 11, 10)
        assert result.delivery_date == date(2026, 11, 13)
        assert result.total == 34.98
        assert result.currency == "USD"
        assert [(item.product, item.quantity) for item in result.items] == [
            ("Wireless Mouse", 1),
            ("Mouse Pad", 1),
        ]
        assert [item.unit_price for item in result.items] == [24.99, 9.99]

    def test_returns_multiple_quantities(self, store: OrderStore) -> None:
        result = get_order_details("1047", store=store)

        assert result.success is True
        assert result.items[0].product == "Wireless Mouse"
        assert result.items[0].quantity == 2

    def test_unknown_order_returns_not_found(self, store: OrderStore) -> None:
        result = get_order_details("does-not-exist", store=store)

        assert result.success is False
        assert result.items == []
        assert result.total is None
        assert "not found" in result.message.lower()


class TestRefundEligibility:
    def test_within_window_unused_and_packaged_is_eligible(
        self, store: OrderStore
    ) -> None:
        result = check_refund_eligibility(
            "1050", reference_date=REFERENCE_DATE, store=store
        )

        assert result.success is True
        assert result.eligible is True
        assert result.requires_human_support is False
        assert result.days_since_delivery == 10
        assert result.return_window_days == 30
        assert result.original_packaging is True

    def test_thirtieth_day_is_still_eligible(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1046", reference_date=date(2026, 12, 13), store=store
        )

        assert result.eligible is True
        assert result.days_since_delivery == 30

    def test_thirty_first_day_is_outside_window(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1046", reference_date=date(2026, 12, 14), store=store
        )

        assert result.eligible is False
        assert result.days_since_delivery == 31
        assert "outside the 30-day return window" in result.reason

    def test_outside_window_is_not_eligible(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1051", reference_date=REFERENCE_DATE, store=store
        )

        assert result.success is True
        assert result.eligible is False
        assert result.days_since_delivery == 45
        assert "outside the 30-day return window" in result.reason

    def test_used_item_is_not_eligible(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1052", reference_date=REFERENCE_DATE, store=store
        )

        assert result.eligible is False
        assert result.requires_human_support is False
        assert result.item_condition == "used"
        assert "must be unused" in result.reason

    def test_missing_original_packaging_is_not_eligible(self) -> None:
        store = make_store({"1046": {"original_packaging": False}})

        result = check_refund_eligibility(
            "1046", reference_date=REFERENCE_DATE, store=store
        )

        assert result.eligible is False
        assert result.requires_human_support is False
        assert "original packaging" in result.reason

    def test_damaged_item_requires_human_support(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1053", reference_date=REFERENCE_DATE, store=store
        )

        assert result.success is True
        assert result.eligible is False
        assert result.requires_human_support is True
        assert "damaged" in result.reason.lower()
        assert "human support" in result.reason.lower()

    def test_missing_item_requires_human_support(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1054", reference_date=REFERENCE_DATE, store=store
        )

        assert result.eligible is False
        assert result.requires_human_support is True
        assert "missing item" in result.reason.lower()

    def test_undelivered_order_is_not_eligible(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1042", reference_date=REFERENCE_DATE, store=store
        )

        assert result.eligible is False
        assert result.days_since_delivery is None
        assert "not been delivered" in result.reason

    def test_cancelled_order_is_not_eligible(self, store: OrderStore) -> None:
        cancel_order("1055", store=store)

        result = check_refund_eligibility(
            "1055", reference_date=REFERENCE_DATE, store=store
        )

        assert result.eligible is False
        assert result.order_status == "cancelled"
        assert "cancelled" in result.reason.lower()

    def test_unknown_order_returns_failure(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "9999", reference_date=REFERENCE_DATE, store=store
        )

        assert result.success is False
        assert result.eligible is False
        assert "not found" in result.message.lower()

    def test_returns_order_information(self, store: OrderStore) -> None:
        result = check_refund_eligibility(
            "1050", reference_date=REFERENCE_DATE, store=store
        )

        assert result.order_status == "delivered"
        assert result.delivery_date == date(2026, 11, 14)
        assert result.item_condition == "unused"
        assert result.total == 79.99
        assert result.currency == "USD"


class TestCancelOrder:
    def test_cancels_processing_order(self, store: OrderStore) -> None:
        result = cancel_order("1055", store=store)

        assert result.success is True
        assert result.cancelled is True
        assert result.previous_status == "processing"
        assert result.status == "cancelled"
        assert store.get_order("1055").status == "cancelled"

    def test_rejects_shipped_order(self, store: OrderStore) -> None:
        result = cancel_order("1056", store=store)

        assert result.success is False
        assert result.cancelled is False
        assert result.status == "shipped"
        assert store.get_order("1056").status == "shipped"

    def test_rejects_delivered_order(self, store: OrderStore) -> None:
        result = cancel_order("1057", store=store)

        assert result.success is False
        assert result.cancelled is False
        assert result.status == "delivered"
        assert store.get_order("1057").status == "delivered"

    def test_rejects_second_cancellation(self, store: OrderStore) -> None:
        first = cancel_order("1055", store=store)
        second = cancel_order("1055", store=store)

        assert first.success is True
        assert second.success is False
        assert second.status == "cancelled"
        assert "cannot be cancelled" in second.message

    def test_unknown_order_returns_failure(self, store: OrderStore) -> None:
        result = cancel_order("9999", store=store)

        assert result.success is False
        assert result.cancelled is False
        assert "not found" in result.message.lower()

    def test_cancellation_does_not_write_to_store_file(
        self, store: OrderStore, store_path: Path
    ) -> None:
        before = hashlib.sha256(store_path.read_bytes()).hexdigest()

        cancel_order("1055", store=store)

        after = hashlib.sha256(store_path.read_bytes()).hexdigest()
        assert before == after

    def test_cancellation_is_isolated_per_store_instance(self) -> None:
        first_store = OrderStore(data=copy.deepcopy(RAW_STORE))
        second_store = OrderStore(data=copy.deepcopy(RAW_STORE))

        cancel_order("1055", store=first_store)

        assert first_store.get_order("1055").status == "cancelled"
        assert second_store.get_order("1055").status == "processing"


class TestEscalateToHuman:
    def test_escalation_with_order_id(
        self, registry: EscalationRegistry
    ) -> None:
        result = escalate_to_human(
            "1059",
            "Customer did not receive a package marked as delivered.",
            registry=registry,
        )

        assert result.success is True
        assert result.order_id == "1059"
        assert result.escalation_id is not None
        assert result.escalation_id.startswith("ESC-")
        assert len(registry.list_escalations()) == 1
        assert registry.list_escalations()[0].reason.startswith("Customer did not")

    def test_escalation_without_order_id(
        self, registry: EscalationRegistry
    ) -> None:
        result = escalate_to_human(
            None,
            "Customer is upset and wants to speak to a person.",
            registry=registry,
        )

        assert result.success is True
        assert result.order_id is None
        assert result.escalation_id is not None
        assert len(registry.list_escalations()) == 1

    def test_escalation_for_unverified_order_is_allowed(
        self, registry: EscalationRegistry
    ) -> None:
        result = escalate_to_human(
            "9999",
            "Customer cannot verify the order.",
            registry=registry,
        )

        assert result.success is True
        assert result.order_id == "9999"

    def test_escalation_normalizes_order_id(
        self, registry: EscalationRegistry
    ) -> None:
        result = escalate_to_human(
            " #1063 ", "Customer requested a human.", registry=registry
        )

        assert result.success is True
        assert result.order_id == "1063"

    def test_escalation_ids_are_unique(self, registry: EscalationRegistry) -> None:
        first = escalate_to_human("1042", "First reason.", registry=registry)
        second = escalate_to_human("1042", "Second reason.", registry=registry)

        assert first.escalation_id != second.escalation_id
        assert len(registry.list_escalations()) == 2

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_empty_reason_is_rejected(
        self, registry: EscalationRegistry, reason: str
    ) -> None:
        result = escalate_to_human("1042", reason, registry=registry)

        assert result.success is False
        assert result.escalation_id is None
        assert registry.list_escalations() == []