"""In-memory data access for the mock customer support store."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import Order, OrderStatus, StoreData

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "mock_store.json"


def load_store_data(data_path: str | Path) -> dict[str, Any]:
    """Load raw mock store data from a JSON file."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mock store data file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class OrderStore:
    """Owns an isolated, in-memory copy of the mock store data.

    Changes made to orders through this class never touch the JSON file on
    disk. Pass ``data`` for tests or ``data_path`` to load a temporary copy.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        if data_path is not None and data is not None:
            raise ValueError("Provide either data_path or data, not both")

        if data is None:
            raw_data = load_store_data(
                DEFAULT_DATA_PATH if data_path is None else data_path
            )
        else:
            raw_data = dict(data)

        self._store = StoreData.model_validate(raw_data)
        self._orders: dict[str, Order] = {
            order.order_id: order for order in self._store.orders
        }

    @property
    def store_name(self) -> str:
        return self._store.store_name

    @property
    def currency(self) -> str:
        return self._store.currency

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def list_orders(self) -> list[Order]:
        return list(self._orders.values())

    def set_order_status(self, order_id: str, status: OrderStatus) -> Order:
        """Update an order status in memory and return the updated copy."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Unknown order ID: {order_id}")

        updated_order = order.model_copy(update={"status": status})
        self._orders[order_id] = updated_order
        return updated_order


_default_store: OrderStore | None = None


def get_default_store() -> OrderStore:
    """Return the process-wide store loaded from ``data/mock_store.json``."""
    global _default_store
    if _default_store is None:
        _default_store = OrderStore()
    return _default_store
