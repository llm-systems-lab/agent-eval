"""Pydantic models for the customer support tool layer."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]


class OrderItem(BaseModel):
    product: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(ge=0)


class Order(BaseModel):
    order_id: str
    status: OrderStatus
    items: list[OrderItem] = Field(default_factory=list)
    order_date: date
    delivery_date: date | None = None
    total: float = Field(ge=0)
    item_condition: str
    original_packaging: bool
    notes: str | None = None


class StoreData(BaseModel):
    store_name: str
    currency: str
    schema_version: str
    orders: list[Order] = Field(default_factory=list)

    @model_validator(mode="after")
    def _order_ids_are_unique(self) -> StoreData:
        order_ids = [order.order_id for order in self.orders]
        if len(order_ids) != len(set(order_ids)):
            raise ValueError("mock store contains duplicate order IDs")
        return self


class OrderStatusResult(BaseModel):
    success: bool
    message: str
    order_id: str | None = None
    status: OrderStatus | None = None


class OrderDetailsResult(BaseModel):
    success: bool
    message: str
    order_id: str | None = None
    items: list[OrderItem] = Field(default_factory=list)
    order_date: date | None = None
    delivery_date: date | None = None
    total: float | None = None
    currency: str | None = None


class RefundEligibilityResult(BaseModel):
    success: bool
    message: str
    order_id: str | None = None
    eligible: bool = False
    reason: str = ""
    requires_human_support: bool = False
    order_status: OrderStatus | None = None
    delivery_date: date | None = None
    days_since_delivery: int | None = None
    return_window_days: int = 30
    item_condition: str | None = None
    original_packaging: bool | None = None
    total: float | None = None
    currency: str | None = None


class CancellationResult(BaseModel):
    success: bool
    message: str
    order_id: str | None = None
    previous_status: OrderStatus | None = None
    status: OrderStatus | None = None
    cancelled: bool = False


class EscalationRequest(BaseModel):
    order_id: str | None = None
    reason: str = Field(min_length=1)


class EscalationRecord(BaseModel):
    escalation_id: str
    order_id: str | None = None
    reason: str
    created_at: datetime


class EscalationResult(BaseModel):
    success: bool
    message: str
    escalation_id: str | None = None
    order_id: str | None = None
    reason: str = ""
