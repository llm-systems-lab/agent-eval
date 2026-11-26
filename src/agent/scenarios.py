"""Loader for the fixed evaluation dataset.

The dataset is a read-only fixture. This module only reads and validates it; it
never writes to it and never hardcodes scenario content.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset.json"


class Scenario(BaseModel):
    """One evaluation scenario from ``evaluation_dataset.json``."""

    id: str
    category: str
    customer_message: str
    order_id: str | None = None
    expected_tool_calls: list[str] = Field(default_factory=list)
    should_escalate: bool = False
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)


class ScenarioDataset(BaseModel):
    scenarios: list[Scenario] = Field(default_factory=list)


def load_scenarios(data_path: str | Path | None = None) -> list[Scenario]:
    """Load and validate the evaluation scenarios."""
    path = Path(data_path) if data_path is not None else DEFAULT_DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    return ScenarioDataset.model_validate(raw).scenarios
