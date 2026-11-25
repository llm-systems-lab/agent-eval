"""Test configuration.

Tests must stay offline and deterministic. ``uv run`` loads the project
``.env`` automatically, which can enable LangSmith tracing and real
credentials; tracing is forced off here so no runs are exported during tests.
"""

from __future__ import annotations

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
