from __future__ import annotations

from collections.abc import Iterator
from io import StringIO

from rich.console import Console

from agent.cli import render_run, run_once
from agent.traces import AgentEvent, AgentRun, ToolCallRecord


class FailingAgent:
    def stream(self, message: str) -> Iterator[AgentEvent]:
        raise RuntimeError("simulated transport failure")


def make_console() -> Console:
    return Console(file=StringIO(), force_terminal=False, width=100)


def test_render_run_shows_tool_calls_and_answer() -> None:
    run = AgentRun(
        run_id="run-1",
        user_message="Status of 1042?",
        model="fake",
        answer="Order 1042 is processing.",
        tool_calls=[
            ToolCallRecord(
                tool_name="get_order_status",
                arguments={"order_id": "1042"},
                result='{"success": true}',
            )
        ],
    )
    console = make_console()

    render_run(console, run)

    output = console.file.getvalue()
    assert "get_order_status" in output
    assert "order_id='1042'" in output
    assert "Order 1042 is processing." in output


def test_run_once_handles_agent_error_gracefully() -> None:
    console = make_console()

    run_once(console, FailingAgent(), "hello")  # type: ignore[arg-type]

    output = console.file.getvalue()
    assert "Agent error" in output
    assert "simulated transport failure" in output
