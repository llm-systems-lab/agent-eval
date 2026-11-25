"""Rich-based terminal interface for the Baseline Agent.

This module only handles presentation. Agent orchestration, tracing, and result
construction stay in ``baseline_agent`` so the two concerns remain independent.
The core agent never prints; everything user-facing is rendered here.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "agent"

import argparse
from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .baseline_agent import BaselineAgent, create_baseline_agent
from .traces import AgentRun

APP_TITLE = "AgentEval — Customer Support Agent"
EXIT_COMMANDS = {"exit", "quit", ":q"}
THINKING = "[bold]Agent is thinking...[/bold]"


def _format_arguments(arguments: dict[str, object]) -> str:
    if not arguments:
        return ""
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


def render_run(console: Console, run: AgentRun) -> None:
    """Render a completed run: an ordered tool-call table and the answer panel."""
    if run.tool_calls:
        table = Table(title=f"Tool calls ({len(run.tool_calls)})", show_lines=True)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Arguments")
        table.add_column("Result")
        for index, record in enumerate(run.tool_calls, start=1):
            if record.error is not None:
                outcome = f"[red]{record.error}[/red]"
            else:
                outcome = record.result or ""
            table.add_row(
                str(index),
                record.tool_name,
                _format_arguments(record.arguments),
                outcome,
            )
        console.print(table)
    else:
        console.print("[dim]No tool calls were made.[/dim]")

    console.print(
        Panel(
            run.answer or "(no response)",
            title="Response",
            border_style="green",
        )
    )


def run_once(console: Console, agent: BaselineAgent, message: str) -> None:
    """Stream one agent run, showing a spinner and tool activity, then render it."""
    run: AgentRun | None = None
    try:
        with console.status(THINKING, spinner="dots") as status:
            for event in agent.stream(message):
                if event.kind == "tool_call" and event.tool_call is not None:
                    status.update(
                        f"{THINKING} calling [cyan]{event.tool_call.tool_name}[/cyan]"
                        f"({_format_arguments(event.tool_call.arguments)})"
                    )
                elif event.kind == "tool_result":
                    status.update(f"{THINKING} processing tool result")
                elif event.kind == "final":
                    run = event.run
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        return
    except Exception as exc:  # noqa: BLE001 - surface any agent/API error to the user
        console.print(Panel(str(exc), title="Agent error", border_style="red"))
        return

    if run is not None:
        render_run(console, run)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "message",
        nargs="?",
        help="Customer message. Runs once and exits; omitted starts an interactive session.",
    )
    parser.add_argument(
        "--reference-date",
        help="Fix the refund reference date as YYYY-MM-DD for reproducible runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()
    console.print(Panel.fit(APP_TITLE, border_style="blue"))

    reference_date = None
    if args.reference_date:
        reference_date = date.fromisoformat(args.reference_date)

    try:
        agent = create_baseline_agent(reference_date=reference_date)
    except Exception as exc:  # noqa: BLE001 - configuration errors are user-facing
        console.print(Panel(str(exc), title="Configuration error", border_style="red"))
        return 1

    if args.message:
        run_once(console, agent, args.message)
        return 0

    console.print("[dim]Type a message, or 'exit' / Ctrl-D to quit.[/dim]")
    while True:
        try:
            message = console.input("[bold]Customer:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not message:
            continue
        if message.lower() in EXIT_COMMANDS:
            console.print("[dim]Goodbye.[/dim]")
            break
        run_once(console, agent, message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
