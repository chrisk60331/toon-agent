from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from time import perf_counter
from typing import Callable, Iterable, Sequence

from pydantic import BaseModel, Field

from agent_run_summary import AgentRunSummary
from main import run_default_agent
from main_toon import run_toon_agent
from main_toon_agent import run_demo


class BenchmarkResult(BaseModel):
    """Timing and usage summary for a single agent entrypoint."""

    name: str
    duration_seconds: float
    summary: AgentRunSummary = Field(..., description="Usage metrics emitted by the run.")


def _run_benchmark(name: str, runner: Callable[[], AgentRunSummary]) -> BenchmarkResult:
    start = perf_counter()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        summary = runner()
    if summary is None:
        raise RuntimeError(f"Runner '{name}' did not produce a summary.")

    elapsed = perf_counter() - start
    return BenchmarkResult(name=name, duration_seconds=elapsed, summary=summary)


def run_benchmarks() -> list[BenchmarkResult]:
    """Execute all entrypoints and collect timing plus token usage."""
    runners: Sequence[tuple[str, Callable[[], AgentRunSummary]]] = [
        ("main", run_default_agent),
        ("main_toon", run_toon_agent),
        ("main_toon_agent", run_demo),
    ]
    return [_run_benchmark(name, runner) for name, runner in runners]


def display_results(results: Iterable[BenchmarkResult]) -> None:
    """Print a concise view of benchmark outcomes."""
    for result in results:
        total_tokens = result.summary.total_tokens
        usage = result.summary.usage or {}
        print(
            f"{result.name}: {result.duration_seconds:.3f}s | "
            f"total_tokens={total_tokens} | usage={usage}\n\n"
        )


if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    display_results(benchmark_results)

