from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from benchmark import BenchmarkScoreResult, benchmark_with_score
import main as default_main
import main_toon as toon_main
import src.constants as constants
import src.main_toon_agent as main_toon_agent_module


class AgentAggregate(BaseModel):
    """Aggregate F1 scores and token usage across the configured test suite."""

    name: str = Field(..., description="Agent entrypoint identifier.")
    runs: int = Field(..., ge=1, description="Number of benchmarked tests.")
    average_f1: float = Field(..., ge=0.0, le=1.0)
    average_duration_seconds: float = Field(..., ge=0.0)
    total_duration_seconds: float = Field(..., ge=0.0)
    runs_with_token_usage: int = Field(..., ge=0)
    average_total_tokens: float | None = Field(default=None, ge=0.0)
    total_tokens: int | None = Field(default=None, ge=0)

    @classmethod
    def from_results(cls, name: str, results: Sequence[BenchmarkScoreResult]) -> "AgentAggregate":
        if not results:
            raise ValueError(f"No benchmark results provided for agent '{name}'.")

        runs = len(results)
        total_duration = sum(item.duration_seconds for item in results)
        total_f1 = sum(item.f1_score for item in results)
        token_values = [
            item.summary.total_tokens for item in results if item.summary.total_tokens is not None
        ]

        total_tokens = sum(token_values) if token_values else None
        average_tokens = (total_tokens / len(token_values)) if token_values else None

        return cls(
            name=name,
            runs=runs,
            average_f1=total_f1 / runs,
            average_duration_seconds=total_duration / runs,
            total_duration_seconds=total_duration,
            runs_with_token_usage=len(token_values),
            average_total_tokens=average_tokens,
            total_tokens=total_tokens,
        )


def _apply_task(task_instruction: str) -> None:
    """Propagate the target task to all benchmarked entrypoints."""
    constants.task = task_instruction
    default_main.task = task_instruction
    toon_main.task = task_instruction
    main_toon_agent_module.task = task_instruction


def _load_reference(reference_path: Path) -> str:
    """Read and validate the reference output for a test case."""
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference summary not found at '{reference_path}'.")

    reference_text = reference_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise ValueError(f"Reference summary at '{reference_path}' is empty.")
    return reference_text


def _accumulate(results: Iterable[BenchmarkScoreResult], store: dict[str, list[BenchmarkScoreResult]]) -> None:
    for result in results:
        store[result.name].append(result)


def main() -> None:
    if not constants.tests:
        raise ValueError("No tests configured in 'src.constants.tests'.")

    base_dir = Path(__file__).resolve().parent
    aggregates: dict[str, list[BenchmarkScoreResult]] = defaultdict(list)
    original_task = constants.task

    try:
        for task_instruction, reference_rel_path in constants.tests:
            if not isinstance(task_instruction, str) or not task_instruction.strip():
                raise ValueError("Each test must define a non-empty task instruction.")
            if not isinstance(reference_rel_path, str) or not reference_rel_path.strip():
                raise ValueError("Each test must provide a non-empty reference path.")

            _apply_task(task_instruction.strip())

            reference_path = (base_dir / reference_rel_path).resolve()
            reference_text = _load_reference(reference_path)

            results = benchmark_with_score(reference_text)
            results.sort(key=lambda item: (-item.f1_score, item.duration_seconds))

            print(f"\nTask: {task_instruction.strip()}")
            for result in results:
                total_tokens = result.summary.total_tokens
                print(
                    f"  {result.name}: f1={result.f1_score:.3f} | "
                    f"duration={result.duration_seconds:.3f}s | "
                    f"total_tokens={total_tokens if total_tokens is not None else 'n/a'}"
                )

            _accumulate(results, aggregates)

        if not aggregates:
            raise RuntimeError("Benchmark execution produced no results.")

        summary = [
            AgentAggregate.from_results(name, agent_results)
            for name, agent_results in aggregates.items()
        ]
        summary.sort(key=lambda item: (-item.average_f1, item.average_duration_seconds))

        print("\n=== Aggregated Scores ===")
        for aggregate in summary:
            token_details = (
                f"avg_tokens={aggregate.average_total_tokens:.1f} | total_tokens={aggregate.total_tokens}"
                if aggregate.average_total_tokens is not None
                else "avg_tokens=n/a | total_tokens=n/a"
            )
            print(
                f"{aggregate.name}: tests={aggregate.runs} | "
                f"avg_f1={aggregate.average_f1:.3f} | "
                f"avg_duration={aggregate.average_duration_seconds:.3f}s | "
                f"total_duration={aggregate.total_duration_seconds:.3f}s | "
                f"{token_details}"
            )
    finally:
        _apply_task(original_task)


if __name__ == "__main__":
    main()


