# Toon Benchmark

## What It Is
- Benchmarks three Toon approaches:
  - Strands agent with built-in file tools (`main`).
  - Strands agent with custom Toon file tools (`main_toon`).
  - Custom Toon agent leveraging Toon compression for files, chain-of-thought, and state management (`main_toon_agent`).
- Records wall-clock runtime plus token usage emitted through `AgentRunSummary`.
- Provides a quick way to compare prompts and behaviors for the Toon agents.

## Running the Benchmark
- Install dependencies (once) with `uv sync`.
- Adjust the task prompt in `constants.py` by editing the `task` value; the benchmark uses this prompt for every agent run.
- Execute all benchmarks with:
  - `PYTHONPATH=. uv run benchmarks/run_benchmark_with_score.py`
- Results print to stdout with one line per entrypoint, including F1 score, runtime, and total token usage.

## Latest Results
- Run date: 2025-11-17 on macOS 14 (Apple Silicon assumed).
- Prompt: `task = "summarize file prize.json"`
- Observed metrics:

| Approach                                                   | Entrypoint        | F1   | Runtime (s) | Total Tokens |
|------------------------------------------------------------|-------------------|------|-------------|--------------|
| Strands agent with built-in file tools                     | `main`            | 0.490| 8.371       | 64,717       |
| Strands agent with custom Toon file tools                  | `main_toon`       | 0.337| 8.246       | 62,106       |
| Custom Toon agent with Toon compression + CoT + state mgmt | `main_toon_agent` | 0.477| 10.925      | 1,567        |

Re-run after changing `constants.py` to capture updated numbers.

