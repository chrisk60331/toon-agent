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
  - `uv run python benchmark.py`
- Results print to stdout with one line per entrypoint.

## Latest Results
- Run date: 2025-11-15 on macOS 14 (Apple Silicon assumed).
- Prompt: `task = "summarize file prize.json"`
- Observed metrics:

| Approach                                                   | Entrypoint        | Runtime (s) | Input/Prompt Tokens | Output/Completion Tokens | Total Tokens |
|------------------------------------------------------------|-------------------|-------------|---------------------|---------------------------|--------------|
| Strands agent with built-in file tools                     | `main`            | 14.230      | 63,882              | 506                       | 64,388       |
| Strands agent with custom Toon file tools                  | `main_toon`       | 14.608      | 61,219              | 470                       | 61,689       |
| Custom Toon agent with Toon compression + CoT + state mgmt | `main_toon_agent` | 8.942       | 836                 | 131                       | 967          |

Re-run after changing `constants.py` to capture updated numbers.

