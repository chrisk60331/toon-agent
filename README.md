# Toon Agent Benchmarking

This project benchmarks and compares different agent implementations for structured data summarization tasks, specifically focusing on file summarization using the TOON schema.

## What This Project Does

This repository contains a benchmarking suite that evaluates three different agent implementations:

1. **`main`** - Default Strands agent with full file operation tools (`file_read`, `file_write`, `editor`)
2. **`main_toon`** - TOON-specific Strands agent with `file_read` tool only
3. **`main_toon_agent`** - Custom `ToonAgent` implementation with a specialized `file_summary` tool that generates structured summaries from JSON datasets

The agents are tested on tasks that involve summarizing structured JSON files (like prize data, food facts, laureate information, etc.) and producing human-readable summaries.

### Key Components

- **`ToonAgent`** (`src/toon_agent.py`): A custom agent implementation that enforces TOON-validated planning and action execution
- **Dataset Fact Sheet Builder** (`src/dataset_fact_sheet.py`): Generates structured fact sheets from JSON data for LLM consumption
- **Benchmark Suite** (`benchmarks/`): Infrastructure for running, scoring, and aggregating benchmark results

## Prerequisites

- Python 3.11+
- AWS credentials configured (for Bedrock access)
- `AWS_REGION` environment variable set (optional, defaults to default region)
- Dependencies installed via `uv` (see `pyproject.toml`)

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Configuration

Benchmark tasks and reference outputs are configured in `src/constants.py`:

```python
task, SUMMARY_FILE = "summarize file data/history.json", "../data/prize_summary.txt"

tests = [
    ("summarize file data/prize.json", "../data/prize_summary.txt"),
    # Add more test cases here
]
```

## Running Benchmarks

### Basic Benchmark (Timing and Token Usage)

Run all agents and collect timing and token usage metrics:

```bash
cd benchmarks
python benchmark.py
```

This will output:
- Agent name
- Execution duration (seconds)
- Total tokens consumed
- Detailed usage breakdown

### Benchmark with F1 Scoring

Run benchmarks and score outputs against a reference answer:

```bash
cd benchmarks
python run_benchmark_with_score.py
```

This requires a reference summary file (configured via `SUMMARY_FILE` in `src/constants.py`). The script:
1. Loads the reference summary
2. Runs all three agents
3. Computes F1 scores comparing each agent's output to the reference
4. Displays results sorted by F1 score (descending), then duration (ascending)

### Aggregate Scores Across Multiple Tests

Run the full test suite and aggregate results:

```bash
# From project root (recommended)
PYTHONPATH=$(pwd):$PYTHONPATH uv run python benchmarks/aggregate_scores.py

# Or if running from benchmarks directory
cd benchmarks
PYTHONPATH=..:$PYTHONPATH uv run python aggregate_scores.py
```

This script:
1. Iterates through all test cases in `src/constants.tests`
2. Runs benchmarks with scoring for each test
3. Aggregates results across all tests
4. Displays:
   - Per-test results for each agent
   - Aggregated statistics (average F1, average duration, total tokens, etc.)

## Interpreting Benchmark Results

### Metrics Explained

#### F1 Score (0.0 - 1.0)
- **What it measures**: Token-level similarity between agent output and reference answer
- **How it's calculated**: 
  - Precision = (matching tokens) / (total tokens in agent output)
  - Recall = (matching tokens) / (total tokens in reference)
  - F1 = 2 × (precision × recall) / (precision + recall)
- **Interpretation**:
  - **1.0**: Perfect match (all tokens align)
  - **0.7-0.9**: Very good similarity
  - **0.5-0.7**: Moderate similarity
  - **<0.5**: Low similarity, may indicate different approach or errors

#### Duration (seconds)
- **What it measures**: Wall-clock time for agent execution
- **Interpretation**: Lower is better. Includes LLM API calls, tool execution, and validation overhead

#### Total Tokens
- **What it measures**: Total token consumption (input + output) across all LLM calls
- **Interpretation**: Lower is better. Includes:
  - Prompt tokens (system prompt, task, state history, tool schemas)
  - Completion tokens (agent responses, tool outputs)
  - Retry attempts (validation failures consume additional tokens)

### Understanding Output

Example benchmark output:

```
Task: summarize file data/prize.json
  main_toon_agent: f1=0.852 | duration=3.245s | total_tokens=2847
  main_toon: f1=0.791 | duration=2.891s | total_tokens=2156
  main: f1=0.743 | duration=4.102s | total_tokens=3124

=== Aggregated Scores ===
main_toon_agent: tests=1 | avg_f1=0.852 | avg_duration=3.245s | total_duration=3.245s | avg_tokens=2847.0 | total_tokens=2847
```

**Reading this output**:
- `main_toon_agent` achieved the highest F1 score (0.852) but used the most tokens
- `main_toon` was fastest (2.891s) with good F1 (0.791) and moderate token usage
- `main` was slowest and had lowest F1, suggesting the additional tools may not help for this task

### What Makes a Good Result?

- **High F1 score**: Agent output closely matches the reference, indicating correct task completion
- **Low duration**: Fast execution, important for user experience
- **Reasonable token usage**: Efficient use of LLM calls without excessive retries

**Trade-offs to consider**:
- **Accuracy vs. Efficiency**: `main` achieves higher F1 scores but consumes 30x more tokens than `main_toon_agent`
- **Token Efficiency**: `main_toon_agent` uses specialized fact sheet generation, dramatically reducing token usage while maintaining reasonable quality
- **Speed**: All agents have similar execution times, so token efficiency is the primary cost differentiator
- The "best" agent depends on your priorities: accuracy (F1), speed, or cost efficiency (tokens)

## Benchmark Results

### Aggregated Scores (5 Test Cases)

| Agent | Tests | Avg F1 | Avg Duration | Total Duration | Avg Tokens | Total Tokens |
|-------|-------|--------|--------------|---------------|------------|--------------|
| `main` | 5 | 0.423 | 7.392s | 36.962s | 62,915.6 | 314,578 |
| `main_toon` | 5 | 0.322 | 6.801s | 34.006s | 52,844.2 | 264,221 |
| `main_toon_agent` | 5 | 0.315 | 6.856s | 34.282s | 2,006.6 | 10,033 |

**Key Observations**:
- **`main`** achieves the highest average F1 score (0.423) but consumes significantly more tokens (~31x more than `main_toon_agent`)
- **`main_toon_agent`** is the most token-efficient, using only ~2,000 tokens per test on average (97% reduction vs `main`)
- **`main_toon`** has the fastest average duration (6.801s) but still consumes high token counts
- All agents have similar execution times (6.8-7.4s average), suggesting token efficiency is the primary differentiator

### Per-Task Breakdown

| Task | Agent | F1 Score | Duration | Total Tokens |
|------|-------|----------|----------|--------------|
| **summarize file data/prize.json** | `main` | 0.545 | 8.041s | 64,741 |
| | `main_toon_agent` | 0.432 | 9.099s | 2,398 |
| | `main_toon` | 0.368 | 9.230s | 62,138 |
| **pick any number between 1 and 100** | `main_toon_agent` | 0.333 | 1.600s | 441 |
| | `main` | 0.185 | 2.078s | 3,227 |
| | `main_toon` | 0.179 | 2.167s | 752 |
| **summarize file data/food_facts.json** | `main` | 0.534 | 8.253s | 22,939 |
| | `main_toon` | 0.437 | 8.279s | 20,034 |
| | `main_toon_agent` | 0.229 | 7.915s | 2,538 |
| **summarize file data/laureate.json** | `main` | 0.419 | 10.094s | 153,047 |
| | `main_toon` | 0.405 | 10.144s | 179,484 |
| | `main_toon_agent` | 0.307 | 7.344s | 2,610 |
| **summarize file data/history.json** | `main` | 0.431 | 8.496s | 70,624 |
| | `main_toon_agent` | 0.276 | 8.323s | 2,046 |
| | `main_toon` | 0.220 | 4.186s | 1,813 |

**Task-Specific Insights**:
- **File summarization tasks**: `main` consistently achieves the highest F1 scores but at a high token cost
- **Simple tasks** (like picking a number): `main_toon_agent` performs best with minimal token usage
- **Token efficiency**: `main_toon_agent` uses 10-30x fewer tokens across all file summarization tasks
- **Performance variance**: F1 scores vary significantly by task, suggesting task-specific optimization may be beneficial

## Project Structure

```
toon/
├── benchmarks/           # Benchmark execution scripts
│   ├── benchmark.py      # Core benchmarking infrastructure
│   ├── run_benchmark_with_score.py  # Benchmarks with F1 scoring
│   └── aggregate_scores.py           # Multi-test aggregation
├── src/
│   ├── toon_agent.py     # Custom ToonAgent implementation
│   ├── main_toon_agent.py # Main entrypoint using ToonAgent
│   ├── dataset_fact_sheet.py  # Fact sheet generation
│   ├── constants.py      # Configuration (tasks, model ID)
│   └── models.py         # Pydantic models for agent state
├── data/                 # Test datasets and reference summaries
└── pyproject.toml        # Project dependencies
```

## Adding New Test Cases

1. Add your test dataset JSON file to `data/`
2. Create a reference summary in `data/` (manually or by running an agent and saving output)
3. Add the test case to `src/constants.py`:

```python
tests = [
    ("summarize file data/your_file.json", "../data/your_reference.txt"),
    # ... existing tests
]
```

4. Run `python benchmarks/aggregate_scores.py` to include it in the suite

## Notes

- All agents use AWS Bedrock with the model specified in `MODEL_ID` (default: `us.anthropic.claude-haiku-4-5-20251001-v1:0`)
- The `ToonAgent` enforces strict TOON schema validation with retry logic (up to 3 attempts per step)
- Token usage tracking includes all LLM calls, including validation retries
- Benchmarks suppress stdout/stderr during execution to avoid cluttering output

