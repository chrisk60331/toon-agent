from __future__ import annotations

from strands import Agent

from agent_run_summary import AgentRunSummary
from toon_tools import file_read
from constants import task
# Define a focused system prompt for file operations
FILE_SYSTEM_PROMPT = """You are a file operations specialist. You help users read, 
write, search, and modify files. Focus on providing clear information about file 
operations and always confirm when files have been modified.

Key Capabilities:
1. Read files with various options (full content, line ranges, search)
2. Create and write to files
3. Edit existing files with precision
4. Report file information and statistics

Always specify the full file path in your responses for clarity.
"""


def run_toon_agent() -> AgentRunSummary:
    """Execute the toon-specific agent and capture its usage metrics."""
    file_agent = Agent(
        system_prompt=FILE_SYSTEM_PROMPT,
        tools=[file_read],
    )
    

    response = file_agent(task)
    metrics_obj = getattr(response, "metrics", None)
    accumulated = getattr(metrics_obj, "accumulated_usage", None) if metrics_obj is not None else None

    return AgentRunSummary.from_usage(accumulated)


if __name__ == "__main__":
    summary = run_toon_agent()
    print(summary.usage or None)