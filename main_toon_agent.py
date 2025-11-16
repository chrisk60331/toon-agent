from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import boto3
from pydantic import BaseModel, constr
from toon import encode

from agent_run_summary import AgentRunSummary
from boto3_bedrock_client import Boto3AnthropicClient
from models import (
    Observation,
    ToolSpec,
    TokenUsage,
)
from toon_agent import (
    LLMClientProtocol,
    ToonAgent,
)
from constants import task


class FileSummaryInput(BaseModel):
    """Schema for the file summary tool."""

    path: constr(min_length=1)


class DefaultAnswerRequest(BaseModel):
    """Payload for the fallback LLM response."""

    task: constr(min_length=1)


def generate_default_answer(
    llm_client: LLMClientProtocol,
    request: DefaultAnswerRequest,
) -> tuple[str, TokenUsage]:
    """Leverage the LLM directly when no tools can satisfy the task."""
    prompt = (
        "Respond to the user request using your own knowledge. "
        "Do not reference unavailable tools.\n"
        f"Task: {request.task}"
    )
    response = llm_client.generate(prompt, system_prompt="You are a helpful assistant.")
    usage = TokenUsage.from_raw(response.usage)
    return response.content.strip(), usage


def _format_payload_for_llm(raw_payload: object) -> str:
    """Return a serialized payload optimized for LLM prompts."""
    try:
        encoded = encode(raw_payload)
    except Exception:  # pragma: no cover - defensive guard against encoding issues
        encoded = ""
    try:
        pretty_json = json.dumps(raw_payload, indent=2)
    except TypeError:
        pretty_json = str(raw_payload)
    if encoded and len(encoded) <= len(pretty_json):
        return encoded
    return pretty_json


def summarize_file_contents(
    *,
    llm_client: LLMClientProtocol,
    payload: object,
) -> str:
    """Invoke the LLM to summarize structured product data."""
    serialized_payload = _format_payload_for_llm(payload)
    prompt = (
        "You are a precise product analyst. Summarize the product information below.\n"
        "Focus on the product name, customer value, standout attributes, and any data quality issues.\n"
        "Respond with 2-3 concise sentences.\n"
        "Product data:\n"
        f"{serialized_payload}"
    )
    response = llm_client.generate(
        prompt,
        system_prompt="You turn structured catalog data into concise product summaries.",
    )
    summary = response.content.strip()
    if not summary:
        raise ValueError("LLM returned an empty summary.")
    return summary


def build_file_summary_handler(
    llm_client: LLMClientProtocol,
) -> Callable[[BaseModel], Observation]:
    """Generate a tool handler bound to the provided LLM client."""

    def handler(payload: BaseModel) -> Observation:
        args = payload.model_dump()
        target_path = Path(args["path"]).expanduser().resolve(strict=True)

        with target_path.open("r", encoding="utf-8") as file_obj:
            try:
                contents = json.load(file_obj)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to decode JSON at {target_path}") from exc

        summary = summarize_file_contents(llm_client=llm_client, payload=contents)
        return Observation(
            tool="file_summary",
            content=summary,
            success=True,
            should_stop=True,
        )

    return handler


def run_demo() -> AgentRunSummary:
    """Run the ToonAgent demo against the sample product file."""
    project_root = Path(__file__).resolve().parent
    target_file = project_root / "5060292302201.json"

    region = os.environ.get("AWS_REGION")
    bedrock_runtime = (
        boto3.client("bedrock-runtime", region_name=region) if region else boto3.client("bedrock-runtime")
    )
    model_id = os.environ.get(
        "TOON_AGENT_MODEL_ID",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )

    llm = Boto3AnthropicClient(
        client=bedrock_runtime,
        model_id=model_id,
        max_tokens=1024,
        temperature=0.0,
    )
    tool = ToolSpec(
        name="file_summary",
        description="Parse a JSON file and generate a concise product summary.",
        input_model=FileSummaryInput,
        handler=build_file_summary_handler(llm),
    )

    agent = ToonAgent(
        llm_client=llm,
        tools=[tool],
        system_prompt=(
            "You are a deterministic planner. Think with scratchpads that match the schema "
            "and issue tool actions only when you have a complete plan."
        ),
        max_steps=6,
        max_validation_attempts=2,
    )

    result = agent.run(task)

    final_action = next(
        (entry for entry in result.state.history if entry.kind in ["action", "thought"]),
        None,
    )
    observation_summary = (
        final_action.observation["content"] if final_action and final_action.observation else None
    )
    metadata: dict[str, object] = {"task": task}
    fallback_answer: str | None = None
    fallback_usage: TokenUsage | None = None
    if observation_summary:
        metadata["observation_summary"] = observation_summary
    else:
        fallback_request = DefaultAnswerRequest(task=task)
        fallback_answer, fallback_usage = generate_default_answer(llm, fallback_request)
        metadata["fallback_answer"] = fallback_answer

    step_summaries = [
        {
            "index": step.step_index,
            "kind": step.output_kind,
            "attempts": step.attempts,
            "usage": AgentRunSummary.serialize_usage(getattr(step, "usage", None)),
        }
        for step in result.steps
    ]

    usage_snapshot: TokenUsage = result.total_usage
    extra_usage: dict[str, object] = {"steps": step_summaries}
    if fallback_usage is not None:
        usage_snapshot = usage_snapshot + fallback_usage
        extra_usage["fallback"] = fallback_usage.model_dump()

    return AgentRunSummary.from_usage(
        usage_snapshot,
        metadata=metadata,
        extra_usage=extra_usage,
    )


if __name__ == "__main__":
    summary = run_demo()

    task = summary.metadata.get("task", "Not specified")
    print("Task:", task)

    observation = summary.metadata.get("observation_summary")
    if observation:
        print("Summary:", observation)
    else:
        print("Summary: not available (no action executed).")

    print("Total tokens:", summary.total_tokens)
