from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import boto3
from pydantic import BaseModel, constr
from toon import encode

from src.agent_run_summary import AgentRunSummary
from src.boto3_bedrock_client import Boto3AnthropicClient
from src.models import (
    Observation,
    ToolSpec,
    TokenUsage,
)
from src.toon_agent import (
    LLMClientProtocol,
    ToonAgent,
)
from src.constants import task, MODEL_ID


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
    file_path: Path,
    task_instruction: str,
) -> tuple[str, TokenUsage]:
    """Invoke the LLM to summarize structured data in a chat-friendly format."""
    serialized_payload = _format_payload_for_llm(payload)
    file_name = file_path.name
    prompt = (
        "You are a versatile assistant crafting the final response for a user.\n"
        f"The user's task is: {task_instruction}\n"
        f"You have already executed tool 'file_read' on '{file_path}'.\n"
        "Respond directly to the user using the template below so the answer aligns with prior summaries:\n"
        f"1. Start with: \"I'll read the file {file_name} for you.\"\n"
        "2. Next line: \"Tool #1: file_read\"\n"
        f"3. Add the heading '## Summary of {file_name}'\n"
        "4. Include a short paragraph: \"This JSON file contains detailed product information\n"
        "5. Produce the following sections using markdown subheadings and bullet lists exactly as titled:\n"
        "6. Keep the tone concise and factual, avoid speculation, and cap the response to roughly 400 tokens.\n"
        "7. If any item is unavailable, omit it rather than inventing data.\n"
        "Ground every statement in the tool output provided below:\n"
        f"{serialized_payload}"
    )
    response = llm_client.generate(
        prompt,
        system_prompt="You turn tool outputs into structured, factual chat responses.",
    )
    summary = response.content.strip()
    if not summary:
        raise ValueError("LLM returned an empty summary.")
    usage = TokenUsage.from_raw(response.usage)
    return summary, usage


def build_file_summary_handler(
    llm_client: LLMClientProtocol,
    task_instruction: str,
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

        summary, usage = summarize_file_contents(
            llm_client=llm_client,
            payload=contents,
            file_path=target_path,
            task_instruction=task_instruction,
        )
        return Observation(
            tool="file_summary",
            content=summary,
            success=True,
            should_stop=True,
            usage=usage,
        )

    return handler


def run_main_toon_agent() -> AgentRunSummary:
    """Run the general-purpose ToonAgent for the configured task."""
    region = os.environ.get("AWS_REGION")
    bedrock_runtime = (
        boto3.client("bedrock-runtime", region_name=region) if region else boto3.client("bedrock-runtime")
    )
    model_id = MODEL_ID

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
        handler=build_file_summary_handler(llm, task_instruction=task),
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

    history = list(result.state.history)
    final_action = next(
        (entry for entry in reversed(history) if entry.kind == "action"),
        None,
    )
    observation_summary = (
        final_action.observation.get("content") if final_action and final_action.observation else None
    )
    scratchpad_notes = next(
        (
            entry.payload.get("notes")
            for entry in reversed(history)
            if entry.kind == "scratchpad" and entry.payload.get("notes")
        ),
        None,
    )

    metadata: dict[str, object] = {"task": task}
    fallback_usage: TokenUsage | None = None

    final_answer: str | None = None
    answer_source = "unknown"

    if observation_summary:
        final_answer = observation_summary.strip()
        answer_source = "tool"
        metadata["observation_summary"] = observation_summary
    elif scratchpad_notes:
        final_answer = scratchpad_notes.strip()
        answer_source = "scratchpad"
        metadata["scratchpad_notes"] = scratchpad_notes
    else:
        fallback_request = DefaultAnswerRequest(task=task)
        fallback_answer, fallback_usage = generate_default_answer(llm, fallback_request)
        final_answer = fallback_answer.strip()
        answer_source = "fallback"
        metadata["fallback_answer"] = fallback_answer

    metadata["final_answer"] = final_answer
    metadata["answer_source"] = answer_source

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
    summary = run_main_toon_agent()

    task = summary.metadata.get("task", "Not specified")
    final_answer = summary.metadata.get("final_answer")
    answer_source = summary.metadata.get("answer_source", "unknown")
    print("Task:", task)

    if final_answer:
        print(f"Summary ({answer_source}): {final_answer}")
    else:
        print("Summary: not available.")

    print("Total tokens:", summary.total_tokens)
