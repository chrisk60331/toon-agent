from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

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
from src.dataset_fact_sheet import build_fact_sheet


class FileSummaryInput(BaseModel):
    """Schema for the file summary tool."""

    path: constr(min_length=1)


class DefaultAnswerRequest(BaseModel):
    """Payload for the fallback LLM response."""

    task: constr(min_length=1)


def _build_summary_prompt(
    *,
    file_name: str,
    task_instruction: str,
    fact_sheet: str,
) -> str:
    header = (
        f"You are preparing the final answer for the task: {task_instruction}.\n"
        "Use only the information contained in the fact sheet; do not rely on external knowledge.\n"
        "Copy the template between the delimiter lines EXACTLY. Replace only the bracketed guidance with factual content from the fact sheet. Leave every other character unchanged, including blank lines.\n"
        "<<TEMPLATE>>\n"
        f"I'll read the file {file_name} for you.\n"
        "Tool #1: file_read\n"
        f"## Summary of {file_name}\n\n"
        "[One or two sentences summarizing dataset scope, size, and time span. Include specific numbers, dates, and key facts from the fact sheet.]\n\n"
        "### **File Structure:**\n"
        "- [Format and organization details]\n"
        "- [Total coverage or record count]\n"
        "### **Data Content:**\n"
        "[Describe main categories, types, or classifications. Use numbered lists if appropriate, or bullets for key categories.]\n"
        "### **Key Information per Record:**\n"
        "- [Field 1]: [Description with specific details]\n"
        "- [Field 2]: [Description with specific details]\n"
        "- [Field 3]: [Description if relevant]\n"
        "- [Field 4]: [Description if relevant]\n"
        "- [Additional fields as needed]\n"
        "### **Notable Features:**\n"
        "- [Feature 1: specific examples, dates, or statistics]\n"
        "- [Feature 2: specific examples, dates, or statistics]\n"
        "- [Feature 3: if relevant]\n"
        "- [Feature 4: if relevant]\n"
        "\n"
        "[Optional concluding sentence about the dataset's value or purpose.]\n"
        "<<END TEMPLATE>>\n"
        "Extract specific numbers, dates, ranges, percentages, and category names directly from the fact sheet. Use exact values when available.\n"
        "For Notable Features, prioritize: multi-prize counts, organizational recipients, gender diversity, geographic coverage, time ranges, derived insights, product details, nutritional information, or any other notable patterns mentioned in the fact sheet.\n"
        "Keep the response comprehensive but concise. Aim for 350-450 tokens. Use bold formatting for section headers as shown in the template.\n"
        "Return only the filled-in template (without the delimiters) and do not add extra commentary.\n"
    )
    return f"{header}\nFACT SHEET:\n{fact_sheet}"


def generate_default_answer(
    llm_client: LLMClientProtocol,
    request: DefaultAnswerRequest,
) -> tuple[str, TokenUsage]:
    """Leverage the LLM directly when no tools can satisfy the task."""
    prompt = f"Task: {request.task}\nAnswer directly and concisely using only generally known facts."
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
        pretty_json = json.dumps(raw_payload, separators=(",", ":"))
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
    file_name = file_path.name
    fact_sheet = build_fact_sheet(payload, file_name)
    if not fact_sheet.strip():
        serialized_payload = _format_payload_for_llm(payload)
        fact_sheet = f"FACT SHEET UNAVAILABLE. RAW PAYLOAD PREVIEW:\n{serialized_payload}"

    prompt = _build_summary_prompt(
        file_name=file_name,
        task_instruction=task_instruction,
        fact_sheet=fact_sheet,
    )
    response = llm_client.generate(
        prompt,
        system_prompt=(
            "You transform fact sheets into structured, factual dataset summaries. "
            "Extract specific details like numbers, dates, ranges, percentages, and categories from the fact sheet. "
            "Follow the template format precisely, using bold headers and structured sections. "
            "Prioritize accuracy and completeness over brevity."
        ),
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
        description="Summarize structured JSON content.",
        input_model=FileSummaryInput,
        handler=build_file_summary_handler(llm, task_instruction=task),
    )

    agent = ToonAgent(
        llm_client=llm,
        tools=[tool],
        system_prompt=(
            "Deterministic planner using TOON schema. Work stepwise and call tools only when ready."
        ),
        max_steps=4,
        max_validation_attempts=3,
    )

    result = agent.run(task)

    history = list(result.state.history)
    final_action = next(
        (entry for entry in reversed(history) if entry.kind == "action"),
        None,
    )
    observation_summary = (
        (
            final_action.observation.get("full_content")
            or final_action.observation.get("content")
        )
        if final_action and final_action.observation
        else None
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
