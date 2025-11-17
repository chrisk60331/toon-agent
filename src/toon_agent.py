from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, ValidationError, constr
from typing_extensions import Literal, runtime_checkable
import toon

from src.models import *

DecodeOptions = toon.DecodeOptions
ToonDecodeError = toon.ToonDecodeError
decode = toon.decode
encode = toon.encode


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal protocol for pluggable LLM clients."""

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        ...


@dataclass
class ValidatedAction:
    """Action payload enriched with validated tool arguments."""

    payload: ActionPayload
    arguments: BaseModel

    @property
    def tool(self) -> str:
        return self.payload.tool

    @property
    def input_dict(self) -> Dict[str, Any]:
        return self.arguments.model_dump()


class ToonAgent:
    """Agent that enforces TOON-validated planning and action execution."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        tools: Sequence[ToolSpec],
        *,
        system_prompt: str,
        max_steps: int = 8,
        max_validation_attempts: int = 3,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_validation_attempts < 1:
            raise ValueError("max_validation_attempts must be at least 1")

        self._llm = llm_client
        self._tool_registry: Dict[str, ToolSpec] = {tool.name: tool for tool in tools}
        self._system_prompt = system_prompt.strip()
        self._max_steps = max_steps
        self._max_validation_attempts = max_validation_attempts
        self._decode_options = DecodeOptions(strict=True)

        if not self._tool_registry:
            raise ValueError("At least one tool must be registered.")

    def run(self, task: str) -> AgentRunResult:
        state = AgentState(task=task)
        step_metrics: List[StepMetrics] = []
        cumulative_usage = TokenUsage()

        for step_idx in range(1, self._max_steps + 1):
            output, metrics = self._next_validated_output(
                state=state,
                step_index=step_idx,
            )

            if isinstance(output, Scratchpad):
                cumulative_usage.accumulate(metrics.usage)
                step_metrics.append(metrics)
                state.append_scratchpad(output)
                if self._should_stop_after_scratchpad(output):
                    break
                continue

            observation = self._execute_action(output)
            if observation.usage is not None:
                metrics.usage.accumulate(observation.usage)

            cumulative_usage.accumulate(metrics.usage)
            step_metrics.append(metrics)
            state.append_action(output, observation)
            if observation.should_stop:
                break

        return AgentRunResult(
            task=task,
            steps=step_metrics,
            state=state,
            total_usage=cumulative_usage,
        )

    def _execute_action(self, action: ValidatedAction) -> Observation:
        tool = self._tool_registry.get(action.tool)
        if tool is None:
            raise ValueError(f"Tool '{action.tool}' is not registered.")

        return tool.handler(action.arguments)

    @staticmethod
    def _should_stop_after_scratchpad(scratchpad: Scratchpad) -> bool:
        if scratchpad.plan and any(step.status != "done" for step in scratchpad.plan):
            return False

        if scratchpad.notes and scratchpad.notes.strip():
            return True

        return False

    def _next_validated_output(
        self,
        *,
        state: AgentState,
        step_index: int,
    ) -> tuple[Scratchpad | ValidatedAction, StepMetrics]:
        validation_errors: List[str] = []
        raw_contents: List[str] = []
        aggregated_usage = TokenUsage()

        for attempt in range(1, self._max_validation_attempts + 1):
            prompt = self._render_prompt(
                task=state.task,
                state_payload=state.prompt_payload(),
                validation_error=validation_errors[-1] if validation_errors else None,
            )
            response = self._llm.generate(prompt, system_prompt=self._system_prompt)
            usage = TokenUsage.from_raw(response.usage)
            aggregated_usage.accumulate(usage)
            raw_contents.append(response.content)

            try:
                parsed = self._parse_output(response.content)
                metrics = StepMetrics(
                    step_index=step_index,
                    output_kind="action" if isinstance(parsed, ValidatedAction) else "scratchpad",
                    attempts=attempt,
                    usage=aggregated_usage,
                    raw_contents=raw_contents,
                    validation_errors=validation_errors,
                )
                return parsed, metrics
            except (ValidationError, ToonDecodeError, ValueError) as err:
                error_message = f"{type(err).__name__}: {err}"
                validation_errors.append(error_message)

        error_details = "\n".join(
            f"Attempt {i+1}: {err}" for i, err in enumerate(validation_errors)
        )
        raw_outputs_preview = "\n".join(
            f"Attempt {i+1} output (first 200 chars): {content[:200]}..."
            if len(content) > 200
            else f"Attempt {i+1} output: {content}"
            for i, content in enumerate(raw_contents)
        )
        raise RuntimeError(
            f"Failed to produce a valid TOON output after {self._max_validation_attempts} attempts.\n"
            f"Validation errors:\n{error_details}\n\n"
            f"Raw LLM outputs:\n{raw_outputs_preview}"
        )

    def _render_prompt(
        self,
        *,
        task: str,
        state_payload: Dict[str, Any],
        validation_error: Optional[str],
    ) -> str:
        tools_summary = self._format_tools_overview()
        history_summary = self._format_history_summary(state_payload.get("history", []))
        schema_notes = self._schema_notes()
        example_payload = self._concise_example()

        prompt_segments = [
            self._system_prompt,
            f"Task:\n{task}",
            "Tools:",
            tools_summary,
            "State:",
            history_summary,
            "Schemas:",
            schema_notes,
            "Example:",
            example_payload,
            "Instruction:",
            (
                "Return exactly one TOON object with no leading or trailing commentary, "
                "markdown, or JSON formatting. Emit either a scratchpad or an action that "
                "conforms to the schemas. Do not include explanations, and do not send "
                "multiple objects. Use the TOON encoding shown in the examples; do not fall "
                "back to plain YAML or Markdown. If you can complete the task without using "
                "tools, set every plan step status to 'done' and place the final answer in the "
                "scratchpad notes."
            ),
        ]

        if validation_error:
            prompt_segments.extend(
                [
                    "Previous attempt failed validation. Fix the issue described below:",
                    validation_error,
                ]
            )

        return "\n".join(segment for segment in prompt_segments if segment)

    def _parse_output(self, content: str) -> Scratchpad | ValidatedAction:
        # Strip markdown code blocks if present
        cleaned_content = content.strip()
        if cleaned_content.startswith("```"):
            # Remove opening code block (with optional language tag)
            lines = cleaned_content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing code block if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_content = "\n".join(lines).strip()
        
        try:
            decoded = decode(cleaned_content, options=self._decode_options)
            if not isinstance(decoded, dict):
                raise ValueError("LLM output must decode to an object.")
            data = decoded
        except ToonDecodeError as e:
            raise ValueError(f"Failed to decode TOON format: {e}") from e

        kind = data.get("kind")
        if kind == "scratchpad":
            return Scratchpad.model_validate(data)

        if kind == "action":
            payload = ActionPayload.model_validate(data)
            tool = self._tool_registry.get(payload.tool)
            if tool is None:
                raise ValueError(f"Tool '{payload.tool}' is not registered.")

            arguments = tool.input_model.model_validate(payload.input)
            return ValidatedAction(payload=payload, arguments=arguments)

        raise ValueError(
            "LLM output must set 'kind' to either 'scratchpad' or 'action'. "
            f"Received payload: {data!r}"
        )

    def _format_tools_overview(self) -> str:
        tools_payload: List[Dict[str, Any]] = []
        for tool in self._tool_registry.values():
            field_entries: List[Dict[str, Any]] = []
            for name, field_info in tool.input_model.model_fields.items():  # type: ignore[attr-defined]
                field_type = self._short_type_name(field_info.annotation)
                required_callable = getattr(field_info, "is_required", None)
                required = bool(required_callable()) if callable(required_callable) else True
                field_entries.append(
                    {"name": name, "type": field_type, "required": required}
                )

            tools_payload.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputs": field_entries,
                }
            )

        return encode({"tools": tools_payload})

    def _format_history_summary(self, history: Sequence[Mapping[str, Any]]) -> str:
        if not history:
            return encode({"history": []})

        history_items: List[Dict[str, Any]] = []
        for entry in history:
            kind = entry.get("kind", "unknown")
            payload = entry.get("payload", {})
            item: Dict[str, Any] = {"kind": kind}

            if kind == "scratchpad":
                item["summary"] = payload.get("summary", "")
                item["plan"] = payload.get("plan", [])
                notes = payload.get("notes")
                if notes:
                    item["notes"] = notes
            elif kind == "action":
                item["tool"] = payload.get("tool", "")
                item["input"] = payload.get("input", "")
            else:
                item["details"] = payload

            observation = entry.get("observation")
            if observation:
                item["observation"] = {
                    "success": observation.get("success", True),
                    "content": observation.get("content", ""),
                }
                if observation.get("should_stop"):
                    item["observation"]["should_stop"] = True

            history_items.append(item)

        return encode({"history": history_items})

    @staticmethod
    def _schema_notes() -> str:
        schema_payload = {
            "scratchpad": {
                "kind": "scratchpad",
                "summary": "string",
                "plan": [{"description": "string", "status": "pending|in_progress|done"}],
                "notes": "optional string",
            },
            "action": {
                "kind": "action",
                "tool": "string",
                "input": {"argument": "value"},
            },
        }
        return encode(schema_payload)

    @staticmethod
    def _concise_example() -> str:
        scratchpad_example = encode(
            {
                "kind": "scratchpad",
                "summary": "Confirm the task before using tools.",
                "plan": [
                    {"description": "Decide whether any tool is required", "status": "pending"},
                ],
            }
        )
        action_example = encode(
            {
                "kind": "action",
                "tool": "tool_name",
                "input": {"example_field": "value"},
            }
        )
        return f"Scratchpad example:\n{scratchpad_example}\nAction example:\n{action_example}"

    @staticmethod
    def _short_type_name(annotation: Any) -> str:
        if annotation is None:
            return "unknown"
        if getattr(annotation, "__module__", "") == "builtins":
            return getattr(annotation, "__name__", str(annotation))

        annotation_str = str(annotation)
        return annotation_str.replace("typing.", "")

