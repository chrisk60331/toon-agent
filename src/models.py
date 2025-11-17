from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, ValidationError, constr
from typing_extensions import Literal, runtime_checkable
from toon import encode



class LLMResponse(BaseModel):
    """Container for LLM responses, including token accounting."""

    content: str
    usage: Mapping[str, int] | None = None



class PlanStep(BaseModel):
    """Single step in the agent's validated plan."""

    description: constr(min_length=1)
    status: Literal["pending", "in_progress", "done"] = "pending"


class Scratchpad(BaseModel):
    """Validated reflective state for intermediate planning."""

    kind: Literal["scratchpad"]
    summary: constr(min_length=1)
    plan: List[PlanStep]
    notes: Optional[str] = None


class ActionPayload(BaseModel):
    """Validated action proposal produced by the model."""

    kind: Literal["action"]
    tool: constr(min_length=1)
    input: Dict[str, Any]


class Observation(BaseModel):
    """Observed result from executing a tool call."""

    tool: constr(min_length=1)
    content: str
    success: bool = True
    should_stop: bool = False
    usage: TokenUsage | None = None


class TokenUsage(BaseModel):
    """Accumulates token usage metrics for benchmarking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, usage: Mapping[str, Any] | None) -> "TokenUsage":
        if not usage:
            return cls()

        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt + completion))
        raw = {key: int(value) for key, value in usage.items()}
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            raw=raw,
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        raw_keys = set(self.raw) | set(other.raw)
        merged_raw = {key: self.raw.get(key, 0) + other.raw.get(key, 0) for key in raw_keys}
        total = self.total_tokens + other.total_tokens
        # Fall back to sum of prompt+completion if totals were not provided.
        if not total:
            total = (self.prompt_tokens + other.prompt_tokens) + (
                self.completion_tokens + other.completion_tokens
            )

        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=total,
            raw=merged_raw,
        )

    def accumulate(self, other: "TokenUsage") -> None:
        updated = self + other
        self.prompt_tokens = updated.prompt_tokens
        self.completion_tokens = updated.completion_tokens
        self.total_tokens = updated.total_tokens
        self.raw = updated.raw


class StepMetrics(BaseModel):
    """Tracks validator loop behavior and token usage per step."""

    step_index: int
    output_kind: Literal["scratchpad", "action"]
    attempts: int
    usage: TokenUsage
    raw_contents: List[str]
    validation_errors: List[str]


class AgentHistoryEntry(BaseModel):
    """History element shared with the LLM in TOON format."""

    kind: Literal["scratchpad", "action"]
    payload: Dict[str, Any]
    observation: Optional[Dict[str, Any]] = None


class AgentState(BaseModel):
    """Structured view of the agent's working memory."""

    task: str
    history: List[AgentHistoryEntry] = Field(default_factory=list)

    MAX_HISTORY_ITEMS: ClassVar[int] = 3
    SUMMARY_LIMIT: ClassVar[int] = 160
    NOTE_LIMIT: ClassVar[int] = 160
    OBSERVATION_LIMIT: ClassVar[int] = 180
    PLAN_LIMIT: ClassVar[int] = 2
    INPUT_LIMIT: ClassVar[int] = 160

    def append_scratchpad(self, scratchpad: Scratchpad) -> None:
        entry = AgentHistoryEntry(
            kind="scratchpad",
            payload={
                "summary": self._clip_text(scratchpad.summary, self.SUMMARY_LIMIT),
                "plan": self._compress_plan(scratchpad.plan),
                "notes": (
                    self._clip_text(scratchpad.notes, self.NOTE_LIMIT)
                    if scratchpad.notes
                    else None
                ),
            },
        )
        self.history.append(entry)

    def append_action(self, action: ValidatedAction, observation: Observation) -> None:
        encoded_input = encode(action.input_dict) if action.input_dict else "{}"
        compressed_input = self._clip_text(encoded_input, self.INPUT_LIMIT)
        compressed_observation = {
            "success": observation.success,
            "content": self._clip_text(observation.content, self.OBSERVATION_LIMIT),
            "should_stop": observation.should_stop,
            "full_content": observation.content,
        }
        entry = AgentHistoryEntry(
            kind="action",
            payload={
                "tool": action.tool,
                "input": compressed_input,
            },
            observation=compressed_observation,
        )
        self.history.append(entry)

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "history": [entry.model_dump() for entry in self.history[-self.MAX_HISTORY_ITEMS :]],
        }

    def _clip_text(self, value: Optional[str], limit: int) -> Optional[str]:
        if value is None:
            return None
        if len(value) <= limit:
            return value
        if limit <= 3:
            return value[:limit]
        return f"{value[: limit - 3]}..."

    def _compress_plan(self, plan: Sequence[PlanStep]) -> List[Dict[str, str]]:
        preview: List[Dict[str, str]] = []
        for step in list(plan)[: self.PLAN_LIMIT]:
            preview.append(
                {
                    "description": self._clip_text(step.description, self.SUMMARY_LIMIT // 2) or "",
                    "status": step.status,
                }
            )
        if len(plan) > self.PLAN_LIMIT:
            preview.append({"description": "...", "status": "pending"})
        return preview


class AgentRunResult(BaseModel):
    """Result container returned after running the agent loop."""

    task: str
    steps: List[StepMetrics]
    state: AgentState
    total_usage: TokenUsage


@dataclass
class ToolSpec:
    """Descriptor for registered tools with schema-first inputs."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], Observation]

