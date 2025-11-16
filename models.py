from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, ValidationError, constr
from typing_extensions import Literal, runtime_checkable



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

    def append_scratchpad(self, scratchpad: Scratchpad) -> None:
        self.history.append(
            AgentHistoryEntry(
                kind="scratchpad",
                payload=scratchpad.model_dump(),
            )
        )

    def append_action(self, action: ValidatedAction, observation: Observation) -> None:
        self.history.append(
            AgentHistoryEntry(
                kind="action",
                payload={
                    "tool": action.tool,
                    "input": action.input_dict,
                },
                observation=observation.model_dump(),
            )
        )

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "history": [entry.model_dump() for entry in self.history],
        }


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

