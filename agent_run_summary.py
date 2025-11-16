from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field


class AgentRunSummary(BaseModel):
    """Normalized view of usage metrics emitted by agent entrypoints."""

    total_tokens: int | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_usage(
        cls,
        usage_obj: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        extra_usage: Mapping[str, Any] | None = None,
    ) -> "AgentRunSummary":
        usage = cls.serialize_usage(usage_obj)
        if extra_usage:
            usage.update(dict(extra_usage))

        total_tokens = cls._extract_total_tokens(usage_obj, usage)
        meta = dict(metadata) if metadata else {}

        return cls(total_tokens=total_tokens, usage=usage, metadata=meta)

    @staticmethod
    def _extract_total_tokens(usage_obj: Any, usage_dict: Mapping[str, Any]) -> int | None:
        if "total_tokens" in usage_dict:
            candidate = usage_dict["total_tokens"]
            if isinstance(candidate, int):
                return candidate
            try:
                return int(candidate)
            except (TypeError, ValueError):
                pass

        if usage_obj is not None:
            candidate = getattr(usage_obj, "total_tokens", None)
            if isinstance(candidate, int):
                return candidate
            try:
                return int(candidate)
            except (TypeError, ValueError):
                return None

        return None

    @staticmethod
    def serialize_usage(usage_obj: Any) -> dict[str, Any]:
        if usage_obj is None:
            return {}

        if isinstance(usage_obj, dict):
            return dict(usage_obj)

        if hasattr(usage_obj, "model_dump"):
            dumped = usage_obj.model_dump()
            if isinstance(dumped, dict):
                return dumped

        if hasattr(usage_obj, "dict"):
            dumped = usage_obj.dict()
            if isinstance(dumped, dict):
                return dumped

        usage: dict[str, Any] = {}
        for attr in (
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = getattr(usage_obj, attr, None)
            if value is not None:
                usage[attr] = value

        return usage

