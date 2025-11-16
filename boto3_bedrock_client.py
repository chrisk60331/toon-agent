from __future__ import annotations
from botocore.client import BaseClient
import json

from typing import Any
from models import *


class Boto3AnthropicClient:
    """LLM client that calls Anthropic models via AWS Bedrock runtime."""

    def __init__(
        self,
        *,
        client: BaseClient,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._anthropic_version = "bedrock-2023-05-31"

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        system_prompt = kwargs.get("system_prompt")

        body = {
            "anthropic_version": self._anthropic_version,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        if system_prompt:
            body["system"] = system_prompt

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        raw_body = response["body"].read()
        payload = json.loads(raw_body.decode("utf-8"))

        content_blocks = payload.get("content", [])
        text = None
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                candidate = block.get("text")
                if candidate:
                    text = candidate
                    break

        if text is None:
            raise RuntimeError("Anthropic response missing text content.")

        usage_data = payload.get("usage") or {}
        prompt_tokens = int(usage_data.get("input_tokens", 0))
        completion_tokens = int(usage_data.get("output_tokens", 0))
        total_tokens = int(
            usage_data.get("total_tokens", prompt_tokens + completion_tokens)
        )

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        return LLMResponse(content=text, usage=usage)
