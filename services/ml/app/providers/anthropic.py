"""Anthropic Messages API adapter hidden behind the gateway provider protocol.

Selected with LLM_PROVIDER=anthropic. Structured output uses
`output_config.format` (constrained JSON Schema decoding), the Claude
counterpart of OpenAI's strict `json_schema`. The gateway still validates the
answer against the pydantic schema and retries once, exactly as for OpenAI.

Replay caveat: cassette keys include the provider and model, and every LLM
cassette in fixtures/cassettes was recorded with OpenAI. With
LLM_PROVIDER=anthropic and GATEWAY_MODE=replay the model-backed endpoints
answer 503 AI_UNAVAILABLE until Claude cassettes are recorded.
"""

from __future__ import annotations

import json
from typing import Any

from ..gateway.errors import GatewayProviderError
from ..gateway.types import ProviderRequest, ProviderResponse


# Claude Haiku 5.5 accepts thinking "disabled"; Claude Sonnet 5.5 rejects it and
# its lowest setting is "between_tools". Thinking tokens count toward
# max_tokens, so leaving thinking on would eat the configured output budget.
_THINKING_BY_MODEL_PREFIX: tuple[tuple[str, dict[str, str]], ...] = (
    ("claude-haiku", {"type": "disabled"}),
    ("claude-sonnet", {"type": "between_tools"}),
)


def _thinking_for(model: str) -> dict[str, str] | None:
    for prefix, thinking in _THINKING_BY_MODEL_PREFIX:
        if model.startswith(prefix):
            return thinking
    return None


class AnthropicProvider:
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        self._client = client

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        arguments: dict[str, Any] = {
            "model": request.model,
            "system": request.prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        request.payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
            ],
            "max_tokens": request.max_tokens,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _wire_schema(request.output_schema.model_json_schema()),
                }
            },
        }
        thinking = _thinking_for(request.model)
        if thinking is not None:
            arguments["thinking"] = thinking

        try:
            response = await self._client.messages.create(**arguments)
        except Exception as error:
            raise GatewayProviderError("Anthropic request failed") from error

        # "refusal" and "max_tokens" may carry text that does not match the schema.
        if getattr(response, "stop_reason", None) != "end_turn":
            raise GatewayProviderError("Anthropic response was incomplete")
        output_text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", None) or []
            if getattr(block, "type", None) == "text"
        )
        if not output_text:
            raise GatewayProviderError("Anthropic response contained no output")

        usage = getattr(response, "usage", None)
        return ProviderResponse(
            content=output_text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            request_id=getattr(response, "id", None),
        )


# Constrained decoding rejects these keywords with a 400. The pydantic model
# still enforces them when the gateway validates the answer.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
    }
)


def _wire_schema(node: Any) -> Any:
    """Strip keywords Claude's JSON Schema grammar does not support."""

    if isinstance(node, list):
        return [_wire_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        if key == "minItems" and isinstance(value, int) and value > 1:
            result[key] = 1
            continue
        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            result[key] = {name: _wire_schema(item) for name, item in value.items()}
            continue
        result[key] = _wire_schema(value)
    if result.get("type") == "object" or "properties" in result:
        result["additionalProperties"] = False
    return result
