"""LLM_PROVIDER switches text tasks between OpenAI and Claude, offline."""

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from services.ml.app.config import Settings, load_settings
from services.ml.app.gateway.config import (
    ModelConfigurationError,
    ModelsConfiguration,
    Provider,
    TaskName,
    load_models_configuration,
)
from services.ml.app.gateway.errors import GatewayProviderError, GatewayReplayError
from services.ml.app.gateway.service import create_gateway
from services.ml.app.gateway.types import GatewayRequest, ProviderRequest
from services.ml.app.modules.judge import JudgeOutput
from services.ml.app.providers.anthropic import AnthropicProvider, _wire_schema


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1, max_length=20)
    score: int = Field(ge=1, le=5)


def provider_request(model: str = "claude-sonnet-5-5") -> ProviderRequest:
    return ProviderRequest(
        task=TaskName.BRIEF,
        model=model,
        prompt="Return JSON.",
        payload={"candidateId": "candidate-a"},
        output_schema=Answer,
        max_tokens=123,
    )


def fake_client(response=None, error: Exception | None = None):
    calls: list[dict] = []

    class Messages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            return response

    return SimpleNamespace(messages=Messages()), calls


def message(text: str = '{"value":"ok","score":3}', stop_reason: str = "end_turn"):
    return SimpleNamespace(
        id="synthetic-anthropic-id",
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=4),
    )


def settings(mode: str, provider: str = "openai") -> Settings:
    return Settings(
        ml_internal_token="test",
        uploads_dir=Path("/tmp"),
        gateway_mode=mode,
        budget_usd_cap=Decimal("1"),
        demo_mode=False,
        usage_log_path=Path("/tmp/llm-provider-test-usage.jsonl"),
        llm_provider=provider,
    )


def test_llm_provider_defaults_to_openai_and_rejects_unknown_values() -> None:
    assert load_settings({}).llm_provider == "openai"
    assert load_settings({"LLM_PROVIDER": " Anthropic "}).llm_provider == "anthropic"
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings({"LLM_PROVIDER": "gemini"})


def test_openai_configuration_is_unchanged_by_default() -> None:
    configuration = load_models_configuration()

    assert configuration.for_llm_provider(Provider.OPENAI) is configuration
    assert configuration.tasks[TaskName.BRIEF].model == "gpt-6-sol"


def test_anthropic_moves_every_text_task_and_leaves_media_alone() -> None:
    configuration = load_models_configuration().for_llm_provider(Provider.ANTHROPIC)

    for name, task in configuration.tasks.items():
        if name in {TaskName.TRANSCRIPTION, TaskName.SPEECH}:
            assert task.provider is Provider.DEEPGRAM
        else:
            assert task.provider is Provider.ANTHROPIC, name
            assert task.model.startswith("claude-"), name
            assert task.max_tokens > 0
    assert configuration.tasks[TaskName.BRIEF].model == "claude-sonnet-5-5"
    assert configuration.tasks[TaskName.SIMULATION_ACTOR].model == "claude-haiku-5-5"


def test_anthropic_cost_limits_cover_a_full_answer() -> None:
    configuration = load_models_configuration().for_llm_provider(Provider.ANTHROPIC)

    for name, task in configuration.tasks.items():
        if task.provider is not Provider.ANTHROPIC:
            continue
        pricing = configuration.models[task.model].pricing
        worst_output = Decimal(task.max_tokens) * pricing.output_usd / Decimal(1_000_000)
        assert worst_output < task.request_cost_limit_usd, name


def test_anthropic_overrides_must_cover_every_text_task() -> None:
    raw = load_models_configuration().model_dump(mode="json")
    raw["llm_providers"]["anthropic"].pop("brief")

    with pytest.raises(ValueError, match="cover exactly"):
        ModelsConfiguration.model_validate(raw)


def test_overrides_must_reference_models_of_their_provider() -> None:
    raw = load_models_configuration().model_dump(mode="json")
    raw["llm_providers"]["anthropic"]["brief"]["model"] = "gpt-6-sol"

    with pytest.raises(ValueError, match="not a anthropic model"):
        ModelsConfiguration.model_validate(raw)


def test_missing_provider_section_is_a_configuration_error() -> None:
    raw = load_models_configuration().model_dump(mode="json")
    raw["llm_providers"] = {}

    with pytest.raises(ModelConfigurationError, match="llm_providers.anthropic"):
        ModelsConfiguration.model_validate(raw).for_llm_provider(Provider.ANTHROPIC)


def test_live_gateway_builds_only_the_selected_text_provider() -> None:
    built: list[Provider] = []

    def factory(provider: Provider):
        def build():
            built.append(provider)
            return SimpleNamespace()

        return build

    configuration = load_models_configuration().for_llm_provider(Provider.ANTHROPIC)
    create_gateway(
        settings("live", "anthropic"),
        configuration,
        provider_factories={
            Provider.OPENAI: factory(Provider.OPENAI),
            Provider.ANTHROPIC: factory(Provider.ANTHROPIC),
        },
    )

    assert built == [Provider.ANTHROPIC]


def test_anthropic_in_replay_has_no_cassettes_yet() -> None:
    configuration = load_models_configuration().for_llm_provider(Provider.ANTHROPIC)
    gateway = create_gateway(settings("replay", "anthropic"), configuration)

    with pytest.raises(GatewayReplayError):
        asyncio.run(
            gateway.execute(
                GatewayRequest(
                    task=TaskName.BRIEF,
                    prompt="Return JSON.",
                    payload={"candidateId": "candidate-a"},
                    output_schema=Answer,
                )
            )
        )


def test_adapter_sends_structured_output_and_reads_the_text_block() -> None:
    client, calls = fake_client(message())

    result = asyncio.run(AnthropicProvider(client).generate(provider_request()))

    assert result.content == '{"value":"ok","score":3}'
    assert (result.input_tokens, result.output_tokens) == (11, 4)
    assert result.request_id == "synthetic-anthropic-id"
    call = calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["system"] == "Return JSON."
    assert call["max_tokens"] == 123
    assert call["messages"] == [{"role": "user", "content": '{"candidateId":"candidate-a"}'}]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["thinking"] == {"type": "between_tools"}


def test_haiku_runs_with_thinking_disabled() -> None:
    client, calls = fake_client(message())

    asyncio.run(AnthropicProvider(client).generate(provider_request("claude-haiku-5-5")))

    assert calls[0]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_incomplete_or_refused_answers_are_provider_errors(stop_reason: str) -> None:
    client, _ = fake_client(message(stop_reason=stop_reason))

    with pytest.raises(GatewayProviderError, match="incomplete"):
        asyncio.run(AnthropicProvider(client).generate(provider_request()))


def test_adapter_hides_provider_error_details() -> None:
    client, _ = fake_client(error=RuntimeError("secret candidate content"))

    with pytest.raises(GatewayProviderError, match="Anthropic request failed") as caught:
        asyncio.run(AnthropicProvider(client).generate(provider_request()))

    assert "secret candidate content" not in str(caught.value)


def test_wire_schema_drops_keywords_claude_rejects() -> None:
    schema = _wire_schema(Answer.model_json_schema())

    assert schema["additionalProperties"] is False
    assert "minLength" not in schema["properties"]["value"]
    assert "maxLength" not in schema["properties"]["value"]
    assert "minimum" not in schema["properties"]["score"]
    assert "maximum" not in schema["properties"]["score"]


def test_every_judge_object_forbids_extra_properties_on_the_wire() -> None:
    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from objects(value)

    for node in objects(_wire_schema(JudgeOutput.model_json_schema())):
        assert node["additionalProperties"] is False
