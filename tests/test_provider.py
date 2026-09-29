import pytest
from pydantic import BaseModel, Field

from src.llm.provider import (
    FakeLLM,
    LLMProviderError,
    _extract_json,
    generate_structured,
)


class DummyModel(BaseModel):
    name: str
    age: int = Field(gt=0)


def test_extract_json():
    # Unfenced
    assert _extract_json('{"a": 1}') == '{"a": 1}'
    # Fenced
    fenced = '```json\n{"a": 2}\n```'
    assert _extract_json(fenced) == '{"a": 2}'
    # Fenced without json
    fenced2 = '```\n{"a": 3}\n```'
    assert _extract_json(fenced2) == '{"a": 3}'


def test_generate_structured_success():
    FakeLLM.canned_responses = {"dummy": '{"name": "Alice", "age": 25}'}
    llm = FakeLLM()

    result = generate_structured("dummy", DummyModel, llm)
    assert result.name == "Alice"
    assert result.age == 25


def test_generate_structured_retry_success():
    class FlakyFakeLLM(FakeLLM):
        attempts: int = 0

        def _generate(self, messages, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                # First attempt: invalid age
                content = '{"name": "Alice", "age": -5}'
            else:
                # Second attempt: valid
                content = '{"name": "Alice", "age": 25}'
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=content))]
            )

    llm = FlakyFakeLLM()
    result = generate_structured("dummy", DummyModel, llm)
    assert result.age == 25


def test_generate_structured_exhaust_retries():
    FakeLLM.canned_responses = {"dummy": '{"name": "Alice", "age": -5}'}
    llm = FakeLLM()
    # Always returns invalid data

    with pytest.raises(LLMProviderError) as exc:
        generate_structured("dummy", DummyModel, llm)

    assert "Failed to generate structured output" in str(exc.value)
