import json
import re
from collections.abc import Generator
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, ValidationError

from src.config import config

T = TypeVar("T", bound=BaseModel)


class LLMProviderError(Exception):
    """Custom exception raised when LLM interactions fail."""



from typing import ClassVar

from langchain_core.outputs import ChatGeneration, ChatResult


class FakeLLM(BaseChatModel):
    canned_responses: ClassVar[dict[str, str]] = {}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_str = str(messages)
        # Attempt to find a canned response, else fallback to a generic JSON object
        for key, value in self.canned_responses.items():
            if key in prompt_str:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=value))]
                )

        # Generic fallback
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="{}"))])

    @property
    def _llm_type(self) -> str:
        return "fake-llm"


def get_llm(temperature: float = 0.7) -> BaseChatModel:
    if config.LLM_PROVIDER.lower() == "fake":
        return FakeLLM()

    # Defaults to HuggingFace
    llm = HuggingFaceEndpoint(
        repo_id=config.HF_LLM_REPO_ID,
        task="text-generation",
        huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
        temperature=temperature,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    return ChatHuggingFace(llm=llm)


def _extract_json(text: str) -> str:
    """Strip markdown fences to extract JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def generate_structured(prompt: str, model_class: type[T], llm: BaseChatModel) -> T:
    """Calls the LLM, extracts JSON, validates with Pydantic, retries up to MAX_REGEN_ROUNDS."""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(config.MAX_REGEN_ROUNDS + 1):
        try:
            response = llm.invoke(messages)
            content = response.content
            json_str = _extract_json(content)

            data = json.loads(json_str)
            return model_class.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == config.MAX_REGEN_ROUNDS:
                raise LLMProviderError(
                    f"Failed to generate structured output after {config.MAX_REGEN_ROUNDS} retries. Error: {e!s}"
                )

            error_msg = f"Validation error on previous attempt: {e!s}\nPlease return valid JSON."
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content if "response" in locals() else "{}",
                }
            )
            messages.append({"role": "user", "content": error_msg})

    raise LLMProviderError("Max retries exceeded.")


def stream_text(prompt: str, llm: BaseChatModel) -> Generator[str, None, None]:
    """Streams text tokens."""
    messages = [{"role": "user", "content": prompt}]
    for chunk in llm.stream(messages):
        yield chunk.content
