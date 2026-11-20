import builtins
from types import SimpleNamespace

import pytest

from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.execution.providers import OpenAIProvider


class FakeOpenAIClient:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []
        self.responses = self

    def create(self, *, model: str, input: str) -> object:
        self.calls.append({"model": model, "input": input})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_provider_initializes_with_injected_client() -> None:
    client = FakeOpenAIClient(SimpleNamespace(output_text="Hello"))

    provider = OpenAIProvider(model="test-model", client=client)

    assert isinstance(provider, OpenAIProvider)


@pytest.mark.parametrize("model", ["", " \t\n "])
def test_empty_or_whitespace_model_is_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        OpenAIProvider(model=model, client=FakeOpenAIClient(object()))


@pytest.mark.parametrize("model", [None, 42])
def test_non_string_model_is_rejected(model: object) -> None:
    with pytest.raises(TypeError, match="model must be a string"):
        OpenAIProvider(model=model, client=FakeOpenAIClient(object()))  # type: ignore[arg-type]


def test_responses_api_receives_model_and_prompt() -> None:
    client = FakeOpenAIClient(SimpleNamespace(output_text="Generated text"))
    provider = OpenAIProvider(model="example-model", client=client)

    provider.execute(ExecutionRequest(prompt="Write a short greeting."))

    assert client.calls == [
        {"model": "example-model", "input": "Write a short greeting."}
    ]


def test_sdk_output_text_is_converted_to_execution_response() -> None:
    client = FakeOpenAIClient(SimpleNamespace(output_text="Generated text"))
    provider = OpenAIProvider(model="example-model", client=client)

    result = provider.execute(ExecutionRequest(prompt="Hello"))

    assert result == ExecutionResponse(content="Generated text")


def test_api_exception_propagates_unchanged() -> None:
    error = RuntimeError("API request failed")
    provider = OpenAIProvider(
        model="example-model",
        client=FakeOpenAIClient(error),
    )

    with pytest.raises(RuntimeError, match="API request failed") as raised:
        provider.execute(ExecutionRequest(prompt="Hello"))

    assert raised.value is error


@pytest.mark.parametrize("response", [object(), SimpleNamespace(output_text=None), 123])
def test_invalid_sdk_response_raises_type_error(response: object) -> None:
    provider = OpenAIProvider(
        model="example-model",
        client=FakeOpenAIClient(response),
    )

    with pytest.raises(TypeError, match="output_text must be a string"):
        provider.execute(ExecutionRequest(prompt="Hello"))


def test_injected_client_must_have_responses_create() -> None:
    with pytest.raises(TypeError, match="responses.create"):
        OpenAIProvider(model="example-model", client=object())  # type: ignore[arg-type]


def test_missing_openai_sdk_has_install_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_openai(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai":
            raise ImportError("simulated missing SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_openai)

    with pytest.raises(ImportError, match=r"pip install tokrix\[openai\]"):
        OpenAIProvider(model="example-model")
