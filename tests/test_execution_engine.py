import pytest

from tokrix.execution import (
    ExecutionEngine,
    ExecutionRequest,
    ExecutionResponse,
)


class FakeProvider:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.received_request: ExecutionRequest | None = None

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        self.received_request = request
        if isinstance(self.response, Exception):
            raise self.response
        return self.response  # type: ignore[return-value]


def test_successful_execution_returns_provider_response() -> None:
    response = ExecutionResponse(content="Generated answer.")
    provider = FakeProvider(response)
    engine = ExecutionEngine(provider)
    request = ExecutionRequest(prompt="Give me an answer.")

    result = engine.execute(request)

    assert result is response
    assert result.content == "Generated answer."
    assert provider.received_request is request


def test_correct_prompt_is_passed_to_provider() -> None:
    provider = FakeProvider(ExecutionResponse(content="Done."))
    engine = ExecutionEngine(provider)

    engine.execute(ExecutionRequest(prompt="Summarize this text."))

    assert provider.received_request is not None
    assert provider.received_request.prompt == "Summarize this text."


def test_provider_without_execute_method_is_rejected() -> None:
    with pytest.raises(TypeError, match="callable execute method"):
        ExecutionEngine(object())  # type: ignore[arg-type]


def test_non_callable_provider_execute_is_rejected() -> None:
    class InvalidProvider:
        execute = None

    with pytest.raises(TypeError, match="callable execute method"):
        ExecutionEngine(InvalidProvider())  # type: ignore[arg-type]


def test_non_request_input_is_rejected() -> None:
    engine = ExecutionEngine(FakeProvider(ExecutionResponse(content="Done.")))

    with pytest.raises(TypeError, match="request must be an ExecutionRequest"):
        engine.execute("not a request")  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt", [None, 42])
def test_non_string_prompt_is_rejected(prompt: object) -> None:
    engine = ExecutionEngine(FakeProvider(ExecutionResponse(content="Done.")))
    request = ExecutionRequest(prompt=prompt)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="request.prompt must be a string"):
        engine.execute(request)


@pytest.mark.parametrize("prompt", ["", " \t\n "])
def test_empty_or_whitespace_prompt_is_rejected(prompt: str) -> None:
    engine = ExecutionEngine(FakeProvider(ExecutionResponse(content="Done.")))

    with pytest.raises(ValueError, match="must not be empty"):
        engine.execute(ExecutionRequest(prompt=prompt))


def test_provider_exception_propagates_unchanged() -> None:
    error = RuntimeError("provider unavailable")
    engine = ExecutionEngine(FakeProvider(error))

    with pytest.raises(RuntimeError, match="provider unavailable") as raised:
        engine.execute(ExecutionRequest(prompt="Hello"))

    assert raised.value is error


def test_non_response_provider_result_is_rejected() -> None:
    engine = ExecutionEngine(FakeProvider("not a response"))

    with pytest.raises(TypeError, match="return an ExecutionResponse"):
        engine.execute(ExecutionRequest(prompt="Hello"))


def test_response_with_non_string_content_is_rejected() -> None:
    response = ExecutionResponse(content=42)  # type: ignore[arg-type]
    engine = ExecutionEngine(FakeProvider(response))

    with pytest.raises(TypeError, match="response.content must be a string"):
        engine.execute(ExecutionRequest(prompt="Hello"))
