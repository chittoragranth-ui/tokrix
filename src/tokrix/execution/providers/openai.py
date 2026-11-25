"""OpenAI Responses API provider."""

from typing import Protocol

from tokrix.execution.engine import ExecutionRequest, ExecutionResponse


class _ResponsesAPI(Protocol):
    def create(self, *, model: str, input: str) -> object:
        """Create a response through the OpenAI Responses API."""
        ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


class OpenAIProvider:
    """Execute Tokrix requests with the official OpenAI Python SDK."""

    def __init__(self, model: str, client: _OpenAIClient | None = None) -> None:
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        if not model.strip():
            raise ValueError("model must not be empty or whitespace-only")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The OpenAI provider requires the optional SDK dependency. "
                    "Install it with `pip install tokrix[openai]`."
                ) from exc
            client = OpenAI()

        responses = getattr(client, "responses", None)
        if not callable(getattr(responses, "create", None)):
            raise TypeError("client must provide a callable responses.create method")

        self._model = model
        self._client = client

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        """Submit a prompt and return the generated text.

        SDK/API errors are intentionally allowed to propagate unchanged.
        """
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        if not isinstance(request.prompt, str):
            raise TypeError("request.prompt must be a string")
        if not request.prompt.strip():
            raise ValueError("request.prompt must not be empty or whitespace-only")

        response = self._client.responses.create(
            model=self._model,
            input=request.prompt,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise TypeError("OpenAI response output_text must be a string")

        usage = getattr(response, "usage", None)
        return ExecutionResponse(
            content=output_text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            provider_reported_cost=getattr(response, "cost", None),
        )