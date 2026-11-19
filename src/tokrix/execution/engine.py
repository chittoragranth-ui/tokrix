"""Provider-independent prompt execution orchestration."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ExecutionRequest:
    """The prompt submitted for execution."""

    prompt: str


@dataclass
class ExecutionResponse:
    """Generated text returned by a provider."""

    content: str


class ModelProvider(Protocol):
    """Contract implemented by synchronous model providers."""

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        """Generate a response for the given request."""
        ...


class ExecutionEngine:
    """Validate and execute requests through a model provider."""

    def __init__(self, provider: ModelProvider) -> None:
        if not callable(getattr(provider, "execute", None)):
            raise TypeError("provider must implement a callable execute method")
        self._provider = provider

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        """Execute a request and validate the provider's response.

        Provider exceptions are intentionally allowed to propagate unchanged.
        """
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        if not isinstance(request.prompt, str):
            raise TypeError("request.prompt must be a string")
        if not request.prompt.strip():
            raise ValueError("request.prompt must not be empty or whitespace-only")

        response = self._provider.execute(request)
        if not isinstance(response, ExecutionResponse):
            raise TypeError("provider must return an ExecutionResponse")
        if not isinstance(response.content, str):
            raise TypeError("response.content must be a string")
        return response