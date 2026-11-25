"""Provider-independent prompt execution orchestration."""

import math
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
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_reported_cost: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TypeError(f"{field_name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        cost = self.provider_reported_cost
        if cost is not None:
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                raise TypeError("provider_reported_cost must be a number or None")
            if (isinstance(cost, float) and not math.isfinite(cost)) or cost < 0:
                raise ValueError(
                    "provider_reported_cost must be finite and non-negative"
                )


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