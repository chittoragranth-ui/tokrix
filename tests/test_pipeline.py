import pytest

from tokrix.budget.adaptive import BudgetMode
from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.pipeline import PipelineResult, TokrixPipeline
from tokrix.routing.router import ModelTier


class FakeProvider:
    def __init__(self, content: str = "fake response") -> None:
        self.content = content
        self.received_request: ExecutionRequest | None = None

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        self.received_request = request
        return ExecutionResponse(content=self.content)


class FailingProvider:
    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        raise RuntimeError("provider failed")


def test_successful_end_to_end_pipeline() -> None:
    low = FakeProvider("low result")
    pipeline = TokrixPipeline({ModelTier.LOW: low})

    result = pipeline.run("Hello there.", BudgetMode.COST_OPTIMIZED)

    assert isinstance(result, PipelineResult)
    assert result.response.content == "low result"
    assert result.complexity_result.score >= 0
    assert result.classification_result.task_type.value
    assert result.estimated_tokens > 0
    assert 1 <= result.budget_decision.budget_score <= 100
    assert result.routing_decision.tier is ModelTier.LOW
    assert result.selected_tier is ModelTier.LOW
    assert low.received_request is not None
    assert low.received_request.prompt == "Hello there."


def test_pipeline_uses_real_complexity_classifier_estimator_and_budget() -> None:
    pipeline = TokrixPipeline(
        {ModelTier.HIGH: FakeProvider("high result")}
    )

    prompt = (
        "Implement a Python data API and design its database architecture. "
        "First, inspect the dataset; then analyze and compare the results; "
        "finally, implement and test the service. "
        "Include validation and preserve compatibility. " * 25
    )
    result = pipeline.run(prompt, BudgetMode.QUALITY_OPTIMIZED)

    assert result.complexity_result.score >= 7
    assert result.classification_result.task_type.value
    assert result.estimated_tokens == (len(prompt) + 3) // 4
    assert result.budget_decision.budget_score == 100
    assert result.routing_decision.tier is ModelTier.HIGH


@pytest.mark.parametrize(
    ("mode", "prompt", "tier"),
    [
        (BudgetMode.COST_OPTIMIZED, "Hello.", ModelTier.LOW),
        (
            BudgetMode.BALANCED,
            "Write a Python function to sort a list.",
            ModelTier.MEDIUM,
        ),
        (
            BudgetMode.QUALITY_OPTIMIZED,
            (
                "Implement a Python API and database architecture. "
                "Inspect the data, analyze results, test and document every requirement."
            ),
            ModelTier.HIGH,
        ),
    ],
)
def test_correct_provider_is_selected_for_each_tier(
    mode: BudgetMode, prompt: str, tier: ModelTier
) -> None:
    providers = {
        ModelTier.LOW: FakeProvider("low"),
        ModelTier.MEDIUM: FakeProvider("medium"),
        ModelTier.HIGH: FakeProvider("high"),
    }
    pipeline = TokrixPipeline(providers)

    result = pipeline.run(prompt, mode)

    assert result.selected_tier is tier
    assert result.response.content == tier.value


def test_partial_provider_mapping_supports_configured_tier() -> None:
    low = FakeProvider("only low")

    result = TokrixPipeline({ModelTier.LOW: low}).run(
        "Hello.",
        BudgetMode.COST_OPTIMIZED,
    )

    assert result.response.content == "only low"


def test_missing_provider_for_selected_tier_is_actionable() -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider()})

    with pytest.raises(ValueError, match=r"HIGH.*tier|provider.*HIGH"):
        pipeline.run(
            "Implement a Python API and database architecture. "
            "Inspect the data, analyze results, test and document every requirement.",
            BudgetMode.QUALITY_OPTIMIZED,
        )


@pytest.mark.parametrize("prompt", [None, 42, ["hello"]])
def test_invalid_prompt_type_is_rejected(prompt: object) -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider()})

    with pytest.raises(TypeError, match="prompt must be a string"):
        pipeline.run(prompt)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt", ["", " \t\n "])
def test_blank_prompt_is_rejected(prompt: str) -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider()})

    with pytest.raises(ValueError, match="must not be empty"):
        pipeline.run(prompt)


def test_invalid_mode_is_rejected() -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider()})

    with pytest.raises(TypeError, match="mode must be a BudgetMode"):
        pipeline.run("Hello.", "balanced")  # type: ignore[arg-type]


@pytest.mark.parametrize("providers", [[], "low"])
def test_provider_mapping_must_be_a_mapping(providers: object) -> None:
    with pytest.raises(TypeError, match="mapping"):
        TokrixPipeline(providers)  # type: ignore[arg-type]


def test_provider_mapping_rejects_invalid_tier_key() -> None:
    with pytest.raises(TypeError, match="keys must be ModelTier"):
        TokrixPipeline({"low": FakeProvider()})  # type: ignore[arg-type]


def test_provider_mapping_rejects_invalid_provider() -> None:
    with pytest.raises(TypeError, match="implement execute"):
        TokrixPipeline({ModelTier.LOW: object()})  # type: ignore[arg-type]


def test_provider_execution_error_propagates_unchanged() -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FailingProvider()})

    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)
