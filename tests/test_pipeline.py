import pytest

from tokrix.budget.adaptive import BudgetMode
from tokrix.decision import Action
from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.pipeline import PipelineResult, TokrixPipeline, orchestrator
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


class SequenceProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return ExecutionResponse(content=self.responses[index])


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
    assert result.verification_result is not None
    assert result.verification_result.passed is True
    assert len(result.verification_result.checks) == 4
    assert result.verification_result.issues == []
    assert isinstance(result.total_duration_ms, (int, float))
    assert result.total_duration_ms >= 0
    assert result.attempt_history is not None
    assert len(result.attempt_history) == 1
    assert isinstance(result.attempt_history[0].execution_latency_ms, (int, float))
    assert result.attempt_history[0].execution_latency_ms >= 0


def test_initial_failure_retries_same_tier_then_succeeds() -> None:
    low = SequenceProvider([" ", "low success"])
    pipeline = TokrixPipeline({ModelTier.LOW: low})

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert low.call_count == 2
    assert result.response.content == "low success"
    assert result.final_tier is ModelTier.LOW
    assert result.final_decision is not None
    assert result.final_decision.action is Action.ACCEPT
    attempts = result.attempt_history or []
    assert [attempt.model_tier for attempt in attempts] == [
        ModelTier.LOW,
        ModelTier.LOW,
    ]
    assert all(
        isinstance(attempt.execution_latency_ms, (int, float))
        and attempt.execution_latency_ms >= 0
        for attempt in attempts
    )
    assert result.total_duration_ms >= 0


def test_timing_measures_provider_calls_and_entire_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([1.0, 2.0, 2.003, 3.0, 3.004, 5.0])
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: next(timestamps))
    pipeline = TokrixPipeline({ModelTier.LOW: SequenceProvider([" ", "success"])})

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert result.attempt_history is not None
    assert result.attempt_history[0].execution_latency_ms == pytest.approx(3.0)
    assert result.attempt_history[1].execution_latency_ms == pytest.approx(4.0)
    assert result.total_duration_ms == pytest.approx(4000.0)


def test_same_tier_retries_exhausted_escalates_then_succeeds() -> None:
    low = SequenceProvider([" ", " "])
    medium = SequenceProvider(["medium success"])
    pipeline = TokrixPipeline(
        {ModelTier.LOW: low, ModelTier.MEDIUM: medium}
    )

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert low.call_count == 2
    assert medium.call_count == 1
    assert result.response.content == "medium success"
    assert result.final_tier is ModelTier.MEDIUM
    assert result.final_decision is not None
    assert result.final_decision.action is Action.ACCEPT


def test_low_medium_high_escalation_ends_at_high() -> None:
    low = SequenceProvider([" "])
    medium = SequenceProvider([" "])
    high = SequenceProvider(["high success"])
    pipeline = TokrixPipeline(
        {
            ModelTier.LOW: low,
            ModelTier.MEDIUM: medium,
            ModelTier.HIGH: high,
        }
    )

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert low.call_count == 2
    assert medium.call_count == 2
    assert high.call_count == 1
    assert result.final_tier is ModelTier.HIGH
    assert result.response.content == "high success"
    assert result.verification_result is not None
    assert result.verification_result.passed is True
    assert result.final_decision is not None
    assert result.final_decision.action is Action.ACCEPT
    assert result.attempt_history is not None
    assert [attempt.model_tier for attempt in result.attempt_history] == [
        ModelTier.LOW,
        ModelTier.LOW,
        ModelTier.MEDIUM,
        ModelTier.MEDIUM,
        ModelTier.HIGH,
    ]
    assert all(attempt.execution_latency_ms >= 0 for attempt in result.attempt_history)
    assert result.total_duration_ms >= 0


def test_all_tiers_fail_stops_at_high() -> None:
    low = SequenceProvider([" "])
    medium = SequenceProvider([" "])
    high = SequenceProvider([" "])
    pipeline = TokrixPipeline(
        {
            ModelTier.LOW: low,
            ModelTier.MEDIUM: medium,
            ModelTier.HIGH: high,
        }
    )

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert low.call_count == 2
    assert medium.call_count == 2
    assert high.call_count == 2
    assert result.final_tier is ModelTier.HIGH
    assert result.final_decision is not None
    assert result.final_decision.action is Action.STOP
    assert result.verification_result is not None
    assert result.verification_result.passed is False


def test_max_retries_zero_escalates_without_same_tier_retry() -> None:
    low = SequenceProvider([" "])
    medium = SequenceProvider(["medium success"])
    pipeline = TokrixPipeline(
        {ModelTier.LOW: low, ModelTier.MEDIUM: medium}
    )

    result = pipeline.run(
        "Hello.",
        BudgetMode.COST_OPTIMIZED,
        max_retries=0,
    )

    assert low.call_count == 1
    assert medium.call_count == 1
    assert result.final_tier is ModelTier.MEDIUM


def test_max_retries_two_allows_two_same_tier_retries() -> None:
    low = SequenceProvider([" ", " ", "low success"])
    pipeline = TokrixPipeline({ModelTier.LOW: low})

    result = pipeline.run(
        "Hello.",
        BudgetMode.COST_OPTIMIZED,
        max_retries=2,
    )

    assert low.call_count == 3
    assert result.response.content == "low success"
    assert result.final_decision is not None
    assert result.final_decision.action is Action.ACCEPT


def test_missing_escalation_provider_stops_safely() -> None:
    low = SequenceProvider([" ", " "])
    pipeline = TokrixPipeline({ModelTier.LOW: low})

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert result.final_decision is not None
    assert result.final_decision.action is Action.STOP
    assert "MEDIUM" in result.final_decision.reason
    assert "no MEDIUM provider" in result.final_decision.reason
    assert result.final_tier is ModelTier.LOW


def test_pipeline_includes_failed_verification_for_blank_response() -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider(" \t\n ")})

    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    assert result.verification_result.passed is False
    assert (
        "Response content is empty or whitespace-only."
        in result.verification_result.issues
    )


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


@pytest.mark.parametrize("max_retries", [-1, -10])
def test_negative_max_retries_is_rejected(max_retries: int) -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider()})

    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        pipeline.run("Hello.", BudgetMode.BALANCED, max_retries)


@pytest.mark.parametrize("max_retries", [True, False])
def test_bool_max_retries_is_rejected(max_retries: bool) -> None:
    pipeline = TokrixPipeline({ModelTier.LOW: FakeProvider()})

    with pytest.raises(TypeError, match="max_retries must be an integer"):
        pipeline.run("Hello.", BudgetMode.BALANCED, max_retries)


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
