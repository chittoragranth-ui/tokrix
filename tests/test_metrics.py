import pytest

from tokrix.budget.adaptive import BudgetMode
from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.intelligence.classifier import TaskType
from tokrix.pipeline import PipelineResult, TokrixPipeline
from tokrix.routing import ModelTier
from tokrix.telemetry import MetricsAggregator, MetricsSnapshot


class SequenceProvider:
    def __init__(self, content: str) -> None:
        self.content = content

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        return ExecutionResponse(content=self.content)


def _pipeline_result(
    content: str,
    mode: BudgetMode = BudgetMode.COST_OPTIMIZED,
    max_retries: int = 0,
    duration_ms: float = 10.0,
) -> PipelineResult:
    result = TokrixPipeline(
        {
            ModelTier.LOW: SequenceProvider(content),
            ModelTier.MEDIUM: SequenceProvider(content),
            ModelTier.HIGH: SequenceProvider(content),
        }
    ).run("What is photosynthesis?", mode, max_retries)
    result.total_duration_ms = duration_ms
    return result


def test_empty_snapshot_has_zero_counts_and_averages() -> None:
    snapshot = MetricsAggregator().snapshot()

    assert isinstance(snapshot, MetricsSnapshot)
    assert snapshot.total_runs == 0
    assert snapshot.accepted_runs == 0
    assert snapshot.stopped_runs == 0
    assert snapshot.average_pipeline_duration_ms == 0.0
    assert snapshot.total_retries == 0
    assert snapshot.total_escalations == 0
    assert snapshot.total_stopped_decisions == 0
    assert all(metrics.total_attempts == 0 for metrics in snapshot.tier_metrics.values())
    assert all(metrics.total_runs == 0 for metrics in snapshot.task_type_metrics.values())


def test_single_successful_run_is_aggregated() -> None:
    result = _pipeline_result("A valid response.", duration_ms=12.0)
    aggregator = MetricsAggregator()

    aggregator.record(result)
    snapshot = aggregator.snapshot()

    assert snapshot.total_runs == 1
    assert snapshot.accepted_runs == 1
    assert snapshot.stopped_runs == 0
    assert snapshot.average_pipeline_duration_ms == 12.0
    low = snapshot.tier_metrics[ModelTier.LOW]
    assert low.total_attempts == 1
    assert low.successful_attempts == 1
    assert low.failed_attempts == 0
    assert low.success_rate == 1.0


def test_multiple_runs_average_duration_and_count_outcomes() -> None:
    aggregator = MetricsAggregator()
    aggregator.record(_pipeline_result("good", duration_ms=10.0))
    aggregator.record(_pipeline_result(" ", duration_ms=30.0))

    snapshot = aggregator.snapshot()

    assert snapshot.total_runs == 2
    assert snapshot.accepted_runs == 1
    assert snapshot.stopped_runs == 1
    assert snapshot.average_pipeline_duration_ms == 20.0


def test_per_tier_attempt_success_failure_and_latency_metrics() -> None:
    failed_result = _pipeline_result(" ", duration_ms=10.0)
    successful_result = _pipeline_result("good", duration_ms=20.0)
    aggregator = MetricsAggregator()
    aggregator.record(failed_result)
    aggregator.record(successful_result)

    low = aggregator.snapshot().tier_metrics[ModelTier.LOW]

    assert low.total_attempts == 2
    assert low.successful_attempts == 1
    assert low.failed_attempts == 1
    assert low.success_rate == 0.5
    expected_latency = (
        failed_result.attempt_history[0].execution_latency_ms
        + successful_result.attempt_history[0].execution_latency_ms
    ) / 2
    assert low.average_execution_latency_ms == expected_latency


def test_retries_and_escalations_are_counted_once_per_attempt_decision() -> None:
    result = TokrixPipeline(
        {
            ModelTier.LOW: SequenceProvider(" "),
            ModelTier.MEDIUM: SequenceProvider("success"),
            ModelTier.HIGH: SequenceProvider("success"),
        }
    ).run("Hello.", BudgetMode.COST_OPTIMIZED)

    aggregator = MetricsAggregator()
    aggregator.record(result)
    snapshot = aggregator.snapshot()

    assert snapshot.total_retries == 1
    assert snapshot.total_escalations == 1


def test_task_type_run_outcomes_are_aggregated() -> None:
    aggregator = MetricsAggregator()
    aggregator.record(_pipeline_result("good"))
    aggregator.record(_pipeline_result(" "))

    task_metrics = aggregator.snapshot().task_type_metrics[
        TaskType.QUESTION_ANSWERING
    ]

    assert task_metrics.total_runs == 2
    assert task_metrics.accepted_runs == 1
    assert task_metrics.stopped_runs == 1


def test_recording_same_result_twice_counts_two_observations() -> None:
    result = _pipeline_result("good")
    aggregator = MetricsAggregator()

    aggregator.record(result)
    aggregator.record(result)

    assert aggregator.snapshot().total_runs == 2
    assert aggregator.snapshot().tier_metrics[ModelTier.LOW].total_attempts == 2


@pytest.mark.parametrize("invalid", [None, object(), "result"])
def test_invalid_pipeline_result_is_rejected(invalid: object) -> None:
    with pytest.raises(TypeError, match="PipelineResult"):
        MetricsAggregator().record(invalid)  # type: ignore[arg-type]


def test_invalid_telemetry_value_is_rejected() -> None:
    result = _pipeline_result("good")
    result.total_duration_ms = -1

    with pytest.raises(ValueError, match="total_duration_ms"):
        MetricsAggregator().record(result)


def test_no_attempt_history_produces_zero_tier_metrics() -> None:
    result = _pipeline_result("good")
    result.attempt_history = None

    aggregator = MetricsAggregator()
    aggregator.record(result)
    snapshot = aggregator.snapshot()

    assert snapshot.total_runs == 1
    assert all(metrics.total_attempts == 0 for metrics in snapshot.tier_metrics.values())


def test_escalation_to_missing_provider_stop_is_not_double_counted() -> None:
    result = TokrixPipeline({ModelTier.LOW: SequenceProvider(" ")}).run(
        "Hello.",
        BudgetMode.COST_OPTIMIZED,
        max_retries=0,
    )

    aggregator = MetricsAggregator()
    aggregator.record(result)
    snapshot = aggregator.snapshot()

    assert snapshot.total_escalations == 1
    assert snapshot.total_stopped_decisions == 1
