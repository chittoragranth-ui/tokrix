import math
from dataclasses import fields

import pytest

from tokrix.budget.adaptive import BudgetMode
from tokrix.decision import Action
from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.feedback import FeedbackCollector, FeedbackRecord
from tokrix.intelligence.classifier import TaskType
from tokrix.pipeline import PipelineResult, TokrixPipeline
from tokrix.routing import ModelTier


class SequenceProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ExecutionResponse(content=content)


def _result(
    responses: list[str],
    *,
    mode: BudgetMode = BudgetMode.COST_OPTIMIZED,
    max_retries: int = 1,
) -> PipelineResult:
    providers = {
        ModelTier.LOW: SequenceProvider(responses),
        ModelTier.MEDIUM: SequenceProvider(responses),
        ModelTier.HIGH: SequenceProvider(responses),
    }
    return TokrixPipeline(providers).run(
        "What is photosynthesis?",
        mode=mode,
        max_retries=max_retries,
    )


def test_empty_collector_snapshot_is_empty_tuple() -> None:
    assert FeedbackCollector().snapshot() == ()


def test_single_attempt_feedback_maps_result_fields() -> None:
    result = _result(["A useful answer."])
    collector = FeedbackCollector()

    collector.record(result)
    records = collector.snapshot()

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, FeedbackRecord)
    assert record.attempt_number == 1
    assert record.task_type is TaskType.QUESTION_ANSWERING
    assert record.model_tier is ModelTier.LOW
    assert record.verification_passed is True
    assert record.action is Action.ACCEPT
    assert record.execution_latency_ms == result.attempt_history[0].execution_latency_ms
    assert record.pipeline_accepted is True
    assert record.pipeline_stopped is False


def test_multiple_attempts_preserve_order_and_retry_feedback() -> None:
    result = _result([" ", "answer"])
    collector = FeedbackCollector()

    collector.record(result)
    records = collector.snapshot()

    assert [record.attempt_number for record in records] == [1, 2]
    assert [record.action for record in records] == [Action.RETRY, Action.ACCEPT]
    assert [record.verification_passed for record in records] == [False, True]
    assert all(record.pipeline_accepted for record in records)


def test_escalation_feedback_records_each_tier() -> None:
    low = SequenceProvider([" ", " "])
    medium = SequenceProvider(["answer"])
    pipeline = TokrixPipeline(
        {ModelTier.LOW: low, ModelTier.MEDIUM: medium}
    )
    result = pipeline.run("Hello.", BudgetMode.COST_OPTIMIZED)

    collector = FeedbackCollector()
    collector.record(result)
    feedback = collector.snapshot()

    assert [record.model_tier for record in feedback] == [
        ModelTier.LOW,
        ModelTier.LOW,
        ModelTier.MEDIUM,
    ]
    assert [record.action for record in feedback] == [
        Action.RETRY,
        Action.ESCALATE,
        Action.ACCEPT,
    ]
    assert all(record.pipeline_accepted for record in feedback)


def test_final_stopped_outcome_is_reflected_in_each_attempt() -> None:
    result = _result([" "], max_retries=0)

    collector = FeedbackCollector()
    collector.record(result)
    feedback = collector.snapshot()

    assert result.final_decision.action is Action.STOP
    assert len(feedback) == 3
    assert all(record.pipeline_accepted is False for record in feedback)
    assert all(record.pipeline_stopped is True for record in feedback)


def test_multiple_results_and_duplicate_record_calls_are_observations() -> None:
    successful = _result(["answer"])
    failed = _result([" "], max_retries=0)
    collector = FeedbackCollector()

    collector.record(successful)
    collector.record(failed)
    collector.record(successful)

    snapshot = collector.snapshot()

    assert len(snapshot) == 5
    assert snapshot[0] == snapshot[4]
    assert snapshot[0].pipeline_accepted is True
    assert all(record.pipeline_stopped for record in snapshot[1:4])


def test_result_without_attempt_history_adds_no_records() -> None:
    result = _result(["answer"])
    result.attempt_history = None

    collector = FeedbackCollector()
    collector.record(result)

    assert collector.snapshot() == ()


def test_invalid_pipeline_result_is_rejected() -> None:
    with pytest.raises(TypeError, match="PipelineResult"):
        FeedbackCollector().record(object())  # type: ignore[arg-type]


def test_invalid_nested_attempt_is_rejected_without_partial_recording() -> None:
    result = _result(["answer"])
    result.attempt_history.append(object())  # type: ignore[arg-type]
    collector = FeedbackCollector()

    with pytest.raises(TypeError, match="PipelineAttempt"):
        collector.record(result)  # type: ignore[arg-type]

    assert collector.snapshot() == ()


@pytest.mark.parametrize(
    "latency",
    [-1.0, math.inf, -math.inf, math.nan, True, "2"],
)
def test_invalid_latency_is_rejected(latency: object) -> None:
    result = _result(["answer"])
    result.attempt_history[0].execution_latency_ms = latency  # type: ignore[assignment]

    with pytest.raises(ValueError, match="execution_latency_ms"):
        FeedbackCollector().record(result)


def test_feedback_does_not_store_prompt_or_response_content() -> None:
    prompt = "What is photosynthesis?"
    response_text = "A distinctive generated answer."
    result = TokrixPipeline(
        {ModelTier.LOW: SequenceProvider([response_text])}
    ).run(prompt, BudgetMode.COST_OPTIMIZED)

    collector = FeedbackCollector()
    collector.record(result)
    feedback = collector.snapshot()[0]

    field_names = {field.name for field in fields(FeedbackRecord)}
    assert "prompt" not in field_names
    assert "response" not in field_names
    assert "content" not in field_names
    assert prompt not in repr(feedback)
    assert response_text not in repr(feedback)


def test_snapshot_is_immutable_and_records_are_frozen() -> None:
    collector = FeedbackCollector()
    collector.record(_result(["answer"]))

    with pytest.raises(AttributeError):
        collector.snapshot()[0].action = Action.STOP  # type: ignore[misc]
