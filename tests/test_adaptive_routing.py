from dataclasses import FrozenInstanceError

import pytest

from tokrix.decision import Action
from tokrix.feedback import FeedbackRecord
from tokrix.intelligence.classifier import TaskType
from tokrix.routing import (
    AdaptiveRoutingResult,
    ModelTier,
    RoutingDecision,
    recommend_tier,
)


def _record(
    tier: ModelTier,
    passed: bool,
    task_type: TaskType = TaskType.CODING,
) -> FeedbackRecord:
    return FeedbackRecord(
        attempt_number=1,
        task_type=task_type,
        model_tier=tier,
        verification_passed=passed,
        action=Action.ACCEPT if passed else Action.RETRY,
        execution_latency_ms=1.0,
        pipeline_accepted=passed,
        pipeline_stopped=not passed,
    )


def _routing(tier: ModelTier = ModelTier.LOW) -> RoutingDecision:
    return RoutingDecision(tier=tier, budget_score=20, reason="static")


def _evidence(
    current: ModelTier,
    current_passes: int,
    current_fails: int,
    higher: ModelTier | None,
    higher_passes: int = 0,
    higher_fails: int = 0,
    other_task_records: int = 0,
) -> list[FeedbackRecord]:
    records = [
        *[_record(current, True) for _ in range(current_passes)],
        *[_record(current, False) for _ in range(current_fails)],
    ]
    if higher is not None:
        records.extend(_record(higher, True) for _ in range(higher_passes))
        records.extend(_record(higher, False) for _ in range(higher_fails))
    records.extend(
        _record(ModelTier.HIGH, True, TaskType.GENERAL)
        for _ in range(other_task_records)
    )
    return records


def test_no_feedback_retains_static_tier() -> None:
    result = recommend_tier(_routing(), TaskType.CODING, [])

    assert result.original_tier is ModelTier.LOW
    assert result.recommended_tier is ModelTier.LOW
    assert result.adjusted is False
    assert result.current_tier_samples == 0
    assert result.current_tier_success_rate == 0.0
    assert result.higher_tier_samples == 0
    assert result.higher_tier_success_rate == 0.0


def test_insufficient_current_tier_samples_retain_static_tier() -> None:
    records = _evidence(ModelTier.LOW, 2, 2, ModelTier.MEDIUM, 5)

    result = recommend_tier(_routing(), TaskType.CODING, records)

    assert result.adjusted is False
    assert result.current_tier_samples == 4
    assert "5 required" in result.reason


def test_insufficient_higher_tier_samples_retain_static_tier() -> None:
    records = _evidence(ModelTier.LOW, 2, 3, ModelTier.MEDIUM, 4)

    result = recommend_tier(_routing(), TaskType.CODING, records)

    assert result.adjusted is False
    assert result.higher_tier_samples == 4
    assert "5 required" in result.reason


def test_other_task_type_feedback_does_not_influence_recommendation() -> None:
    records = _evidence(
        ModelTier.LOW,
        2,
        3,
        ModelTier.MEDIUM,
        4,
        1,
        other_task_records=10,
    )

    result = recommend_tier(_routing(), TaskType.CODING, records)

    assert result.adjusted is True
    assert result.current_tier_samples == 5
    assert result.higher_tier_samples == 5
    assert result.higher_tier_success_rate == pytest.approx(0.8)


def test_successful_evidence_recommends_exactly_one_tier_upgrade() -> None:
    records = _evidence(ModelTier.LOW, 1, 4, ModelTier.MEDIUM, 5)

    result = recommend_tier(_routing(), TaskType.CODING, records)

    assert isinstance(result, AdaptiveRoutingResult)
    assert result.original_tier is ModelTier.LOW
    assert result.recommended_tier is ModelTier.MEDIUM
    assert result.adjusted is True
    assert result.current_tier_success_rate == pytest.approx(0.2)
    assert result.higher_tier_success_rate == pytest.approx(1.0)
    assert "Recommend MEDIUM" in result.reason


def test_high_tier_never_recommends_upgrade() -> None:
    records = [_record(ModelTier.HIGH, False) for _ in range(10)]

    result = recommend_tier(_routing(ModelTier.HIGH), TaskType.CODING, records)

    assert result.adjusted is False
    assert result.recommended_tier is ModelTier.HIGH
    assert result.higher_tier_samples is None
    assert result.higher_tier_success_rate is None


def test_current_tier_performs_well_so_policy_retains_it() -> None:
    records = _evidence(ModelTier.LOW, 4, 1, ModelTier.MEDIUM, 5)

    result = recommend_tier(_routing(), TaskType.CODING, records)

    assert result.current_tier_success_rate == pytest.approx(0.8)
    assert result.adjusted is False


@pytest.mark.parametrize(
    ("current_passes", "higher_passes", "higher_fails", "expected_adjusted"),
    [
        (3, 4, 1, True),  # exactly 60%, 80%, and 20 points better
        (3, 3, 2, False),  # higher rate below 80%
        (4, 5, 0, False),  # current tier above 60%
        (2, 3, 2, False),  # only 20 points higher, but below 80%
        (2, 4, 1, True),  # 40% to 80%, exactly 40 points higher
    ],
)
def test_threshold_boundaries(
    current_passes: int,
    higher_passes: int,
    higher_fails: int,
    expected_adjusted: bool,
) -> None:
    records = _evidence(
        ModelTier.LOW,
        current_passes,
        5 - current_passes,
        ModelTier.MEDIUM,
        higher_passes,
        higher_fails,
    )

    result = recommend_tier(_routing(), TaskType.CODING, records)

    assert result.adjusted is expected_adjusted


def test_medium_upgrade_is_only_to_high() -> None:
    records = _evidence(ModelTier.MEDIUM, 1, 4, ModelTier.HIGH, 5)

    result = recommend_tier(
        _routing(ModelTier.MEDIUM),
        TaskType.CODING,
        records,
    )

    assert result.recommended_tier is ModelTier.HIGH


@pytest.mark.parametrize("routing", [None, object()])
def test_invalid_routing_decision_is_rejected(routing: object) -> None:
    with pytest.raises(TypeError, match="RoutingDecision"):
        recommend_tier(routing, TaskType.CODING, [])  # type: ignore[arg-type]


def test_invalid_tier_inside_routing_decision_is_rejected() -> None:
    routing = RoutingDecision(tier="low", budget_score=20, reason="invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="tier must be a ModelTier"):
        recommend_tier(routing, TaskType.CODING, [])


def test_invalid_task_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="task_type must be a TaskType"):
        recommend_tier(_routing(), "coding", [])  # type: ignore[arg-type]


@pytest.mark.parametrize("feedback", [None, "not feedback", [object()]])
def test_invalid_feedback_is_rejected(feedback: object) -> None:
    with pytest.raises(TypeError):
        recommend_tier(
            _routing(),
            TaskType.CODING,
            feedback,  # type: ignore[arg-type]
        )


def test_result_is_immutable() -> None:
    result = recommend_tier(_routing(), TaskType.CODING, [])

    with pytest.raises(FrozenInstanceError):
        result.adjusted = True  # type: ignore[misc]
