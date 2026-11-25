from dataclasses import FrozenInstanceError

import pytest

from tokrix.decision import Action
from tokrix.feedback import FeedbackRecord
from tokrix.intelligence.classifier import TaskType
from tokrix.routing import (
    ModelTier,
    RoutingBenchmarkCase,
    RoutingDecision,
    evaluate_routing,
)


def _feedback(
    tier: ModelTier,
    passed: bool,
    task_type: TaskType,
    attempt_number: int,
) -> FeedbackRecord:
    return FeedbackRecord(
        attempt_number=attempt_number,
        task_type=task_type,
        model_tier=tier,
        verification_passed=passed,
        action=Action.ACCEPT if passed else Action.RETRY,
        execution_latency_ms=1.0,
        pipeline_accepted=passed,
        pipeline_stopped=not passed,
    )


def _upgrade_evidence(task_type: TaskType = TaskType.CODING) -> tuple[FeedbackRecord, ...]:
    return (
        *(
            _feedback(ModelTier.LOW, passed, task_type, index + 1)
            for index, passed in enumerate((True, False, False, False, False))
        ),
        *(
            _feedback(ModelTier.MEDIUM, True, task_type, index + 1)
            for index in range(5)
        ),
    )


def _case(
    tier: ModelTier = ModelTier.LOW,
    task_type: TaskType = TaskType.CODING,
    feedback: tuple[FeedbackRecord, ...] = (),
    outcomes: dict[ModelTier, bool] | None = None,
) -> RoutingBenchmarkCase:
    if outcomes is None:
        outcomes = {
            ModelTier.LOW: True,
            ModelTier.MEDIUM: True,
            ModelTier.HIGH: True,
        }
    return RoutingBenchmarkCase(
        task_type=task_type,
        static_decision=RoutingDecision(tier=tier, budget_score=20, reason="test"),
        historical_feedback=feedback,
        expected_outcomes=outcomes,
    )


def test_empty_benchmark_has_zero_metrics_and_all_tiers_in_distributions() -> None:
    result = evaluate_routing([])

    assert result.case_count == 0
    assert result.static_metrics.verification_successes == 0
    assert result.static_metrics.success_rate == 0.0
    assert result.static_metrics.average_tier_cost_proxy == 0.0
    assert result.adaptive_metrics.verification_successes == 0
    assert result.adaptive_metrics.success_rate == 0.0
    assert result.adaptive_metrics.average_tier_cost_proxy == 0.0
    assert result.adaptive_upgrade_count == 0
    assert result.adaptive_upgrade_rate == 0.0
    assert result.static_metrics.tier_distribution == {
        tier: 0 for tier in ModelTier
    }
    assert result.adaptive_metrics.tier_distribution == {
        tier: 0 for tier in ModelTier
    }


def test_static_and_adaptive_same_tier_are_scored_against_expected_outcome() -> None:
    result = evaluate_routing([_case(outcomes={
        ModelTier.LOW: False,
        ModelTier.MEDIUM: True,
        ModelTier.HIGH: True,
    })])

    assert result.case_results[0].static_tier is ModelTier.LOW
    assert result.case_results[0].adaptive_tier is ModelTier.LOW
    assert result.static_metrics.verification_successes == 0
    assert result.adaptive_metrics.verification_successes == 0
    assert result.adaptive_upgrade_count == 0


def test_adaptive_upgrade_can_improve_controlled_success_and_cost() -> None:
    result = evaluate_routing(
        [
            _case(
                feedback=_upgrade_evidence(),
                outcomes={
                    ModelTier.LOW: False,
                    ModelTier.MEDIUM: True,
                    ModelTier.HIGH: False,
                },
            )
        ]
    )

    assert result.static_metrics.verification_successes == 0
    assert result.static_metrics.success_rate == 0.0
    assert result.adaptive_metrics.verification_successes == 1
    assert result.adaptive_metrics.success_rate == 1.0
    assert result.static_metrics.average_tier_cost_proxy == 1.0
    assert result.adaptive_metrics.average_tier_cost_proxy == 2.0
    assert result.adaptive_upgrade_count == 1
    assert result.adaptive_upgrade_rate == 1.0


def test_adaptive_upgrade_need_not_improve_verification() -> None:
    result = evaluate_routing(
        [
            _case(
                feedback=_upgrade_evidence(),
                outcomes={
                    ModelTier.LOW: False,
                    ModelTier.MEDIUM: False,
                    ModelTier.HIGH: True,
                },
            )
        ]
    )

    assert result.adaptive_upgrade_count == 1
    assert result.static_metrics.verification_successes == 0
    assert result.adaptive_metrics.verification_successes == 0
    assert result.static_metrics.average_tier_cost_proxy == 1.0
    assert result.adaptive_metrics.average_tier_cost_proxy == 2.0


def test_tier_distributions_upgrade_counts_and_rates() -> None:
    cases = [
        _case(feedback=_upgrade_evidence()),
        _case(
            feedback=_upgrade_evidence(TaskType.REASONING),
            task_type=TaskType.REASONING,
        ),
        _case(tier=ModelTier.MEDIUM, task_type=TaskType.GENERAL),
        _case(tier=ModelTier.HIGH, task_type=TaskType.SUMMARIZATION),
    ]

    result = evaluate_routing(cases)

    assert result.case_count == 4
    assert result.static_metrics.tier_distribution == {
        ModelTier.LOW: 2,
        ModelTier.MEDIUM: 1,
        ModelTier.HIGH: 1,
    }
    assert result.adaptive_metrics.tier_distribution == {
        ModelTier.LOW: 0,
        ModelTier.MEDIUM: 3,
        ModelTier.HIGH: 1,
    }
    assert result.adaptive_upgrade_count == 2
    assert result.adaptive_upgrade_rate == 0.5
    assert [case.task_type for case in result.case_results] == [
        TaskType.CODING,
        TaskType.REASONING,
        TaskType.GENERAL,
        TaskType.SUMMARIZATION,
    ]


def test_exact_average_success_rates_and_cost_proxies() -> None:
    cases = [
        _case(
            tier=ModelTier.LOW,
            outcomes={
                ModelTier.LOW: True,
                ModelTier.MEDIUM: False,
                ModelTier.HIGH: True,
            },
        ),
        _case(
            tier=ModelTier.MEDIUM,
            outcomes={
                ModelTier.LOW: False,
                ModelTier.MEDIUM: True,
                ModelTier.HIGH: False,
            },
        ),
        _case(
            tier=ModelTier.HIGH,
            outcomes={
                ModelTier.LOW: False,
                ModelTier.MEDIUM: False,
                ModelTier.HIGH: False,
            },
        ),
    ]

    result = evaluate_routing(cases)

    assert result.static_metrics.verification_successes == 2
    assert result.static_metrics.success_rate == 2 / 3
    assert result.static_metrics.average_tier_cost_proxy == 2.0
    assert result.adaptive_metrics.verification_successes == 2
    assert result.adaptive_metrics.success_rate == 2 / 3
    assert result.adaptive_metrics.average_tier_cost_proxy == 2.0


@pytest.mark.parametrize(
    ("task_type", "feedback", "outcomes", "error", "message"),
    [
        ("coding", (), {tier: True for tier in ModelTier}, TypeError, "task_type"),
        (
            TaskType.CODING,
            (object(),),  # type: ignore[arg-type]
            {tier: True for tier in ModelTier},
            TypeError,
            "FeedbackRecord",
        ),
        (
            TaskType.CODING,
            (),
            {ModelTier.LOW: True},
            ValueError,
            "every ModelTier",
        ),
        (
            TaskType.CODING,
            (),
            {
                ModelTier.LOW: True,
                ModelTier.MEDIUM: 1,  # type: ignore[dict-item]
                ModelTier.HIGH: False,
            },
            TypeError,
            "values must be bools",
        ),
        (
            TaskType.CODING,
            (),
            {"low": True},  # type: ignore[dict-item]
            TypeError,
            "keys must be ModelTier",
        ),
    ],
)
def test_invalid_benchmark_case_fields_are_rejected(
    task_type: object,
    feedback: tuple[FeedbackRecord, ...],
    outcomes: dict[ModelTier, bool],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RoutingBenchmarkCase(
            task_type=task_type,  # type: ignore[arg-type]
            static_decision=RoutingDecision(
                tier=ModelTier.LOW,
                budget_score=20,
                reason="test",
            ),
            historical_feedback=feedback,
            expected_outcomes=outcomes,
        )


def test_evaluator_rejects_invalid_cases_and_static_decisions() -> None:
    with pytest.raises(TypeError, match="iterable"):
        evaluate_routing(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entries"):
        evaluate_routing([object()])  # type: ignore[list-item]

    invalid_decision = RoutingDecision(
        tier=ModelTier.LOW,
        budget_score=True,  # type: ignore[arg-type]
        reason="test",
    )
    case = RoutingBenchmarkCase(
        task_type=TaskType.CODING,
        static_decision=invalid_decision,
        historical_feedback=(),
        expected_outcomes={tier: True for tier in ModelTier},
    )
    with pytest.raises(TypeError, match="budget_score"):
        evaluate_routing([case])


def test_static_decision_cannot_be_mutated_after_case_construction() -> None:
    decision = RoutingDecision(
        tier=ModelTier.LOW,
        budget_score=20,
        reason="test",
    )
    case = RoutingBenchmarkCase(
        task_type=TaskType.CODING,
        static_decision=decision,
        historical_feedback=(),
        expected_outcomes={
            ModelTier.LOW: True,
            ModelTier.MEDIUM: False,
            ModelTier.HIGH: False,
        },
    )
    decision.budget_score = 80

    assert case.static_decision.budget_score == 20
    with pytest.raises(FrozenInstanceError):
        case.static_decision.budget_score = 80  # type: ignore[misc]
    result = evaluate_routing([case])
    assert result.case_results[0].static_tier is ModelTier.LOW


def test_case_nested_state_cannot_be_mutated_through_mapping_or_dict() -> None:
    outcomes = {
        ModelTier.LOW: True,
        ModelTier.MEDIUM: False,
        ModelTier.HIGH: False,
    }
    decision = RoutingDecision(
        tier=ModelTier.LOW,
        budget_score=20,
        reason="test",
    )
    case = RoutingBenchmarkCase(
        task_type=TaskType.CODING,
        static_decision=decision,
        historical_feedback=(),
        expected_outcomes=outcomes,
    )
    outcomes[ModelTier.LOW] = False

    with pytest.raises(TypeError):
        case.expected_outcomes[ModelTier.LOW] = False  # type: ignore[index]

    decision_state = case.static_decision.__dict__
    decision_state["tier"] = ModelTier.HIGH
    decision_state["budget_score"] = 99
    decision_state["reason"] = "mutated"

    assert case.expected_outcomes[ModelTier.LOW] is True
    assert case.static_decision.tier is ModelTier.LOW
    assert case.static_decision.budget_score == 20
    assert case.static_decision.reason == "test"


def test_case_and_results_are_immutable() -> None:
    outcomes = {
        ModelTier.LOW: True,
        ModelTier.MEDIUM: False,
        ModelTier.HIGH: True,
    }
    case = _case(outcomes=outcomes)
    outcomes[ModelTier.LOW] = False
    result = evaluate_routing([case])

    assert case.expected_outcomes[ModelTier.LOW] is True
    with pytest.raises(TypeError):
        case.expected_outcomes[ModelTier.LOW] = False  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        case.task_type = TaskType.GENERAL  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.static_metrics.tier_distribution[ModelTier.LOW] = 8  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.case_count = 2  # type: ignore[misc]


def test_case_outcomes_are_separate_from_historical_feedback() -> None:
    feedback = _upgrade_evidence()
    case = _case(
        feedback=feedback,
        outcomes={
            ModelTier.LOW: True,
            ModelTier.MEDIUM: False,
            ModelTier.HIGH: False,
        },
    )

    result = evaluate_routing([case])

    assert result.case_results[0].adaptive_tier is ModelTier.MEDIUM
    assert result.case_results[0].static_verification_passed is True
    assert result.case_results[0].adaptive_verification_passed is False
