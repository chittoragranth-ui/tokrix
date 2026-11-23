from dataclasses import FrozenInstanceError

import pytest

from tokrix.intelligence.classifier import TaskType
from tokrix.routing import (
    ModelTier,
    RoutingBenchmarkCaseResult,
    RoutingBenchmarkResult,
    RoutingStrategyMetrics,
    compare_routing_results,
    evaluate_routing,
    get_benchmark_dataset,
)

_TIER_COST = {
    ModelTier.LOW: 1,
    ModelTier.MEDIUM: 2,
    ModelTier.HIGH: 3,
}


def _result(
    cases: tuple[tuple[ModelTier, ModelTier, bool, bool], ...],
) -> RoutingBenchmarkResult:
    case_results = tuple(
        RoutingBenchmarkCaseResult(
            task_type=TaskType.GENERAL,
            static_tier=static_tier,
            adaptive_tier=adaptive_tier,
            static_verification_passed=static_passed,
            adaptive_verification_passed=adaptive_passed,
        )
        for static_tier, adaptive_tier, static_passed, adaptive_passed in cases
    )
    case_count = len(case_results)
    static_successes = sum(case.static_verification_passed for case in case_results)
    adaptive_successes = sum(
        case.adaptive_verification_passed for case in case_results
    )
    static_distribution = {tier: 0 for tier in ModelTier}
    adaptive_distribution = {tier: 0 for tier in ModelTier}
    for case in case_results:
        static_distribution[case.static_tier] += 1
        adaptive_distribution[case.adaptive_tier] += 1

    return RoutingBenchmarkResult(
        case_count=case_count,
        static_metrics=RoutingStrategyMetrics(
            verification_successes=static_successes,
            success_rate=static_successes / case_count if case_count else 0.0,
            average_tier_cost_proxy=(
                sum(_TIER_COST[case.static_tier] for case in case_results)
                / case_count
                if case_count
                else 0.0
            ),
            tier_distribution=static_distribution,
        ),
        adaptive_metrics=RoutingStrategyMetrics(
            verification_successes=adaptive_successes,
            success_rate=adaptive_successes / case_count if case_count else 0.0,
            average_tier_cost_proxy=(
                sum(_TIER_COST[case.adaptive_tier] for case in case_results)
                / case_count
                if case_count
                else 0.0
            ),
            tier_distribution=adaptive_distribution,
        ),
        adaptive_upgrade_count=0,
        adaptive_upgrade_rate=0.0,
        case_results=case_results,
    )


@pytest.mark.parametrize(
    ("outcomes", "expected_difference"),
    [
        (((False, True),), 100.0),
        (((True, False),), -100.0),
        (((True, True), (False, False)), 0.0),
    ],
)
def test_success_rate_difference_is_adaptive_minus_static(
    outcomes: tuple[tuple[bool, bool], ...],
    expected_difference: float,
) -> None:
    result = _result(
        tuple(
            (ModelTier.LOW, ModelTier.LOW, static, adaptive)
            for static, adaptive in outcomes
        )
    )

    comparison = compare_routing_results(result)

    assert comparison.success_rate_difference_percentage_points == expected_difference


def test_adaptive_higher_tier_cost_reports_absolute_relative_and_direction() -> None:
    comparison = compare_routing_results(
        _result(((ModelTier.LOW, ModelTier.MEDIUM, True, True),))
    )

    assert comparison.average_tier_cost_proxy_difference == 1.0
    assert comparison.relative_tier_cost_proxy_change_percent == 100.0
    assert comparison.adaptive_cost_direction == "more_expensive"


def test_adaptive_lower_tier_cost_reports_negative_relative_change() -> None:
    comparison = compare_routing_results(
        _result(((ModelTier.MEDIUM, ModelTier.LOW, True, True),))
    )

    assert comparison.average_tier_cost_proxy_difference == -1.0
    assert comparison.relative_tier_cost_proxy_change_percent == -50.0
    assert comparison.adaptive_cost_direction == "less_expensive"


def test_equal_tier_cost_is_reported_as_unchanged() -> None:
    comparison = compare_routing_results(
        _result(((ModelTier.LOW, ModelTier.LOW, True, True),))
    )

    assert comparison.average_tier_cost_proxy_difference == 0.0
    assert comparison.relative_tier_cost_proxy_change_percent == 0.0
    assert comparison.adaptive_cost_direction == "same_cost"


def test_empty_result_has_no_relative_percentages_or_cost_comparison() -> None:
    comparison = compare_routing_results(_result(()))

    assert comparison.case_count == 0
    assert comparison.success_rate_difference_percentage_points == 0.0
    assert comparison.average_tier_cost_proxy_difference == 0.0
    assert comparison.relative_tier_cost_proxy_change_percent is None
    assert comparison.adaptive_cost_direction == "not_comparable"
    assert comparison.tier_distribution_differences == {
        tier: 0 for tier in ModelTier
    }
    assert comparison.outcome_comparison.improved_percentage is None
    assert comparison.outcome_comparison.degraded_percentage is None
    assert comparison.outcome_comparison.unchanged_percentage is None


def test_case_outcome_counts_and_percentages() -> None:
    comparison = compare_routing_results(
        _result(
            (
                (ModelTier.LOW, ModelTier.MEDIUM, False, True),
                (ModelTier.MEDIUM, ModelTier.HIGH, True, False),
                (ModelTier.HIGH, ModelTier.HIGH, True, True),
                (ModelTier.LOW, ModelTier.LOW, False, False),
            )
        )
    )

    outcomes = comparison.outcome_comparison
    assert outcomes.improved_count == 1
    assert outcomes.improved_percentage == 25.0
    assert outcomes.degraded_count == 1
    assert outcomes.degraded_percentage == 25.0
    assert outcomes.unchanged_count == 2
    assert outcomes.unchanged_percentage == 50.0


def test_tier_distribution_difference_uses_adaptive_minus_static() -> None:
    comparison = compare_routing_results(
        _result(
            (
                (ModelTier.LOW, ModelTier.MEDIUM, True, True),
                (ModelTier.LOW, ModelTier.HIGH, True, True),
                (ModelTier.MEDIUM, ModelTier.HIGH, True, True),
            )
        )
    )

    assert comparison.tier_distribution_differences == {
        ModelTier.LOW: -2,
        ModelTier.MEDIUM: 0,
        ModelTier.HIGH: 2,
    }


def test_comparison_result_and_distribution_are_immutable() -> None:
    comparison = compare_routing_results(
        _result(((ModelTier.LOW, ModelTier.MEDIUM, False, True),))
    )

    with pytest.raises(FrozenInstanceError):
        comparison.case_count = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        comparison.tier_distribution_differences[ModelTier.LOW] = 4  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        comparison.outcome_comparison.improved_count = 0  # type: ignore[misc]


def test_comparison_uses_existing_evaluator_and_dataset_results() -> None:
    result = evaluate_routing(
        entry.benchmark_case for entry in get_benchmark_dataset()
    )

    comparison = compare_routing_results(result)

    assert comparison.case_count == result.case_count == 15
    assert comparison.outcome_comparison.improved_count > 0
    assert comparison.outcome_comparison.degraded_count > 0
    assert sum(comparison.tier_distribution_differences.values()) == 0


def test_non_result_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="RoutingBenchmarkResult"):
        compare_routing_results(None)  # type: ignore[arg-type]
