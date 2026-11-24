import math
from dataclasses import FrozenInstanceError

import pytest

from tokrix.decision import Action
from tokrix.feedback import FeedbackRecord
from tokrix.intelligence.classifier import TaskType
from tokrix.routing import (
    ModelTier,
    PerformanceBenchmarkCase,
    RoutingBenchmarkCase,
    RoutingDecision,
    TierPerformance,
    evaluate_routing,
    evaluate_routing_performance,
    get_benchmark_dataset,
    get_performance_benchmark_dataset,
)


def _upgrade_feedback() -> tuple[FeedbackRecord, ...]:
    records = [
        FeedbackRecord(
            attempt_number=index + 1,
            task_type=TaskType.CODING,
            model_tier=ModelTier.LOW,
            verification_passed=index == 0,
            action=Action.ACCEPT if index == 0 else Action.RETRY,
            execution_latency_ms=1.0,
            pipeline_accepted=index == 0,
            pipeline_stopped=False,
        )
        for index in range(5)
    ]
    records.extend(
        FeedbackRecord(
            attempt_number=index + 1,
            task_type=TaskType.CODING,
            model_tier=ModelTier.MEDIUM,
            verification_passed=True,
            action=Action.ACCEPT,
            execution_latency_ms=1.0,
            pipeline_accepted=True,
            pipeline_stopped=False,
        )
        for index in range(5)
    )
    return tuple(records)


def _routing_case(
    tier: ModelTier = ModelTier.LOW,
    task_type: TaskType = TaskType.CODING,
    feedback: tuple[FeedbackRecord, ...] = (),
) -> RoutingBenchmarkCase:
    return RoutingBenchmarkCase(
        task_type=task_type,
        static_decision=RoutingDecision(tier, 20, "test"),
        historical_feedback=feedback,
        expected_outcomes={selected_tier: True for selected_tier in ModelTier},
    )


def _profile_case(
    benchmark_case: RoutingBenchmarkCase,
    profiles: dict[ModelTier, TierPerformance] | None = None,
    scenario_id: str | None = None,
) -> PerformanceBenchmarkCase:
    if profiles is None:
        profiles = {
            ModelTier.LOW: TierPerformance(40, 1, 100),
            ModelTier.MEDIUM: TierPerformance(90, 2.5, 200),
            ModelTier.HIGH: TierPerformance(100, 5, 400),
        }
    return PerformanceBenchmarkCase(
        benchmark_case=benchmark_case,
        tier_performance=profiles,
        scenario_id=scenario_id,
    )


def test_tier_performance_accepts_valid_measurements() -> None:
    profile = TierPerformance(quality_score=85, cost_units=1.25, latency_ms=20)

    assert profile.quality_score == 85
    assert profile.cost_units == 1.25
    assert profile.latency_ms == 20


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("quality_score", -1, ValueError, "between 0 and 100"),
        ("quality_score", 101, ValueError, "between 0 and 100"),
        ("quality_score", True, TypeError, "quality_score must be a number"),
        ("quality_score", float("nan"), ValueError, "must be finite"),
        ("cost_units", 0, ValueError, "cost_units must be positive"),
        ("cost_units", -1, ValueError, "cost_units must be positive"),
        ("cost_units", False, TypeError, "cost_units must be a number"),
        ("cost_units", float("inf"), ValueError, "must be finite"),
        ("latency_ms", 0, ValueError, "latency_ms must be positive"),
        ("latency_ms", -5, ValueError, "latency_ms must be positive"),
        ("latency_ms", True, TypeError, "latency_ms must be a number"),
    ],
)
def test_invalid_tier_performance_values_are_rejected(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    values: dict[str, int | float] = {
        "quality_score": 80,
        "cost_units": 2,
        "latency_ms": 100,
    }
    values[field] = value  # type: ignore[assignment]

    with pytest.raises(error, match=message):
        TierPerformance(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("profiles", "error", "message"),
    [
        ({ModelTier.LOW: TierPerformance(80, 1, 100)}, ValueError, "every ModelTier"),
        (
            {
                **{
                    tier: TierPerformance(80, 1, 100)
                    for tier in ModelTier
                },
                "low": TierPerformance(80, 1, 100),  # type: ignore[dict-item]
            },
            TypeError,
            "keys must be ModelTier",
        ),
        (
            {
                ModelTier.LOW: TierPerformance(80, 1, 100),
                ModelTier.MEDIUM: object(),  # type: ignore[dict-item]
                ModelTier.HIGH: TierPerformance(80, 1, 100),
            },
            TypeError,
            "values must be TierPerformance",
        ),
    ],
)
def test_performance_case_rejects_invalid_tier_profiles(
    profiles: dict[ModelTier, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        PerformanceBenchmarkCase(
            _routing_case(),
            profiles,  # type: ignore[arg-type]
        )


def test_performance_case_defensively_copies_and_freezes_profiles() -> None:
    profiles = {
        tier: TierPerformance(80, 1, 100)
        for tier in ModelTier
    }
    case = PerformanceBenchmarkCase(_routing_case(), profiles)
    profiles[ModelTier.LOW] = TierPerformance(0, 1, 100)

    assert case.tier_performance[ModelTier.LOW].quality_score == 80
    with pytest.raises(TypeError):
        case.tier_performance[ModelTier.LOW] = TierPerformance(0, 1, 100)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        case.tier_performance[ModelTier.LOW].quality_score = 0  # type: ignore[misc]


def test_performance_evaluation_aggregates_exact_static_and_adaptive_values() -> None:
    cases = (
        _profile_case(
            _routing_case(feedback=_upgrade_feedback()),
            scenario_id="upgrade",
        ),
        _profile_case(
            _routing_case(tier=ModelTier.HIGH, task_type=TaskType.GENERAL),
            {
                ModelTier.LOW: TierPerformance(20, 1, 100),
                ModelTier.MEDIUM: TierPerformance(50, 3, 300),
                ModelTier.HIGH: TierPerformance(80, 6, 600),
            },
            scenario_id="retain-high",
        ),
    )

    result = evaluate_routing_performance(cases)

    assert result.case_count == 2
    assert result.static_metrics.average_quality_score == 60.0
    assert result.static_metrics.average_cost_units == 3.5
    assert result.static_metrics.average_latency_ms == 350.0
    assert result.adaptive_metrics.average_quality_score == 85.0
    assert result.adaptive_metrics.average_cost_units == 4.25
    assert result.adaptive_metrics.average_latency_ms == 400.0
    assert result.case_results[0].scenario_id == "upgrade"
    assert result.case_results[0].static_tier is ModelTier.LOW
    assert result.case_results[0].adaptive_tier is ModelTier.MEDIUM
    assert result.case_results[0].static_performance == TierPerformance(40, 1, 100)
    assert result.case_results[0].adaptive_performance == TierPerformance(90, 2.5, 200)
    assert result.case_results[1].static_tier is ModelTier.HIGH
    assert result.case_results[1].adaptive_tier is ModelTier.HIGH


def test_large_integer_cost_and_latency_averages_remain_finite() -> None:
    large_value = 10**300
    profiles = {
        tier: TierPerformance(80, large_value, large_value + 1)
        for tier in ModelTier
    }
    cases = (
        _profile_case(_routing_case(), profiles, "large-a"),
        _profile_case(_routing_case(task_type=TaskType.GENERAL), profiles, "large-b"),
    )

    result = evaluate_routing_performance(cases)

    assert result.static_metrics.average_cost_units == float(large_value)
    assert result.static_metrics.average_latency_ms == float(large_value + 1)
    assert math.isfinite(result.static_metrics.average_cost_units)
    assert math.isfinite(result.static_metrics.average_latency_ms)


def test_large_finite_float_averages_do_not_overflow_naive_sums() -> None:
    cost = 1.7e308
    latency = 1.6e308
    profiles = {
        tier: TierPerformance(80, cost, latency)
        for tier in ModelTier
    }
    cases = (
        _profile_case(_routing_case(), profiles, "large-float-a"),
        _profile_case(
            _routing_case(task_type=TaskType.GENERAL),
            profiles,
            "large-float-b",
        ),
    )

    result = evaluate_routing_performance(cases)

    assert result.static_metrics.average_cost_units == cost
    assert result.static_metrics.average_latency_ms == latency


@pytest.mark.parametrize("field", ["cost_units", "latency_ms"])
def test_unrepresentable_average_raises_clear_overflow_error(field: str) -> None:
    too_large = 10**1000
    values: dict[str, int | float] = {
        "quality_score": 80,
        "cost_units": 1,
        "latency_ms": 1,
    }
    values[field] = too_large
    profiles = {
        tier: TierPerformance(**values)
        for tier in ModelTier
    }
    cases = (
        _profile_case(_routing_case(), profiles, "overflow-a"),
        _profile_case(
            _routing_case(task_type=TaskType.GENERAL),
            profiles,
            "overflow-b",
        ),
    )

    with pytest.raises(
        OverflowError,
        match="average measurement cannot be represented as a finite float",
    ):
        evaluate_routing_performance(cases)


def test_empty_performance_evaluation_has_zero_averages() -> None:
    result = evaluate_routing_performance(())

    assert result.case_count == 0
    assert result.static_metrics.average_quality_score == 0.0
    assert result.static_metrics.average_cost_units == 0.0
    assert result.static_metrics.average_latency_ms == 0.0
    assert result.adaptive_metrics.average_quality_score == 0.0
    assert result.adaptive_metrics.average_cost_units == 0.0
    assert result.adaptive_metrics.average_latency_ms == 0.0
    assert result.case_results == ()


@pytest.mark.parametrize(
    ("cases", "message"),
    [(None, "iterable"), ("cases", "iterable"), ([object()], "entries")],
)
def test_performance_evaluation_rejects_invalid_inputs(
    cases: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        evaluate_routing_performance(cases)  # type: ignore[arg-type]


def test_performance_dataset_is_deterministic_complete_and_compatible() -> None:
    dataset = get_performance_benchmark_dataset()
    assert get_performance_benchmark_dataset() is dataset
    assert len(dataset) == len(get_benchmark_dataset())
    assert tuple(case.scenario_id for case in dataset) == tuple(
        entry.case_id for entry in get_benchmark_dataset()
    )
    assert all(set(case.tier_performance) == set(ModelTier) for case in dataset)

    result = evaluate_routing_performance(dataset)
    legacy_result = evaluate_routing(
        tuple(case.benchmark_case for case in dataset)
    )
    assert result.case_count == legacy_result.case_count == len(dataset)


def test_synthetic_dataset_models_quality_cost_and_latency_tradeoffs() -> None:
    dataset = {
        case.scenario_id: case
        for case in get_performance_benchmark_dataset()
    }
    quality_gain_case = dataset["coding-multi-constraint-generation"]
    low = quality_gain_case.tier_performance[ModelTier.LOW]
    high = quality_gain_case.tier_performance[ModelTier.HIGH]
    assert high.quality_score > low.quality_score
    assert high.cost_units > low.cost_units
    assert high.latency_ms > low.latency_ms

    low_sufficient_case = dataset["general-all-success"]
    low = low_sufficient_case.tier_performance[ModelTier.LOW]
    high = low_sufficient_case.tier_performance[ModelTier.HIGH]
    assert high.quality_score == low.quality_score
    assert high.cost_units > low.cost_units
    assert high.latency_ms > low.latency_ms
