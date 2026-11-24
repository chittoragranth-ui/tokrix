"""Offline quality, cost, and latency evaluation for routing strategies."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from tokrix.routing.adaptive import recommend_tier
from tokrix.routing.benchmark import RoutingBenchmarkCase
from tokrix.routing.router import ModelTier


@dataclass(frozen=True)
class TierPerformance:
    """Synthetic or supplied performance measurements for one model tier."""

    quality_score: int | float
    cost_units: int | float
    latency_ms: int | float

    def __post_init__(self) -> None:
        for field_name in ("quality_score", "cost_units", "latency_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0 <= self.quality_score <= 100:
            raise ValueError("quality_score must be between 0 and 100")
        if self.cost_units <= 0:
            raise ValueError("cost_units must be positive")
        if self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive")


@dataclass(frozen=True)
class PerformanceBenchmarkCase:
    """A routing case with a complete performance profile per model tier."""

    benchmark_case: RoutingBenchmarkCase
    tier_performance: Mapping[ModelTier, TierPerformance]
    scenario_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_case, RoutingBenchmarkCase):
            raise TypeError("benchmark_case must be a RoutingBenchmarkCase")
        if self.scenario_id is not None:
            if not isinstance(self.scenario_id, str):
                raise TypeError("scenario_id must be a string or None")
            if not self.scenario_id.strip():
                raise ValueError("scenario_id must not be empty or whitespace-only")
        if not isinstance(self.tier_performance, Mapping):
            raise TypeError("tier_performance must be a mapping")
        if any(not isinstance(tier, ModelTier) for tier in self.tier_performance):
            raise TypeError("tier_performance keys must be ModelTier values")
        if set(self.tier_performance) != set(ModelTier):
            raise ValueError("tier_performance must contain every ModelTier")
        if any(
            not isinstance(profile, TierPerformance)
            for profile in self.tier_performance.values()
        ):
            raise TypeError("tier_performance values must be TierPerformance values")
        object.__setattr__(
            self,
            "tier_performance",
            MappingProxyType(dict(self.tier_performance)),
        )


@dataclass(frozen=True)
class RoutingPerformanceCaseResult:
    """Per-scenario selected tiers and their corresponding performance."""

    scenario_id: str | None
    static_tier: ModelTier
    adaptive_tier: ModelTier
    static_performance: TierPerformance
    adaptive_performance: TierPerformance


@dataclass(frozen=True)
class RoutingPerformanceMetrics:
    """Average quality, normalized cost, and latency for a routing strategy."""

    average_quality_score: float
    average_cost_units: float
    average_latency_ms: float


@dataclass(frozen=True)
class RoutingPerformanceResult:
    """Immutable aggregate and per-case routing performance comparison."""

    case_count: int
    static_metrics: RoutingPerformanceMetrics
    adaptive_metrics: RoutingPerformanceMetrics
    case_results: tuple[RoutingPerformanceCaseResult, ...]


def _average(values: list[int | float]) -> float:
    if not values:
        return 0.0
    exact_average = sum(Fraction(value) for value in values) / len(values)
    try:
        average = float(exact_average)
    except OverflowError as error:
        raise OverflowError(
            "average measurement cannot be represented as a finite float"
        ) from error
    if not math.isfinite(average):
        raise OverflowError(
            "average measurement cannot be represented as a finite float"
        )
    return average


def evaluate_routing_performance(
    cases: Iterable[PerformanceBenchmarkCase],
) -> RoutingPerformanceResult:
    """Evaluate static and adaptive tiers against shared per-case profiles.

    Raises:
        OverflowError: If an aggregate average cannot be represented as a
            finite float.
    """
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Iterable):
        raise TypeError(
            "cases must be an iterable of PerformanceBenchmarkCase values"
        )
    benchmark_cases = tuple(cases)
    if any(
        not isinstance(case, PerformanceBenchmarkCase) for case in benchmark_cases
    ):
        raise TypeError("cases entries must be PerformanceBenchmarkCase instances")

    static_quality: list[int | float] = []
    static_cost: list[int | float] = []
    static_latency: list[int | float] = []
    adaptive_quality: list[int | float] = []
    adaptive_cost: list[int | float] = []
    adaptive_latency: list[int | float] = []
    case_results: list[RoutingPerformanceCaseResult] = []

    for case in benchmark_cases:
        benchmark_case = case.benchmark_case
        recommendation = recommend_tier(
            benchmark_case.static_decision,
            benchmark_case.task_type,
            benchmark_case.historical_feedback,
        )
        static_tier = benchmark_case.static_decision.tier
        adaptive_tier = recommendation.recommended_tier
        static_profile = case.tier_performance[static_tier]
        adaptive_profile = case.tier_performance[adaptive_tier]

        static_quality.append(static_profile.quality_score)
        static_cost.append(static_profile.cost_units)
        static_latency.append(static_profile.latency_ms)
        adaptive_quality.append(adaptive_profile.quality_score)
        adaptive_cost.append(adaptive_profile.cost_units)
        adaptive_latency.append(adaptive_profile.latency_ms)
        case_results.append(
            RoutingPerformanceCaseResult(
                scenario_id=case.scenario_id,
                static_tier=static_tier,
                adaptive_tier=adaptive_tier,
                static_performance=static_profile,
                adaptive_performance=adaptive_profile,
            )
        )

    case_count = len(benchmark_cases)
    return RoutingPerformanceResult(
        case_count=case_count,
        static_metrics=RoutingPerformanceMetrics(
            average_quality_score=_average(static_quality),
            average_cost_units=_average(static_cost),
            average_latency_ms=_average(static_latency),
        ),
        adaptive_metrics=RoutingPerformanceMetrics(
            average_quality_score=_average(adaptive_quality),
            average_cost_units=_average(adaptive_cost),
            average_latency_ms=_average(adaptive_latency),
        ),
        case_results=tuple(case_results),
    )
