"""Offline evaluation of static and adaptive routing decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from tokrix.intelligence.classifier import TaskType
from tokrix.routing.adaptive import AdaptiveRoutingResult, recommend_tier
from tokrix.routing.router import ModelTier, RoutingDecision

if TYPE_CHECKING:
    from tokrix.feedback import FeedbackRecord

_TIER_COST_PROXY = MappingProxyType(
    {
        ModelTier.LOW: 1,
        ModelTier.MEDIUM: 2,
        ModelTier.HIGH: 3,
    }
)


class _ImmutableRoutingDecision(RoutingDecision):
    """Routing decision snapshot that cannot be changed through the case."""

    __slots__ = ()

    def __init__(self, decision: RoutingDecision) -> None:
        object.__setattr__(self, "tier", decision.tier)
        object.__setattr__(self, "budget_score", decision.budget_score)
        object.__setattr__(self, "reason", decision.reason)

    def __setattr__(self, name: str, value: object) -> None:
        del value
        raise FrozenInstanceError(f"cannot assign to field {name!r}")


@dataclass(frozen=True)
class RoutingBenchmarkCase:
    """Controlled outcomes for evaluating one routing decision offline.

    Historical feedback informs the adaptive recommendation. Expected outcomes
    are separate benchmark data used only to score selected tiers.
    """

    task_type: TaskType
    static_decision: RoutingDecision
    historical_feedback: tuple[FeedbackRecord, ...]
    expected_outcomes: Mapping[ModelTier, bool]

    def __post_init__(self) -> None:
        from tokrix.feedback import FeedbackRecord

        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type must be a TaskType")
        if not isinstance(self.static_decision, RoutingDecision):
            raise TypeError("static_decision must be a RoutingDecision")
        if not isinstance(self.historical_feedback, tuple):
            raise TypeError("historical_feedback must be a tuple")
        if any(
            not isinstance(record, FeedbackRecord)
            for record in self.historical_feedback
        ):
            raise TypeError(
                "historical_feedback entries must be FeedbackRecord instances"
            )
        if not isinstance(self.expected_outcomes, Mapping):
            raise TypeError("expected_outcomes must be a mapping")
        if any(not isinstance(tier, ModelTier) for tier in self.expected_outcomes):
            raise TypeError("expected_outcomes keys must be ModelTier values")
        if set(self.expected_outcomes) != set(ModelTier):
            raise ValueError("expected_outcomes must contain every ModelTier")
        if any(
            not isinstance(passed, bool)
            for passed in self.expected_outcomes.values()
        ):
            raise TypeError("expected_outcomes values must be bools")

        object.__setattr__(
            self,
            "static_decision",
            _ImmutableRoutingDecision(self.static_decision),
        )
        object.__setattr__(
            self,
            "expected_outcomes",
            MappingProxyType(dict(self.expected_outcomes)),
        )


@dataclass(frozen=True)
class RoutingBenchmarkCaseResult:
    """Selected tiers and controlled verification outcomes for one case."""

    task_type: TaskType
    static_tier: ModelTier
    adaptive_tier: ModelTier
    static_verification_passed: bool
    adaptive_verification_passed: bool


@dataclass(frozen=True)
class RoutingStrategyMetrics:
    """Aggregate verification and relative-cost metrics for one strategy."""

    verification_successes: int
    success_rate: float
    average_tier_cost_proxy: float
    tier_distribution: Mapping[ModelTier, int]


@dataclass(frozen=True)
class RoutingBenchmarkResult:
    """Immutable aggregate comparison of static and adaptive routing."""

    case_count: int
    static_metrics: RoutingStrategyMetrics
    adaptive_metrics: RoutingStrategyMetrics
    adaptive_upgrade_count: int
    adaptive_upgrade_rate: float
    case_results: tuple[RoutingBenchmarkCaseResult, ...]


def _strategy_metrics(
    successes: int,
    tier_cost_total: int,
    distribution: dict[ModelTier, int],
    case_count: int,
) -> RoutingStrategyMetrics:
    return RoutingStrategyMetrics(
        verification_successes=successes,
        success_rate=successes / case_count if case_count else 0.0,
        average_tier_cost_proxy=(
            tier_cost_total / case_count if case_count else 0.0
        ),
        tier_distribution=MappingProxyType(distribution.copy()),
    )


def evaluate_routing(
    cases: Iterable[RoutingBenchmarkCase],
) -> RoutingBenchmarkResult:
    """Compare static and adaptive routing using controlled outcomes only.

    An empty iterable produces zero rates, zero average costs, and empty
    success counts; each tier remains present with a zero count in distributions.
    Tier costs are arbitrary relative units (LOW=1, MEDIUM=2, HIGH=3), not
    provider pricing or token costs.
    """
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Iterable):
        raise TypeError("cases must be an iterable of RoutingBenchmarkCase values")

    benchmark_cases = tuple(cases)
    if any(not isinstance(case, RoutingBenchmarkCase) for case in benchmark_cases):
        raise TypeError("cases entries must be RoutingBenchmarkCase instances")

    static_successes = 0
    adaptive_successes = 0
    static_cost_total = 0
    adaptive_cost_total = 0
    static_distribution = {tier: 0 for tier in ModelTier}
    adaptive_distribution = {tier: 0 for tier in ModelTier}
    upgrades = 0
    case_results: list[RoutingBenchmarkCaseResult] = []

    for case in benchmark_cases:
        recommendation: AdaptiveRoutingResult = recommend_tier(
            case.static_decision,
            case.task_type,
            case.historical_feedback,
        )
        static_tier = case.static_decision.tier
        adaptive_tier = recommendation.recommended_tier
        static_passed = case.expected_outcomes[static_tier]
        adaptive_passed = case.expected_outcomes[adaptive_tier]

        static_successes += static_passed
        adaptive_successes += adaptive_passed
        static_cost_total += _TIER_COST_PROXY[static_tier]
        adaptive_cost_total += _TIER_COST_PROXY[adaptive_tier]
        static_distribution[static_tier] += 1
        adaptive_distribution[adaptive_tier] += 1
        upgrades += adaptive_tier is not static_tier
        case_results.append(
            RoutingBenchmarkCaseResult(
                task_type=case.task_type,
                static_tier=static_tier,
                adaptive_tier=adaptive_tier,
                static_verification_passed=static_passed,
                adaptive_verification_passed=adaptive_passed,
            )
        )

    case_count = len(benchmark_cases)
    return RoutingBenchmarkResult(
        case_count=case_count,
        static_metrics=_strategy_metrics(
            static_successes,
            static_cost_total,
            static_distribution,
            case_count,
        ),
        adaptive_metrics=_strategy_metrics(
            adaptive_successes,
            adaptive_cost_total,
            adaptive_distribution,
            case_count,
        ),
        adaptive_upgrade_count=upgrades,
        adaptive_upgrade_rate=upgrades / case_count if case_count else 0.0,
        case_results=tuple(case_results),
    )
