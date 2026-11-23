"""Comparison and delta reporting for offline routing benchmark results."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tokrix.routing.benchmark import RoutingBenchmarkResult
from tokrix.routing.router import ModelTier


@dataclass(frozen=True)
class RoutingOutcomeComparison:
    """Case outcome counts and percentages for adaptive routing.

    Percentages use all benchmark cases as the denominator, and are ``None``
    when the result contains no cases.
    """

    improved_count: int
    improved_percentage: float | None
    degraded_count: int
    degraded_percentage: float | None
    unchanged_count: int
    unchanged_percentage: float | None


@dataclass(frozen=True)
class RoutingBenchmarkComparison:
    """Immutable deltas between static and adaptive benchmark strategies.

    Success-rate difference is in percentage points. Cost and tier-count
    differences use adaptive minus static, so positive values mean more
    relative tier cost or more usage of that tier. Cost units are the
    evaluator's arbitrary tier-cost proxies, not actual provider costs.
    ``relative_tier_cost_proxy_change_percent`` is ``None`` if there are no
    cases or the static average cost is zero.
    """

    case_count: int
    success_rate_difference_percentage_points: float
    average_tier_cost_proxy_difference: float
    relative_tier_cost_proxy_change_percent: float | None
    adaptive_cost_direction: str
    tier_distribution_differences: Mapping[ModelTier, int]
    outcome_comparison: RoutingOutcomeComparison

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tier_distribution_differences",
            MappingProxyType(dict(self.tier_distribution_differences)),
        )


def compare_routing_results(
    result: RoutingBenchmarkResult,
) -> RoutingBenchmarkComparison:
    """Calculate adaptive-minus-static metrics from an evaluated result.

    Relative tier-cost change is the signed average-cost difference divided by
    static average tier-cost proxy, expressed as a percentage. It is undefined
    (``None``) for empty results or a zero static baseline. Empty results use
    zero deltas, ``not_comparable`` cost direction, and ``None`` outcome
    percentages.
    """
    if not isinstance(result, RoutingBenchmarkResult):
        raise TypeError("result must be a RoutingBenchmarkResult")

    case_count = result.case_count
    success_difference = (
        result.adaptive_metrics.success_rate - result.static_metrics.success_rate
    ) * 100
    cost_difference = (
        result.adaptive_metrics.average_tier_cost_proxy
        - result.static_metrics.average_tier_cost_proxy
    )
    if case_count == 0:
        relative_cost_change = None
        cost_direction = "not_comparable"
        outcome_percentage: float | None = None
    else:
        static_cost = result.static_metrics.average_tier_cost_proxy
        relative_cost_change = (
            cost_difference / static_cost * 100 if static_cost != 0 else None
        )
        cost_direction = (
            "more_expensive"
            if cost_difference > 0
            else "less_expensive"
            if cost_difference < 0
            else "same_cost"
        )
        outcome_percentage = 100 / case_count

    improved = sum(
        case.adaptive_verification_passed and not case.static_verification_passed
        for case in result.case_results
    )
    degraded = sum(
        case.static_verification_passed and not case.adaptive_verification_passed
        for case in result.case_results
    )
    unchanged = case_count - improved - degraded

    return RoutingBenchmarkComparison(
        case_count=case_count,
        success_rate_difference_percentage_points=success_difference,
        average_tier_cost_proxy_difference=cost_difference,
        relative_tier_cost_proxy_change_percent=relative_cost_change,
        adaptive_cost_direction=cost_direction,
        tier_distribution_differences={
            tier: (
                result.adaptive_metrics.tier_distribution[tier]
                - result.static_metrics.tier_distribution[tier]
            )
            for tier in ModelTier
        },
        outcome_comparison=RoutingOutcomeComparison(
            improved_count=improved,
            improved_percentage=(
                improved * outcome_percentage if outcome_percentage is not None else None
            ),
            degraded_count=degraded,
            degraded_percentage=(
                degraded * outcome_percentage if outcome_percentage is not None else None
            ),
            unchanged_count=unchanged,
            unchanged_percentage=(
                unchanged * outcome_percentage
                if outcome_percentage is not None
                else None
            ),
        ),
    )
