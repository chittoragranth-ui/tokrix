"""Deterministic synthetic per-tier performance profiles for routing scenarios."""

from __future__ import annotations

from functools import lru_cache

from tokrix.routing.dataset import (
    BenchmarkDatasetEntry,
    ResourceDemand,
    get_benchmark_dataset,
)
from tokrix.routing.performance import PerformanceBenchmarkCase, TierPerformance
from tokrix.routing.router import ModelTier

_BASE_COST = {
    ModelTier.LOW: 1.0,
    ModelTier.MEDIUM: 2.0,
    ModelTier.HIGH: 4.0,
}
_BASE_LATENCY_MS = {
    ModelTier.LOW: 120.0,
    ModelTier.MEDIUM: 250.0,
    ModelTier.HIGH: 480.0,
}
_RESOURCE_MULTIPLIER = {
    ResourceDemand.LOW: 0.85,
    ResourceDemand.MEDIUM: 1.0,
    ResourceDemand.HIGH: 1.35,
}


def _performance_case(entry: BenchmarkDatasetEntry) -> PerformanceBenchmarkCase:
    resource_factor = _RESOURCE_MULTIPLIER[entry.resource_demand]
    complexity_factor = 0.8 + 0.2 * entry.complexity_level
    profiles = {
        tier: TierPerformance(
            quality_score=90.0 if entry.expected_outcomes[tier] else 35.0,
            cost_units=_BASE_COST[tier] * resource_factor,
            latency_ms=(
                _BASE_LATENCY_MS[tier]
                * resource_factor
                * complexity_factor
            ),
        )
        for tier in ModelTier
    }
    return PerformanceBenchmarkCase(
        benchmark_case=entry.benchmark_case,
        tier_performance=profiles,
        scenario_id=entry.case_id,
    )


@lru_cache(maxsize=1)
def get_performance_benchmark_dataset() -> tuple[PerformanceBenchmarkCase, ...]:
    """Return deterministic synthetic performance profiles for all scenarios.

    Profiles are modeled assumptions derived from scenario outcome, complexity,
    and resource demand. They are not measurements of real model providers.
    """
    return tuple(_performance_case(entry) for entry in get_benchmark_dataset())
