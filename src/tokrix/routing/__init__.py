"""Provider-independent model tier routing."""

from tokrix.routing.adaptive import (
    AdaptiveRoutingExplanation,
    AdaptiveRoutingPolicy,
    AdaptiveRoutingReason,
    AdaptiveRoutingResult,
    recommend_tier,
)
from tokrix.routing.benchmark import (
    RoutingBenchmarkCase,
    RoutingBenchmarkCaseResult,
    RoutingBenchmarkResult,
    RoutingStrategyMetrics,
    evaluate_routing,
)
from tokrix.routing.comparison import (
    RoutingBenchmarkComparison,
    RoutingOutcomeComparison,
    compare_routing_results,
)
from tokrix.routing.dataset import BenchmarkDatasetEntry, get_benchmark_dataset
from tokrix.routing.router import ModelTier, RoutingDecision, route_model

__all__ = [
    "AdaptiveRoutingExplanation",
    "AdaptiveRoutingPolicy",
    "AdaptiveRoutingReason",
    "AdaptiveRoutingResult",
    "BenchmarkDatasetEntry",
    "ModelTier",
    "RoutingBenchmarkCase",
    "RoutingBenchmarkCaseResult",
    "RoutingBenchmarkComparison",
    "RoutingBenchmarkResult",
    "RoutingDecision",
    "RoutingOutcomeComparison",
    "RoutingStrategyMetrics",
    "compare_routing_results",
    "evaluate_routing",
    "get_benchmark_dataset",
    "recommend_tier",
    "route_model",
]