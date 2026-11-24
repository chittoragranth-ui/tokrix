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
from tokrix.routing.dataset import (
    BenchmarkDatasetEntry,
    ResourceDemand,
    WorkloadGroup,
    get_benchmark_dataset,
    validate_benchmark_dataset,
)
from tokrix.routing.performance import (
    PerformanceBenchmarkCase,
    RoutingPerformanceCaseResult,
    RoutingPerformanceMetrics,
    RoutingPerformanceResult,
    TierPerformance,
    evaluate_routing_performance,
)
from tokrix.routing.performance_dataset import get_performance_benchmark_dataset
from tokrix.routing.router import ModelTier, RoutingDecision, route_model

__all__ = [
    "AdaptiveRoutingExplanation",
    "AdaptiveRoutingPolicy",
    "AdaptiveRoutingReason",
    "AdaptiveRoutingResult",
    "BenchmarkDatasetEntry",
    "ModelTier",
    "PerformanceBenchmarkCase",
    "ResourceDemand",
    "RoutingBenchmarkCase",
    "RoutingBenchmarkCaseResult",
    "RoutingBenchmarkComparison",
    "RoutingBenchmarkResult",
    "RoutingDecision",
    "RoutingOutcomeComparison",
    "RoutingPerformanceCaseResult",
    "RoutingPerformanceMetrics",
    "RoutingPerformanceResult",
    "RoutingStrategyMetrics",
    "TierPerformance",
    "WorkloadGroup",
    "compare_routing_results",
    "evaluate_routing",
    "evaluate_routing_performance",
    "get_benchmark_dataset",
    "get_performance_benchmark_dataset",
    "recommend_tier",
    "route_model",
    "validate_benchmark_dataset",
]