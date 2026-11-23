"""Provider-independent model tier routing."""

from tokrix.routing.adaptive import AdaptiveRoutingResult, recommend_tier
from tokrix.routing.benchmark import (
    RoutingBenchmarkCase,
    RoutingBenchmarkCaseResult,
    RoutingBenchmarkResult,
    RoutingStrategyMetrics,
    evaluate_routing,
)
from tokrix.routing.dataset import BenchmarkDatasetEntry, get_benchmark_dataset
from tokrix.routing.router import ModelTier, RoutingDecision, route_model

__all__ = [
    "AdaptiveRoutingResult",
    "BenchmarkDatasetEntry",
    "ModelTier",
    "RoutingBenchmarkCase",
    "RoutingBenchmarkCaseResult",
    "RoutingBenchmarkResult",
    "RoutingDecision",
    "RoutingStrategyMetrics",
    "evaluate_routing",
    "get_benchmark_dataset",
    "recommend_tier",
    "route_model",
]