"""Explicit, provider-backed empirical benchmark runs."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from tokrix.execution import (
    ExecutionEngine,
    ExecutionRequest,
    ExecutionResponse,
    ModelProvider,
)
from tokrix.routing.router import ModelTier

MAX_BENCHMARK_REPETITIONS = 100


@dataclass(frozen=True)
class EmpiricalBenchmarkCase:
    """Prompt and optional immutable metadata for one empirical case."""

    case_id: str
    prompt: str
    metadata: Mapping[str, str | int | float | bool | None] | None = None
    evaluation_context: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str):
            raise TypeError("case_id must be a string")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty or whitespace-only")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        if self.evaluation_context is not None and not isinstance(
            self.evaluation_context, str
        ):
            raise TypeError("evaluation_context must be a string or None")
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise TypeError("metadata must be a mapping or None")
            for key, value in self.metadata.items():
                if not isinstance(key, str):
                    raise TypeError("metadata keys must be strings")
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    raise TypeError("metadata values must be scalar values or None")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("metadata numeric values must be finite")
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class TokenPricing:
    """Explicit cost per million input and output tokens for one model."""

    input_cost_per_million_tokens: int | float
    output_cost_per_million_tokens: int | float

    def __post_init__(self) -> None:
        for field_name in (
            "input_cost_per_million_tokens",
            "output_cost_per_million_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number")
            if (isinstance(value, float) and not math.isfinite(value)) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True)
class BenchmarkTarget:
    """One explicitly selected provider/model/tier and optional pricing."""

    target_id: str
    provider: ModelProvider
    model_name: str | None = None
    tier: ModelTier | None = None
    pricing: TokenPricing | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str):
            raise TypeError("target_id must be a string")
        if not self.target_id.strip():
            raise ValueError("target_id must not be empty or whitespace-only")
        if not callable(getattr(self.provider, "execute", None)):
            raise TypeError("provider must implement a callable execute method")
        if self.model_name is not None:
            if not isinstance(self.model_name, str):
                raise TypeError("model_name must be a string or None")
            if not self.model_name.strip():
                raise ValueError("model_name must not be empty or whitespace-only")
        if self.tier is not None and not isinstance(self.tier, ModelTier):
            raise TypeError("tier must be a ModelTier or None")
        if self.pricing is not None and not isinstance(self.pricing, TokenPricing):
            raise TypeError("pricing must be a TokenPricing or None")


QualityEvaluator = Callable[[EmpiricalBenchmarkCase, ExecutionResponse], int | float]


@dataclass(frozen=True)
class EmpiricalBenchmarkConfig:
    """Immutable cases, explicitly selected targets, and bounded run options."""

    cases: tuple[EmpiricalBenchmarkCase, ...]
    targets: tuple[BenchmarkTarget, ...]
    repetitions: int = 1
    quality_evaluator: QualityEvaluator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cases, (tuple, list)):
            raise TypeError("cases must be a tuple or list of EmpiricalBenchmarkCase")
        if not isinstance(self.targets, (tuple, list)):
            raise TypeError("targets must be a tuple or list of BenchmarkTarget")
        cases = tuple(self.cases)
        targets = tuple(self.targets)
        if any(not isinstance(case, EmpiricalBenchmarkCase) for case in cases):
            raise TypeError("cases entries must be EmpiricalBenchmarkCase instances")
        if any(not isinstance(target, BenchmarkTarget) for target in targets):
            raise TypeError("targets entries must be BenchmarkTarget instances")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("case identifiers must be unique")
        if not targets:
            raise ValueError("at least one benchmark target must be configured")
        if len({target.target_id for target in targets}) != len(targets):
            raise ValueError("target identifiers must be unique")
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
        ):
            raise TypeError("repetitions must be an integer")
        if not 1 <= self.repetitions <= MAX_BENCHMARK_REPETITIONS:
            raise ValueError(
                f"repetitions must be between 1 and {MAX_BENCHMARK_REPETITIONS}"
            )
        if self.quality_evaluator is not None and not callable(self.quality_evaluator):
            raise TypeError("quality_evaluator must be callable or None")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class EmpiricalRunMeasurement:
    """Auditable measurements for one case/target/repetition execution."""

    target_id: str
    model_name: str | None
    tier: ModelTier | None
    case_id: str
    repetition: int
    latency_ms: float
    execution_succeeded: bool
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    provider_reported_cost: float | None
    estimated_cost: float | None
    quality_score: float | None
    error_type: str | None
    error_message: str | None
    quality_error: str | None
    estimated_cost_error: str | None


@dataclass(frozen=True)
class MetricAggregate:
    """Mean of available measurements and their coverage."""

    average: float | None
    available_count: int
    total_count: int
    aggregation_error: str | None = None


@dataclass(frozen=True)
class EmpiricalProviderSummary:
    """Per-target runs and aggregates, with partial data made explicit."""

    target_id: str
    model_name: str | None
    tier: ModelTier | None
    runs: tuple[EmpiricalRunMeasurement, ...]
    successful_runs: int
    failed_runs: int
    latency: MetricAggregate
    input_tokens: MetricAggregate
    output_tokens: MetricAggregate
    total_tokens: MetricAggregate
    provider_reported_cost: MetricAggregate
    estimated_cost: MetricAggregate
    quality: MetricAggregate


@dataclass(frozen=True)
class EmpiricalBenchmarkResult:
    """Immutable aggregate result for a complete explicit benchmark run."""

    case_count: int
    repetitions: int
    providers: tuple[EmpiricalProviderSummary, ...]


def _aggregate(values: list[int | float | None], total_count: int) -> MetricAggregate:
    available = [value for value in values if value is not None]
    average = None
    aggregation_error = None
    if available:
        exact_average = sum(Fraction(value) for value in available) / len(available)
        try:
            average = float(exact_average)
        except OverflowError as error:
            raise OverflowError(
                "aggregate average is not representable as a finite float"
            ) from error
        if not math.isfinite(average):
            raise OverflowError(
                "aggregate average is not representable as a finite float"
            )
        if average == 0.0 and exact_average:
            average = None
            aggregation_error = (
                "positive aggregate average underflowed float representation"
            )
    return MetricAggregate(average, len(available), total_count, aggregation_error)


def _quality_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("quality evaluator must return a numeric score")
    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ) or not 0 <= value <= 100:
        raise ValueError("quality score must be finite and between 0 and 100")
    return float(value)


def run_empirical_benchmark(
    config: EmpiricalBenchmarkConfig,
) -> EmpiricalBenchmarkResult:
    """Explicitly execute configured cases and targets; this may incur cost.

    Provider execution failures and quality-evaluator failures are retained on
    individual run records. Missing measurements remain ``None`` and aggregates
    report both measurement coverage and total run counts. Aggregates that
    overflow a finite float raise ``OverflowError``; positive values that
    underflow to zero are reported as unavailable.
    """
    if not isinstance(config, EmpiricalBenchmarkConfig):
        raise TypeError("config must be an EmpiricalBenchmarkConfig")

    summaries: list[EmpiricalProviderSummary] = []
    for target in config.targets:
        target_runs: list[EmpiricalRunMeasurement] = []
        engine = ExecutionEngine(target.provider)
        for case in config.cases:
            for repetition in range(1, config.repetitions + 1):
                started = time.perf_counter_ns()
                response: ExecutionResponse | None = None
                execution_error: Exception | None = None
                try:
                    response = engine.execute(ExecutionRequest(prompt=case.prompt))
                except Exception as error:  # noqa: BLE001
                    execution_error = error
                latency_ms = max(0, time.perf_counter_ns() - started) / 1_000_000

                quality_score = None
                quality_error = None
                if response is not None and config.quality_evaluator is not None:
                    try:
                        quality_score = _quality_score(
                            config.quality_evaluator(case, response)
                        )
                    except Exception as error:  # noqa: BLE001
                        quality_error = f"{type(error).__name__}: {error}"

                input_tokens = response.input_tokens if response else None
                output_tokens = response.output_tokens if response else None
                total_tokens = response.total_tokens if response else None
                reported_cost = (
                    response.provider_reported_cost if response else None
                )
                estimated_cost = None
                estimated_cost_error = None
                if (
                    response is not None
                    and target.pricing is not None
                    and input_tokens is not None
                    and output_tokens is not None
                ):
                    exact_cost = (
                        Fraction(input_tokens)
                        * Fraction(target.pricing.input_cost_per_million_tokens)
                        + Fraction(output_tokens)
                        * Fraction(target.pricing.output_cost_per_million_tokens)
                    ) / 1_000_000
                    try:
                        estimated_cost = float(exact_cost)
                    except OverflowError:
                        estimated_cost_error = (
                            "estimated cost is not representable as a finite float"
                        )
                    else:
                        if not math.isfinite(estimated_cost):
                            estimated_cost = None
                            estimated_cost_error = (
                                "estimated cost is not representable as a finite float"
                            )
                        elif estimated_cost == 0.0 and exact_cost:
                            estimated_cost = None
                            estimated_cost_error = (
                                "estimated cost is too small to represent as a "
                                "non-zero float"
                            )

                target_runs.append(
                    EmpiricalRunMeasurement(
                        target_id=target.target_id,
                        model_name=target.model_name,
                        tier=target.tier,
                        case_id=case.case_id,
                        repetition=repetition,
                        latency_ms=latency_ms,
                        execution_succeeded=execution_error is None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        provider_reported_cost=reported_cost,
                        estimated_cost=estimated_cost,
                        quality_score=quality_score,
                        error_type=(
                            type(execution_error).__name__
                            if execution_error is not None
                            else None
                        ),
                        error_message=(
                            str(execution_error)
                            if execution_error is not None
                            else None
                        ),
                        quality_error=quality_error,
                        estimated_cost_error=estimated_cost_error,
                    )
                )

        total_runs = len(target_runs)
        successful_runs = sum(run.execution_succeeded for run in target_runs)
        summaries.append(
            EmpiricalProviderSummary(
                target_id=target.target_id,
                model_name=target.model_name,
                tier=target.tier,
                runs=tuple(target_runs),
                successful_runs=successful_runs,
                failed_runs=total_runs - successful_runs,
                latency=_aggregate(
                    [run.latency_ms for run in target_runs],
                    total_runs,
                ),
                input_tokens=_aggregate(
                    [run.input_tokens for run in target_runs],
                    total_runs,
                ),
                output_tokens=_aggregate(
                    [run.output_tokens for run in target_runs],
                    total_runs,
                ),
                total_tokens=_aggregate(
                    [run.total_tokens for run in target_runs],
                    total_runs,
                ),
                provider_reported_cost=_aggregate(
                    [run.provider_reported_cost for run in target_runs],
                    total_runs,
                ),
                estimated_cost=_aggregate(
                    [run.estimated_cost for run in target_runs],
                    total_runs,
                ),
                quality=_aggregate(
                    [run.quality_score for run in target_runs],
                    total_runs,
                ),
            )
        )

    return EmpiricalBenchmarkResult(
        case_count=len(config.cases),
        repetitions=config.repetitions,
        providers=tuple(summaries),
    )
