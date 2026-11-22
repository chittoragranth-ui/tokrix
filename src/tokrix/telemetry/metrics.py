"""In-memory aggregation of completed Tokrix pipeline results."""

import math
from dataclasses import dataclass

from tokrix.decision.engine import Action, DecisionResult
from tokrix.intelligence.classifier import ClassificationResult, TaskType
from tokrix.pipeline.orchestrator import PipelineResult
from tokrix.routing.router import ModelTier
from tokrix.verification.verifier import VerificationResult


@dataclass
class TierMetrics:
    """Attempt and latency metrics for one model tier."""

    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    success_rate: float
    average_execution_latency_ms: float


@dataclass
class TaskTypeMetrics:
    """Run outcomes for one classified task type."""

    total_runs: int
    accepted_runs: int
    stopped_runs: int


@dataclass
class MetricsSnapshot:
    """Aggregated run, attempt, decision, tier, and task metrics."""

    total_runs: int
    accepted_runs: int
    stopped_runs: int
    average_pipeline_duration_ms: float
    total_retries: int
    total_escalations: int
    total_stopped_decisions: int
    tier_metrics: dict[ModelTier, TierMetrics]
    task_type_metrics: dict[TaskType, TaskTypeMetrics]


class MetricsAggregator:
    """Aggregate metrics from each recorded PipelineResult observation.

    Calling ``record`` more than once for the same result counts each call as
    a separate observation.
    """

    def __init__(self) -> None:
        self._total_runs = 0
        self._accepted_runs = 0
        self._stopped_runs = 0
        self._pipeline_duration_total = 0.0
        self._total_retries = 0
        self._total_escalations = 0
        self._total_stopped_decisions = 0
        self._tier_attempts = {tier: 0 for tier in ModelTier}
        self._tier_successes = {tier: 0 for tier in ModelTier}
        self._tier_failures = {tier: 0 for tier in ModelTier}
        self._tier_latency_total = {tier: 0.0 for tier in ModelTier}
        self._task_runs = {task_type: 0 for task_type in TaskType}
        self._task_accepted = {task_type: 0 for task_type in TaskType}
        self._task_stopped = {task_type: 0 for task_type in TaskType}

    def record(self, result: PipelineResult) -> None:
        """Record one completed pipeline result without mutating it."""
        if not isinstance(result, PipelineResult):
            raise TypeError("result must be a PipelineResult")
        duration = result.total_duration_ms
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("total_duration_ms must be a finite non-negative number")
        if not isinstance(result.classification_result, ClassificationResult):
            raise TypeError("classification_result must be a ClassificationResult")
        if not isinstance(result.classification_result.task_type, TaskType):
            raise TypeError("classification_result.task_type must be a TaskType")

        attempts = result.attempt_history or []
        for attempt in attempts:
            if not isinstance(attempt.verification_result, VerificationResult):
                raise TypeError("attempt verification_result must be a VerificationResult")
            if not isinstance(attempt.decision, DecisionResult):
                raise TypeError("attempt decision must be a DecisionResult")
            if not isinstance(attempt.model_tier, ModelTier):
                raise TypeError("attempt model_tier must be a ModelTier")
            latency = attempt.execution_latency_ms
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError(
                    "execution_latency_ms must be a finite non-negative number"
                )
            if not isinstance(attempt.decision.action, Action):
                raise TypeError("attempt decision action must be an Action")

        final_decision = result.final_decision
        if final_decision is not None:
            if not isinstance(final_decision, DecisionResult):
                raise TypeError("final_decision must be a DecisionResult")
            if not isinstance(final_decision.action, Action):
                raise TypeError("final decision action must be an Action")

        task_type = result.classification_result.task_type
        stopped = final_decision is not None and final_decision.action is Action.STOP
        accepted = final_decision is not None and final_decision.action is Action.ACCEPT

        self._total_runs += 1
        self._pipeline_duration_total += duration
        self._accepted_runs += int(accepted)
        self._stopped_runs += int(stopped)
        self._task_runs[task_type] += 1
        self._task_accepted[task_type] += int(accepted)
        self._task_stopped[task_type] += int(stopped)

        for attempt in attempts:
            tier = attempt.model_tier
            self._tier_attempts[tier] += 1
            self._tier_latency_total[tier] += attempt.execution_latency_ms
            if attempt.verification_result.passed:
                self._tier_successes[tier] += 1
            else:
                self._tier_failures[tier] += 1

            action = attempt.decision.action
            self._total_retries += int(action is Action.RETRY)
            self._total_escalations += int(action is Action.ESCALATE)
            self._total_stopped_decisions += int(action is Action.STOP)

        if (
            stopped
            and (not attempts or attempts[-1].decision.action is not Action.STOP)
        ):
            self._total_stopped_decisions += 1

    def snapshot(self) -> MetricsSnapshot:
        """Return a structured snapshot of all recorded metrics."""
        tier_metrics = {
            tier: TierMetrics(
                total_attempts=self._tier_attempts[tier],
                successful_attempts=self._tier_successes[tier],
                failed_attempts=self._tier_failures[tier],
                success_rate=(
                    self._tier_successes[tier] / self._tier_attempts[tier]
                    if self._tier_attempts[tier]
                    else 0.0
                ),
                average_execution_latency_ms=(
                    self._tier_latency_total[tier] / self._tier_attempts[tier]
                    if self._tier_attempts[tier]
                    else 0.0
                ),
            )
            for tier in ModelTier
        }
        task_type_metrics = {
            task_type: TaskTypeMetrics(
                total_runs=self._task_runs[task_type],
                accepted_runs=self._task_accepted[task_type],
                stopped_runs=self._task_stopped[task_type],
            )
            for task_type in TaskType
        }
        return MetricsSnapshot(
            total_runs=self._total_runs,
            accepted_runs=self._accepted_runs,
            stopped_runs=self._stopped_runs,
            average_pipeline_duration_ms=(
                self._pipeline_duration_total / self._total_runs
                if self._total_runs
                else 0.0
            ),
            total_retries=self._total_retries,
            total_escalations=self._total_escalations,
            total_stopped_decisions=self._total_stopped_decisions,
            tier_metrics=tier_metrics,
            task_type_metrics=task_type_metrics,
        )