"""In-memory collection of per-attempt pipeline feedback."""

import math
from dataclasses import dataclass

from tokrix.decision.engine import Action, DecisionResult
from tokrix.intelligence.classifier import ClassificationResult, TaskType
from tokrix.pipeline.orchestrator import PipelineAttempt, PipelineResult
from tokrix.routing.router import ModelTier
from tokrix.verification.verifier import VerificationResult


@dataclass(frozen=True)
class FeedbackRecord:
    """Outcome metadata for one pipeline execution attempt."""

    attempt_number: int
    task_type: TaskType
    model_tier: ModelTier
    verification_passed: bool
    action: Action
    execution_latency_ms: float
    pipeline_accepted: bool
    pipeline_stopped: bool


class FeedbackCollector:
    """Collect immutable per-attempt feedback records in memory."""

    def __init__(self) -> None:
        self._records: list[FeedbackRecord] = []

    def record(self, result: PipelineResult) -> None:
        """Create a feedback record for each attempt, in attempt order."""
        if not isinstance(result, PipelineResult):
            raise TypeError("result must be a PipelineResult")
        if not isinstance(result.classification_result, ClassificationResult):
            raise TypeError("classification_result must be a ClassificationResult")

        task_type = result.classification_result.task_type
        if not isinstance(task_type, TaskType):
            raise TypeError("classification_result.task_type must be a TaskType")

        if result.attempt_history is None:
            attempts: list[PipelineAttempt] = []
        elif isinstance(result.attempt_history, list):
            attempts = result.attempt_history
        else:
            raise TypeError("attempt_history must be a list or None")

        final_decision = result.final_decision
        if final_decision is not None:
            if not isinstance(final_decision, DecisionResult):
                raise TypeError("final_decision must be a DecisionResult or None")
            if not isinstance(final_decision.action, Action):
                raise TypeError("final_decision.action must be an Action")

        pipeline_accepted = (
            final_decision is not None and final_decision.action is Action.ACCEPT
        )
        pipeline_stopped = (
            final_decision is not None and final_decision.action is Action.STOP
        )

        new_records: list[FeedbackRecord] = []
        for attempt in attempts:
            if not isinstance(attempt, PipelineAttempt):
                raise TypeError("attempt_history entries must be PipelineAttempt values")
            if not isinstance(attempt.model_tier, ModelTier):
                raise TypeError("attempt.model_tier must be a ModelTier")
            if not isinstance(attempt.verification_result, VerificationResult):
                raise TypeError(
                    "attempt.verification_result must be a VerificationResult"
                )
            if not isinstance(attempt.verification_result.passed, bool):
                raise TypeError("attempt verification passed value must be a bool")
            if not isinstance(attempt.decision, DecisionResult):
                raise TypeError("attempt.decision must be a DecisionResult")
            if not isinstance(attempt.decision.action, Action):
                raise TypeError("attempt.decision.action must be an Action")
            if (
                not isinstance(attempt.attempt_number, int)
                or isinstance(attempt.attempt_number, bool)
                or attempt.attempt_number < 1
            ):
                raise ValueError("attempt_number must be a positive integer")

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

            new_records.append(
                FeedbackRecord(
                    attempt_number=attempt.attempt_number,
                    task_type=task_type,
                    model_tier=attempt.model_tier,
                    verification_passed=attempt.verification_result.passed,
                    action=attempt.decision.action,
                    execution_latency_ms=float(latency),
                    pipeline_accepted=pipeline_accepted,
                    pipeline_stopped=pipeline_stopped,
                )
            )

        self._records.extend(new_records)

    def snapshot(self) -> tuple[FeedbackRecord, ...]:
        """Return a read-only snapshot of collected feedback records."""
        return tuple(self._records)