"""Conservative, evidence-based adaptive routing recommendations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from tokrix.intelligence.classifier import TaskType
from tokrix.routing.router import ModelTier, RoutingDecision

if TYPE_CHECKING:
    from tokrix.feedback.collector import FeedbackRecord


MINIMUM_ATTEMPTS = 5
MAX_CURRENT_SUCCESS_RATE = Fraction(3, 5)
MIN_HIGHER_SUCCESS_RATE = Fraction(4, 5)
MIN_SUCCESS_RATE_IMPROVEMENT = Fraction(1, 5)

_NEXT_TIER = {
    ModelTier.LOW: ModelTier.MEDIUM,
    ModelTier.MEDIUM: ModelTier.HIGH,
}


@dataclass(frozen=True)
class AdaptiveRoutingResult:
    """Static and evidence-informed tier recommendations."""

    original_tier: ModelTier
    recommended_tier: ModelTier
    adjusted: bool
    task_type: TaskType
    current_tier_samples: int
    current_tier_success_rate: float
    higher_tier_samples: int | None
    higher_tier_success_rate: float | None
    reason: str


def _validate_feedback_record(record: FeedbackRecord) -> None:
    from tokrix.decision.engine import Action
    from tokrix.feedback.collector import FeedbackRecord

    if not isinstance(record, FeedbackRecord):
        raise TypeError("feedback entries must be FeedbackRecord instances")
    if not isinstance(record.task_type, TaskType):
        raise TypeError("feedback task_type must be a TaskType")
    if not isinstance(record.model_tier, ModelTier):
        raise TypeError("feedback model_tier must be a ModelTier")
    if not isinstance(record.verification_passed, bool):
        raise TypeError("feedback verification_passed must be a bool")
    if not isinstance(record.action, Action):
        raise TypeError("feedback action must be an Action")
    if (
        not isinstance(record.attempt_number, int)
        or isinstance(record.attempt_number, bool)
        or record.attempt_number < 1
    ):
        raise ValueError("feedback attempt_number must be a positive integer")
    latency = record.execution_latency_ms
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
    ):
        raise ValueError(
            "feedback execution_latency_ms must be a finite non-negative number"
        )
    if not isinstance(record.pipeline_accepted, bool):
        raise TypeError("feedback pipeline_accepted must be a bool")
    if not isinstance(record.pipeline_stopped, bool):
        raise TypeError("feedback pipeline_stopped must be a bool")


def recommend_tier(
    routing_decision: RoutingDecision,
    task_type: TaskType,
    feedback: Iterable[FeedbackRecord],
) -> AdaptiveRoutingResult:
    """Recommend retaining or upgrading the static tier using feedback.

    Only the requested task type and the current/immediately higher tier are
    compared. This function returns a recommendation; it does not route or
    execute a request. ``task_type`` is explicit because RoutingDecision does
    not contain one.
    """
    if not isinstance(routing_decision, RoutingDecision):
        raise TypeError("routing_decision must be a RoutingDecision")
    if not isinstance(routing_decision.tier, ModelTier):
        raise TypeError("routing_decision.tier must be a ModelTier")
    if not isinstance(task_type, TaskType):
        raise TypeError("task_type must be a TaskType")
    score = routing_decision.budget_score
    if not isinstance(score, int) or isinstance(score, bool):
        raise TypeError("routing_decision.budget_score must be an integer")
    if not 1 <= score <= 100:
        raise ValueError("routing_decision.budget_score must be between 1 and 100")
    if not isinstance(routing_decision.reason, str):
        raise TypeError("routing_decision.reason must be a string")
    if isinstance(feedback, (str, bytes)) or not isinstance(feedback, Iterable):
        raise TypeError("feedback must be an iterable of FeedbackRecord instances")

    records = list(feedback)
    for record in records:
        _validate_feedback_record(record)

    task_type_records = [
        record for record in records if record.task_type is task_type
    ]

    current_tier = routing_decision.tier
    next_tier = _NEXT_TIER.get(current_tier)
    current_records = [
        record
        for record in task_type_records
        if record.model_tier is current_tier
    ]
    current_samples = len(current_records)
    current_successes = sum(record.verification_passed for record in current_records)
    current_rate = (
        current_successes / current_samples if current_samples else 0.0
    )

    if next_tier is None:
        return AdaptiveRoutingResult(
            original_tier=current_tier,
            recommended_tier=current_tier,
            adjusted=False,
            task_type=task_type,
            current_tier_samples=current_samples,
            current_tier_success_rate=current_rate,
            higher_tier_samples=None,
            higher_tier_success_rate=None,
            reason="HIGH is the maximum tier; no upgrade is available.",
        )

    higher_records = [
        record
        for record in task_type_records
        if record.model_tier is next_tier
    ]
    higher_samples = len(higher_records)
    higher_successes = sum(record.verification_passed for record in higher_records)
    higher_rate = higher_successes / higher_samples if higher_samples else 0.0

    if current_samples < MINIMUM_ATTEMPTS:
        reason = (
            f"Retain {current_tier.value.upper()}: only {current_samples} "
            f"current-tier attempts; {MINIMUM_ATTEMPTS} required."
        )
        adjusted = False
    elif current_successes * MAX_CURRENT_SUCCESS_RATE.denominator > (
        current_samples * MAX_CURRENT_SUCCESS_RATE.numerator
    ):
        reason = (
            f"Retain {current_tier.value.upper()}: current-tier success rate "
            f"{current_rate:.0%} is above the {float(MAX_CURRENT_SUCCESS_RATE):.0%} "
            "upgrade limit."
        )
        adjusted = False
    elif higher_samples < MINIMUM_ATTEMPTS:
        reason = (
            f"Retain {current_tier.value.upper()}: only {higher_samples} "
            f"{next_tier.value.upper()} attempts for {task_type.value}; "
            f"{MINIMUM_ATTEMPTS} required."
        )
        adjusted = False
    else:
        current_fraction = Fraction(current_successes, current_samples)
        higher_fraction = Fraction(higher_successes, higher_samples)
        improvement = higher_fraction - current_fraction
        if (
            higher_fraction >= MIN_HIGHER_SUCCESS_RATE
            and improvement >= MIN_SUCCESS_RATE_IMPROVEMENT
        ):
            reason = (
                f"Recommend {next_tier.value.upper()}: same-task success rate "
                f"improves from {current_rate:.0%} to {higher_rate:.0%} "
                f"with at least {MINIMUM_ATTEMPTS} attempts per tier."
            )
            adjusted = True
        else:
            reason = (
                f"Retain {current_tier.value.upper()}: higher-tier evidence "
                f"({higher_rate:.0%}) does not meet the minimum success rate "
                f"and improvement thresholds."
            )
            adjusted = False

    return AdaptiveRoutingResult(
        original_tier=current_tier,
        recommended_tier=next_tier if adjusted else current_tier,
        adjusted=adjusted,
        task_type=task_type,
        current_tier_samples=current_samples,
        current_tier_success_rate=current_rate,
        higher_tier_samples=higher_samples,
        higher_tier_success_rate=higher_rate,
        reason=reason,
    )
