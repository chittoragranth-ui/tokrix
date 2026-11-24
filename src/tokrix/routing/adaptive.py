"""Conservative, evidence-based adaptive routing recommendations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
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


class AdaptiveRoutingReason(Enum):
    """Stable reason codes for adaptive routing outcomes."""

    UPGRADE_APPROVED = "upgrade_approved"
    CURRENT_TIER_SUFFICIENT = "current_tier_sufficient"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HIGHER_TIER_POLICY_NOT_MET = "higher_tier_policy_not_met"
    NO_ELIGIBLE_HIGHER_TIER = "no_eligible_higher_tier"
    ADAPTIVE_ROUTING_DISABLED = "adaptive_routing_disabled"
    FEEDBACK_COLLECTOR_UNAVAILABLE = "feedback_collector_unavailable"


@dataclass(frozen=True)
class AdaptiveRoutingPolicy:
    """Immutable thresholds governing evidence-based adaptive upgrades."""

    minimum_samples_per_tier: int = MINIMUM_ATTEMPTS
    max_current_success_rate: Fraction | int | float = MAX_CURRENT_SUCCESS_RATE
    min_higher_success_rate: Fraction | int | float = MIN_HIGHER_SUCCESS_RATE
    min_success_rate_improvement: Fraction | int | float = (
        MIN_SUCCESS_RATE_IMPROVEMENT
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_samples_per_tier, int)
            or isinstance(self.minimum_samples_per_tier, bool)
        ):
            raise TypeError("minimum_samples_per_tier must be an integer")
        if self.minimum_samples_per_tier < 1:
            raise ValueError("minimum_samples_per_tier must be positive")

        for field_name in (
            "max_current_success_rate",
            "min_higher_success_rate",
            "min_success_rate_improvement",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(
                value, (Fraction, int, float)
            ):
                raise TypeError(f"{field_name} must be a number between 0 and 1")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            fraction = value if isinstance(value, Fraction) else Fraction(str(value))
            if not 0 <= fraction <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, fraction)


def _immutable_tier_mapping(
    values: Mapping[ModelTier, int | float | None],
) -> Mapping[ModelTier, int | float | None]:
    return MappingProxyType(dict(values))


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
    reason_code: AdaptiveRoutingReason = AdaptiveRoutingReason.INSUFFICIENT_EVIDENCE
    evidence_sufficient: bool | None = None
    tier_sample_counts: Mapping[ModelTier, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    tier_success_rates: Mapping[ModelTier, float | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tier_sample_counts",
            _immutable_tier_mapping(self.tier_sample_counts),
        )
        object.__setattr__(
            self,
            "tier_success_rates",
            _immutable_tier_mapping(self.tier_success_rates),
        )


@dataclass(frozen=True)
class AdaptiveRoutingExplanation:
    """Immutable explanation of an adaptive routing decision or skipped check."""

    original_tier: ModelTier
    recommended_tier: ModelTier
    task_type: TaskType
    adaptive_routing_enabled: bool
    feedback_available: bool
    evidence_sufficient: bool | None
    tier_sample_counts: Mapping[ModelTier, int]
    tier_success_rates: Mapping[ModelTier, float | None]
    policy: AdaptiveRoutingPolicy
    upgrade_made: bool
    reason_code: AdaptiveRoutingReason
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tier_sample_counts",
            _immutable_tier_mapping(self.tier_sample_counts),
        )
        object.__setattr__(
            self,
            "tier_success_rates",
            _immutable_tier_mapping(self.tier_success_rates),
        )


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
    policy: AdaptiveRoutingPolicy | None = None,
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
    if policy is None:
        policy = AdaptiveRoutingPolicy()
    elif not isinstance(policy, AdaptiveRoutingPolicy):
        raise TypeError("policy must be an AdaptiveRoutingPolicy or None")
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
    tier_samples = {
        tier: sum(record.model_tier is tier for record in task_type_records)
        for tier in ModelTier
    }
    tier_rates: dict[ModelTier, float | None] = {}
    for tier in ModelTier:
        tier_records = [
            record for record in task_type_records if record.model_tier is tier
        ]
        tier_rates[tier] = (
            sum(record.verification_passed for record in tier_records)
            / len(tier_records)
            if tier_records
            else None
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
            reason_code=AdaptiveRoutingReason.NO_ELIGIBLE_HIGHER_TIER,
            evidence_sufficient=None,
            tier_sample_counts=tier_samples,
            tier_success_rates=tier_rates,
        )

    higher_records = [
        record
        for record in task_type_records
        if record.model_tier is next_tier
    ]
    higher_samples = len(higher_records)
    higher_successes = sum(record.verification_passed for record in higher_records)
    higher_rate = higher_successes / higher_samples if higher_samples else 0.0

    if current_samples < policy.minimum_samples_per_tier:
        reason = (
            f"Retain {current_tier.value.upper()}: only {current_samples} "
            f"current-tier attempts; {policy.minimum_samples_per_tier} required."
        )
        adjusted = False
        reason_code = AdaptiveRoutingReason.INSUFFICIENT_EVIDENCE
        evidence_sufficient = False
    elif current_successes * policy.max_current_success_rate.denominator > (
        current_samples * policy.max_current_success_rate.numerator
    ):
        reason = (
            f"Retain {current_tier.value.upper()}: current-tier success rate "
            f"{current_rate:.0%} is above the "
            f"{float(policy.max_current_success_rate):.0%} "
            "upgrade limit."
        )
        adjusted = False
        reason_code = AdaptiveRoutingReason.CURRENT_TIER_SUFFICIENT
        evidence_sufficient = True
    elif higher_samples < policy.minimum_samples_per_tier:
        reason = (
            f"Retain {current_tier.value.upper()}: only {higher_samples} "
            f"{next_tier.value.upper()} attempts for {task_type.value}; "
            f"{policy.minimum_samples_per_tier} required."
        )
        adjusted = False
        reason_code = AdaptiveRoutingReason.INSUFFICIENT_EVIDENCE
        evidence_sufficient = False
    else:
        current_fraction = Fraction(current_successes, current_samples)
        higher_fraction = Fraction(higher_successes, higher_samples)
        improvement = higher_fraction - current_fraction
        if (
            higher_fraction >= policy.min_higher_success_rate
            and improvement >= policy.min_success_rate_improvement
        ):
            reason = (
                f"Recommend {next_tier.value.upper()}: same-task success rate "
                f"improves from {current_rate:.0%} to {higher_rate:.0%} "
                f"with at least {policy.minimum_samples_per_tier} attempts "
                "per tier."
            )
            adjusted = True
            reason_code = AdaptiveRoutingReason.UPGRADE_APPROVED
            evidence_sufficient = True
        else:
            reason = (
                f"Retain {current_tier.value.upper()}: higher-tier evidence "
                f"({higher_rate:.0%}) does not meet the minimum success rate "
                f"and improvement thresholds."
            )
            adjusted = False
            reason_code = AdaptiveRoutingReason.HIGHER_TIER_POLICY_NOT_MET
            evidence_sufficient = True

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
        reason_code=reason_code,
        evidence_sufficient=evidence_sufficient,
        tier_sample_counts=tier_samples,
        tier_success_rates=tier_rates,
    )
