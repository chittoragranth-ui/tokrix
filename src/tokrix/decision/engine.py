"""Decision logic for retrying or escalating failed verification."""

from dataclasses import dataclass
from enum import Enum

from tokrix.routing.router import ModelTier
from tokrix.verification.verifier import VerificationResult


class Action(Enum):
    """Action to take after response verification."""

    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"
    STOP = "stop"


@dataclass
class DecisionResult:
    """A deterministic action and the tier it applies to."""

    action: Action
    current_tier: ModelTier
    next_tier: ModelTier | None
    reason: str


RetryAction = Action
RetryDecision = DecisionResult


def decide_action(
    verification: VerificationResult,
    current_tier: ModelTier,
    retry_count: int,
    max_retries: int = 1,
) -> DecisionResult:
    """Choose whether to accept, retry, escalate, or stop."""
    if not isinstance(verification, VerificationResult):
        raise TypeError("verification must be a VerificationResult")
    if not isinstance(current_tier, ModelTier):
        raise TypeError("current_tier must be a ModelTier")
    if not isinstance(retry_count, int) or isinstance(retry_count, bool):
        raise TypeError("retry_count must be an integer")
    if retry_count < 0:
        raise ValueError("retry_count must be non-negative")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool):
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    if verification.passed:
        return DecisionResult(
            action=Action.ACCEPT,
            current_tier=current_tier,
            next_tier=None,
            reason="Verification passed; accept the response.",
        )

    if retry_count < max_retries:
        return DecisionResult(
            action=Action.RETRY,
            current_tier=current_tier,
            next_tier=None,
            reason=(
                f"Verification failed; retry at the "
                f"{current_tier.value.upper()} tier."
            ),
        )

    if current_tier is ModelTier.LOW:
        return DecisionResult(
            action=Action.ESCALATE,
            current_tier=current_tier,
            next_tier=ModelTier.MEDIUM,
            reason="Retries exhausted at LOW; escalate to MEDIUM.",
        )
    if current_tier is ModelTier.MEDIUM:
        return DecisionResult(
            action=Action.ESCALATE,
            current_tier=current_tier,
            next_tier=ModelTier.HIGH,
            reason="Retries exhausted at MEDIUM; escalate to HIGH.",
        )
    return DecisionResult(
        action=Action.STOP,
        current_tier=current_tier,
        next_tier=None,
        reason="Retries exhausted at HIGH; stop.",
    )


def decide_retry_action(
    verification: VerificationResult,
    current_tier: ModelTier,
    retry_count: int,
    max_retries: int = 1,
) -> DecisionResult:
    """Alias with explicit retry terminology."""
    return decide_action(
        verification,
        current_tier,
        retry_count,
        max_retries,
    )
