import pytest

from tokrix.decision import Action, DecisionResult, decide_action
from tokrix.routing import ModelTier
from tokrix.verification import VerificationResult


def failed_verification() -> VerificationResult:
    return VerificationResult(
        passed=False,
        checks=["Response exists."],
        issues=["Response content is empty or whitespace-only."],
    )


def test_passed_verification_accepts() -> None:
    verification = VerificationResult(passed=True, checks=["passed"], issues=[])

    result = decide_action(verification, ModelTier.MEDIUM, retry_count=0)

    assert isinstance(result, DecisionResult)
    assert result.action is Action.ACCEPT
    assert result.current_tier is ModelTier.MEDIUM
    assert result.next_tier is None
    assert "passed" in result.reason


def test_failed_verification_retries_at_same_tier() -> None:
    result = decide_action(failed_verification(), ModelTier.LOW, retry_count=0)

    assert result.action is Action.RETRY
    assert result.current_tier is ModelTier.LOW
    assert result.next_tier is None
    assert "LOW" in result.reason


def test_retry_exhausted_at_low_escalates_to_medium() -> None:
    result = decide_action(failed_verification(), ModelTier.LOW, retry_count=1)

    assert result.action is Action.ESCALATE
    assert result.current_tier is ModelTier.LOW
    assert result.next_tier is ModelTier.MEDIUM


def test_retry_exhausted_at_medium_escalates_to_high() -> None:
    result = decide_action(failed_verification(), ModelTier.MEDIUM, retry_count=1)

    assert result.action is Action.ESCALATE
    assert result.current_tier is ModelTier.MEDIUM
    assert result.next_tier is ModelTier.HIGH


def test_retry_exhausted_at_high_stops() -> None:
    result = decide_action(failed_verification(), ModelTier.HIGH, retry_count=1)

    assert result.action is Action.STOP
    assert result.current_tier is ModelTier.HIGH
    assert result.next_tier is None


def test_max_retries_zero_has_no_retry() -> None:
    result = decide_action(
        failed_verification(),
        ModelTier.LOW,
        retry_count=0,
        max_retries=0,
    )

    assert result.action is Action.ESCALATE
    assert result.next_tier is ModelTier.MEDIUM


def test_invalid_verification_result_is_rejected() -> None:
    with pytest.raises(TypeError, match="verification must be a VerificationResult"):
        decide_action("failed", ModelTier.LOW, 0)  # type: ignore[arg-type]


def test_invalid_tier_is_rejected() -> None:
    with pytest.raises(TypeError, match="current_tier must be a ModelTier"):
        decide_action(failed_verification(), "low", 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("retry_count", [-1, -10])
def test_negative_retry_count_is_rejected(retry_count: int) -> None:
    with pytest.raises(ValueError, match="retry_count must be non-negative"):
        decide_action(failed_verification(), ModelTier.LOW, retry_count)


@pytest.mark.parametrize("max_retries", [-1, -10])
def test_negative_max_retries_is_rejected(max_retries: int) -> None:
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        decide_action(
            failed_verification(),
            ModelTier.LOW,
            0,
            max_retries,
        )


@pytest.mark.parametrize("retry_count", [True, False])
def test_bool_retry_count_is_rejected(retry_count: bool) -> None:
    with pytest.raises(TypeError, match="retry_count must be an integer"):
        decide_action(failed_verification(), ModelTier.LOW, retry_count)


@pytest.mark.parametrize("max_retries", [True, False])
def test_bool_max_retries_is_rejected(max_retries: bool) -> None:
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        decide_action(
            failed_verification(),
            ModelTier.LOW,
            0,
            max_retries,
        )
