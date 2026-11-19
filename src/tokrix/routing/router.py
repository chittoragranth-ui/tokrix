"""Provider-independent routing from budget scores to model tiers."""

from dataclasses import dataclass
from enum import Enum

from tokrix.budget.adaptive import BudgetDecision


class ModelTier(Enum):
    """Relative model capability tier selected by the router."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RoutingDecision:
    """Selected model tier and the budget score that determined it."""

    tier: ModelTier
    budget_score: int
    reason: str


LOW_MAX_SCORE = 35
MEDIUM_MIN_SCORE = 36
MEDIUM_MAX_SCORE = 70
HIGH_MIN_SCORE = 71
MAX_BUDGET_SCORE = 100


def route_model(budget_decision: BudgetDecision) -> RoutingDecision:
    """Route a valid budget decision to a tier using its score only."""
    if not isinstance(budget_decision, BudgetDecision):
        raise TypeError("budget_decision must be a BudgetDecision")

    score = budget_decision.budget_score
    if not isinstance(score, int) or isinstance(score, bool):
        raise TypeError("budget_decision.budget_score must be an integer")
    if not 1 <= score <= MAX_BUDGET_SCORE:
        raise ValueError("budget_decision.budget_score must be between 1 and 100")

    if score <= LOW_MAX_SCORE:
        tier = ModelTier.LOW
        score_range = f"1–{LOW_MAX_SCORE}"
    elif score <= MEDIUM_MAX_SCORE:
        tier = ModelTier.MEDIUM
        score_range = f"{MEDIUM_MIN_SCORE}–{MEDIUM_MAX_SCORE}"
    else:
        tier = ModelTier.HIGH
        score_range = f"{HIGH_MIN_SCORE}–{MAX_BUDGET_SCORE}"

    return RoutingDecision(
        tier=tier,
        budget_score=score,
        reason=(
            f"Budget score {score} selected the {tier.value.upper()} tier "
            f"(range {score_range})."
        ),
    )