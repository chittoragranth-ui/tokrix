"""Deterministic internal resource-allocation scoring."""

from dataclasses import dataclass
from enum import Enum

from tokrix.intelligence.classifier import TaskType


class BudgetMode(Enum):
    """Optimization preference used to set the starting budget score."""

    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    QUALITY_OPTIMIZED = "quality_optimized"


@dataclass
class BudgetDecision:
    """Internal resource-allocation score and its explanation."""

    budget_score: int
    mode: BudgetMode
    reason: str


_STARTING_BUDGETS = {
    BudgetMode.COST_OPTIMIZED: 25,
    BudgetMode.BALANCED: 50,
    BudgetMode.QUALITY_OPTIMIZED: 75,
}

_TASK_ADJUSTMENTS = {
    TaskType.CODING: 3,
    TaskType.REASONING: 3,
    TaskType.DATA_ANALYSIS: 3,
}


def _token_adjustment(estimated_tokens: int) -> int:
    if estimated_tokens <= 100:
        return 0
    if estimated_tokens <= 500:
        return 3
    if estimated_tokens <= 1500:
        return 5
    return 8


def allocate_budget(
    complexity_score: int,
    task_type: TaskType,
    estimated_tokens: int,
    mode: BudgetMode = BudgetMode.BALANCED,
) -> BudgetDecision:
    """Allocate an internal score from 1 to 100 using fixed heuristics."""
    if not isinstance(complexity_score, int) or isinstance(complexity_score, bool):
        raise TypeError("complexity_score must be an integer")
    if not 0 <= complexity_score <= 10:
        raise ValueError("complexity_score must be between 0 and 10 inclusive")

    if not isinstance(estimated_tokens, int) or isinstance(estimated_tokens, bool):
        raise TypeError("estimated_tokens must be an integer")
    if estimated_tokens <= 0:
        raise ValueError("estimated_tokens must be a positive integer")

    if not isinstance(task_type, TaskType):
        raise TypeError("task_type must be a TaskType")
    if not isinstance(mode, BudgetMode):
        raise TypeError("mode must be a BudgetMode")

    starting_budget = _STARTING_BUDGETS[mode]
    complexity_adjustment = complexity_score * 3
    token_adjustment = _token_adjustment(estimated_tokens)
    task_adjustment = _TASK_ADJUSTMENTS.get(task_type, 0)
    raw_score = (
        starting_budget
        + complexity_adjustment
        + token_adjustment
        + task_adjustment
    )
    budget_score = max(1, min(100, raw_score))

    reason = (
        f"{mode.value.replace('_', ' ').title()} mode (base {starting_budget}): "
        f"complexity +{complexity_adjustment}, token size +{token_adjustment}, "
        f"task +{task_adjustment}; "
    )
    if budget_score != raw_score:
        reason += f"raw score {raw_score} capped to {budget_score}."
    else:
        reason += f"final score {budget_score}."

    return BudgetDecision(
        budget_score=budget_score,
        mode=mode,
        reason=reason,
    )