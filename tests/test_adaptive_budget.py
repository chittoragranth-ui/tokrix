import pytest

from tokrix.budget.adaptive import (
    BudgetDecision,
    BudgetMode,
    allocate_budget,
)
from tokrix.intelligence.classifier import TaskType


@pytest.mark.parametrize(
    ("mode", "expected_score"),
    [
        (BudgetMode.COST_OPTIMIZED, 25),
        (BudgetMode.BALANCED, 50),
        (BudgetMode.QUALITY_OPTIMIZED, 75),
    ],
)
def test_mode_sets_starting_budget(mode: BudgetMode, expected_score: int) -> None:
    decision = allocate_budget(0, TaskType.GENERAL, 100, mode)

    assert decision.budget_score == expected_score
    assert decision.mode is mode


@pytest.mark.parametrize(
    ("complexity_score", "expected_score"),
    [(0, 50), (1, 53), (5, 65), (10, 80)],
)
def test_complexity_adds_three_points_per_point(
    complexity_score: int, expected_score: int
) -> None:
    decision = allocate_budget(
        complexity_score, TaskType.GENERAL, 100, BudgetMode.BALANCED
    )

    assert decision.budget_score == expected_score


@pytest.mark.parametrize(
    ("estimated_tokens", "expected_adjustment"),
    [
        (1, 0),
        (100, 0),
        (101, 3),
        (500, 3),
        (501, 5),
        (1500, 5),
        (1501, 8),
    ],
)
def test_token_size_adjustment_boundaries(
    estimated_tokens: int, expected_adjustment: int
) -> None:
    decision = allocate_budget(
        0, TaskType.GENERAL, estimated_tokens, BudgetMode.BALANCED
    )

    assert decision.budget_score == 50 + expected_adjustment


@pytest.mark.parametrize(
    ("task_type", "expected_adjustment"),
    [
        (TaskType.CODING, 3),
        (TaskType.REASONING, 3),
        (TaskType.DATA_ANALYSIS, 3),
        (TaskType.SUMMARIZATION, 0),
        (TaskType.QUESTION_ANSWERING, 0),
        (TaskType.CREATIVE_WRITING, 0),
        (TaskType.GENERAL, 0),
    ],
)
def test_task_adjustments(
    task_type: TaskType, expected_adjustment: int
) -> None:
    decision = allocate_budget(0, task_type, 100)

    assert decision.budget_score == 50 + expected_adjustment


def test_score_is_clamped_at_100_and_explains_the_cap() -> None:
    decision = allocate_budget(
        10, TaskType.CODING, 1501, BudgetMode.QUALITY_OPTIMIZED
    )

    assert decision.budget_score == 100
    assert "raw score 116 capped to 100" in decision.reason


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, ValueError),
        (11, ValueError),
        (1.5, TypeError),
        ("5", TypeError),
        (True, TypeError),
        (False, TypeError),
    ],
)
def test_invalid_complexity_scores(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        allocate_budget(value, TaskType.GENERAL, 100)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        ("100", TypeError),
        (True, TypeError),
        (False, TypeError),
    ],
)
def test_invalid_token_estimates(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        allocate_budget(0, TaskType.GENERAL, value)  # type: ignore[arg-type]


def test_invalid_task_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="task_type must be a TaskType"):
        allocate_budget(0, "coding", 100)  # type: ignore[arg-type]


def test_invalid_mode_raises_type_error() -> None:
    with pytest.raises(TypeError, match="mode must be a BudgetMode"):
        allocate_budget(0, TaskType.GENERAL, 100, "balanced")  # type: ignore[arg-type]


def test_default_mode_is_balanced() -> None:
    decision = allocate_budget(0, TaskType.GENERAL, 100)

    assert decision.mode is BudgetMode.BALANCED
    assert decision.budget_score == 50


def test_decision_fields_and_reason_explain_all_contributions() -> None:
    decision = allocate_budget(2, TaskType.CODING, 101, BudgetMode.COST_OPTIMIZED)

    assert isinstance(decision, BudgetDecision)
    assert decision.budget_score == 37
    assert decision.mode is BudgetMode.COST_OPTIMIZED
    assert "Cost Optimized mode (base 25)" in decision.reason
    assert "complexity +6" in decision.reason
    assert "token size +3" in decision.reason
    assert "task +3" in decision.reason
    assert "final score 37" in decision.reason
