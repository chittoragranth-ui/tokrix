import pytest

from tokrix.budget.adaptive import BudgetDecision, BudgetMode
from tokrix.routing.router import ModelTier, RoutingDecision, route_model


def _decision(score: object) -> BudgetDecision:
    return BudgetDecision(
        budget_score=score,  # type: ignore[arg-type]
        mode=BudgetMode.BALANCED,
        reason="test input",
    )


@pytest.mark.parametrize("score", [1, 10, 35])
def test_low_tier_scores(score: int) -> None:
    assert route_model(_decision(score)).tier is ModelTier.LOW


def test_low_boundary_at_35() -> None:
    result = route_model(_decision(35))

    assert result.tier is ModelTier.LOW
    assert "range 1–35" in result.reason


def test_medium_boundary_at_36() -> None:
    result = route_model(_decision(36))

    assert result.tier is ModelTier.MEDIUM
    assert "range 36–70" in result.reason


@pytest.mark.parametrize("score", [40, 55, 69])
def test_medium_tier_scores(score: int) -> None:
    assert route_model(_decision(score)).tier is ModelTier.MEDIUM


def test_medium_boundary_at_70() -> None:
    result = route_model(_decision(70))

    assert result.tier is ModelTier.MEDIUM
    assert "range 36–70" in result.reason


def test_high_boundary_at_71() -> None:
    result = route_model(_decision(71))

    assert result.tier is ModelTier.HIGH
    assert "range 71–100" in result.reason


@pytest.mark.parametrize("score", [80, 99])
def test_high_tier_scores(score: int) -> None:
    assert route_model(_decision(score)).tier is ModelTier.HIGH


def test_score_100_is_high() -> None:
    result = route_model(_decision(100))

    assert result.tier is ModelTier.HIGH
    assert result.budget_score == 100
    assert "range 71–100" in result.reason


def test_returns_routing_decision_with_explanatory_reason() -> None:
    decision = route_model(_decision(55))

    assert isinstance(decision, RoutingDecision)
    assert decision.tier is ModelTier.MEDIUM
    assert decision.budget_score == 55
    assert "55" in decision.reason
    assert "MEDIUM" in decision.reason
    assert "36–70" in decision.reason


def test_invalid_input_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="budget_decision must be a BudgetDecision"):
        route_model("not a decision")  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [0, -1, -25, 101])
def test_out_of_range_scores_raise_value_error(score: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        route_model(_decision(score))


@pytest.mark.parametrize("score", [True, False])
def test_boolean_scores_raise_type_error(score: bool) -> None:
    with pytest.raises(TypeError, match="budget_score must be an integer"):
        route_model(_decision(score))


@pytest.mark.parametrize("score", [1.5, "50", None])
def test_non_integer_scores_raise_type_error(score: object) -> None:
    with pytest.raises(TypeError, match="budget_score must be an integer"):
        route_model(_decision(score))