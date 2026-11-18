import pytest

from tokrix.intelligence.complexity import (
    ComplexityLevel,
    ComplexityResult,
    _level_for_score,
    analyze_complexity,
)


def test_simple_prompt_is_low_complexity() -> None:
    result = analyze_complexity("Hi, how are you?")

    assert result.score == 0
    assert result.level is ComplexityLevel.LOW
    assert result.reasons


def test_simple_python_programming_prompt_is_low_complexity() -> None:
    result = analyze_complexity("Write a Python function to add two numbers.")

    assert result.score == 2
    assert result.level is ComplexityLevel.LOW


def test_simple_technical_question_is_low_complexity() -> None:
    result = analyze_complexity("What is an API?")

    assert result.score == 1
    assert result.level is ComplexityLevel.LOW


def test_sequence_markers_without_multiple_instructions_are_not_complex() -> None:
    result = analyze_complexity(
        "First, what is a sunrise? Then, finally, what is sunset?"
    )

    assert result.score == 0
    assert result.level is ComplexityLevel.LOW
    assert all("actionable instructions" not in reason for reason in result.reasons)


def test_medium_complexity_prompt() -> None:
    result = analyze_complexity(
        "Write a Python function to sort records by date. Include a short "
        "explanation of the algorithm and preserve the input unchanged."
    )

    assert result.score == 6
    assert result.level is ComplexityLevel.MEDIUM


def test_multi_requirement_technical_prompt_is_high_complexity() -> None:
    prompt = (
        "Implement a data-processing service in Python. First, inspect the CSV "
        "dataset; then design a modular architecture for the pipeline; finally, "
        "write tests for the SQL queries. The implementation must preserve "
        "existing behavior, avoid additional dependencies, include error "
        "handling, and document the API."
    )

    result = analyze_complexity(prompt)

    assert result.score == 9
    assert result.level is ComplexityLevel.HIGH
    assert len(result.reasons) == 5


@pytest.mark.parametrize("prompt", [None, 42, ["text"]])
def test_non_string_prompt_is_rejected(prompt: object) -> None:
    with pytest.raises(TypeError, match="prompt must be a string"):
        analyze_complexity(prompt)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt", ["", " \t\n "])
def test_empty_or_whitespace_prompt_is_rejected(prompt: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_complexity(prompt)


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (0, ComplexityLevel.LOW),
        (3, ComplexityLevel.LOW),
        (4, ComplexityLevel.MEDIUM),
        (6, ComplexityLevel.MEDIUM),
        (7, ComplexityLevel.HIGH),
        (10, ComplexityLevel.HIGH),
    ],
)
def test_score_boundaries_map_to_expected_levels(
    score: int, level: ComplexityLevel
) -> None:
    assert _level_for_score(score) is level


def test_result_has_valid_structure_and_bounded_integer_score() -> None:
    result = analyze_complexity("Explain this.")

    assert isinstance(result, ComplexityResult)
    assert type(result.score) is int
    assert 0 <= result.score <= 10
    assert isinstance(result.level, ComplexityLevel)
    assert isinstance(result.reasons, list)
    assert all(isinstance(reason, str) for reason in result.reasons)


def test_complex_prompt_reaches_maximum_score() -> None:
    prompt = (
        "Write a Python API and design its database architecture. First, inspect "
        "the data; then analyze and compare the results; finally, implement and "
        "test the solution. Include validation, preserve compatibility, and "
        "document every requirement and must exactly follow the specification. "
        + "Provide detailed examples and explanations. " * 25
    )

    result = analyze_complexity(prompt)

    assert result.score == 10
    assert result.level is ComplexityLevel.HIGH
    assert any("(+3)" in reason for reason in result.reasons)