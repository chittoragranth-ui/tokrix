import pytest

from tokrix.intelligence.classifier import (
    ClassificationResult,
    TaskType,
    classify_task,
)


@pytest.mark.parametrize(
    ("prompt", "expected_type"),
    [
        ("Write code to implement a sorting function.", TaskType.CODING),
        ("Solve this step by step and prove the result.", TaskType.REASONING),
        ("Summarize the key points of this article.", TaskType.SUMMARIZATION),
        ("What is photosynthesis?", TaskType.QUESTION_ANSWERING),
        ("Write a short story about a dragon.", TaskType.CREATIVE_WRITING),
        ("Analyze the data in this dataset and identify trends.", TaskType.DATA_ANALYSIS),
    ],
)
def test_classifies_each_specific_category(
    prompt: str, expected_type: TaskType
) -> None:
    result = classify_task(prompt)

    assert result.task_type is expected_type
    assert result.reasons
    assert all(isinstance(reason, str) for reason in result.reasons)


def test_general_is_returned_when_evidence_is_insufficient() -> None:
    result = classify_task("Please help me with this.")

    assert result.task_type is TaskType.GENERAL
    assert result.confidence == 0.0
    assert result.reasons


@pytest.mark.parametrize("prompt", [None, 42, ["write code"]])
def test_non_string_prompt_is_rejected(prompt: object) -> None:
    with pytest.raises(TypeError, match="prompt must be a string"):
        classify_task(prompt)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt", ["", " \t\n "])
def test_empty_or_whitespace_prompt_is_rejected(prompt: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        classify_task(prompt)


def test_confidence_is_a_bounded_float_with_explainable_reasons() -> None:
    result = classify_task("Write code to implement a Python function.")

    assert isinstance(result, ClassificationResult)
    assert type(result.confidence) is float
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasons


def test_ambiguous_prompt_with_tied_categories_returns_general() -> None:
    result = classify_task("Write a poem and write code.")

    assert result.task_type is TaskType.GENERAL
    assert result.confidence == 0.0
    assert "classified as general" in result.reasons[0]


@pytest.mark.parametrize(
    ("prompt", "expected_indicator", "overlapping_indicator"),
    [
        ("Write a poem", "write a poem", "poem"),
        ("Write code", "write code", "code"),
    ],
)
def test_overlapping_phrases_count_as_single_piece_of_evidence(
    prompt: str, expected_indicator: str, overlapping_indicator: str
) -> None:
    result = classify_task(prompt)

    assert result.confidence == 0.5
    assert result.reasons[0].endswith(f"indicators: {expected_indicator}.")
    assert f"indicators: {overlapping_indicator}" not in result.reasons[0]


def test_distinct_indicators_still_contribute_independent_evidence() -> None:
    result = classify_task("Write a poem and add lyrics.")

    assert result.task_type is TaskType.CREATIVE_WRITING
    assert result.confidence == 0.83
    assert "write a poem" in result.reasons[0]
    assert "lyrics" in result.reasons[0]