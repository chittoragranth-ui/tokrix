import pytest

from tokrix.intelligence.token_estimator import estimate_tokens


def test_short_text_estimates_at_least_one_token() -> None:
    assert estimate_tokens("Hi") == 1


def test_exactly_four_characters_estimate_one_token() -> None:
    assert estimate_tokens("abcd") == 1


def test_more_than_four_characters_round_up() -> None:
    assert estimate_tokens("abcde") == 2


def test_longer_text_estimates_tokens_by_rounding_up() -> None:
    assert estimate_tokens("a" * 100) == 25
    assert estimate_tokens("a" * 101) == 26


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_or_whitespace_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        estimate_tokens(text)


@pytest.mark.parametrize("text", [None, 42, ["text"]])
def test_non_string_text_is_rejected(text: object) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        estimate_tokens(text)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["x", "four", "five!", "some longer text"])
def test_valid_text_returns_positive_integer(text: str) -> None:
    result = estimate_tokens(text)

    assert type(result) is int
    assert result > 0