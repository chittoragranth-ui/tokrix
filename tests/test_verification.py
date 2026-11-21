import pytest

from tokrix.execution import ExecutionResponse
from tokrix.verification import VerificationResult, verify_response


def test_valid_response_passes_all_checks() -> None:
    result = verify_response(ExecutionResponse(content="A generated answer"))

    assert isinstance(result, VerificationResult)
    assert result.passed is True
    assert len(result.checks) == 4
    assert result.issues == []


@pytest.mark.parametrize("content", ["", " \t\n "])
def test_empty_or_whitespace_response_fails(content: str) -> None:
    result = verify_response(ExecutionResponse(content=content))

    assert result.passed is False
    assert "Response content is empty or whitespace-only." in result.issues
    assert "Response content is a string." in result.checks
    assert "Response content is not empty or whitespace-only." not in result.checks


def test_invalid_response_type_is_handled() -> None:
    result = verify_response("not a response")

    assert result.passed is False
    assert "Response exists." in result.checks
    assert "Response is not an ExecutionResponse." in result.issues


@pytest.mark.parametrize("response", [None])
def test_missing_response_is_handled(response: object) -> None:
    result = verify_response(response)

    assert result.passed is False
    assert result.checks == []
    assert "Response is missing." in result.issues


def test_non_string_content_is_reported() -> None:
    result = verify_response(
        ExecutionResponse(content=42)  # type: ignore[arg-type]
    )

    assert result.passed is False
    assert "Response is an ExecutionResponse." in result.checks
    assert "Response content is not a string." in result.issues
