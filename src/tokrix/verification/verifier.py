"""Deterministic verification for execution responses."""

from dataclasses import dataclass

from tokrix.execution.engine import ExecutionResponse


@dataclass
class VerificationResult:
    """Outcome of deterministic response checks."""

    passed: bool
    checks: list[str]
    issues: list[str]


def verify_response(response: object) -> VerificationResult:
    """Check whether a response has usable generated content."""
    checks: list[str] = []
    issues: list[str] = []

    if response is None:
        issues.append("Response is missing.")
    else:
        checks.append("Response exists.")

    if not isinstance(response, ExecutionResponse):
        if response is not None:
            issues.append("Response is not an ExecutionResponse.")
        return VerificationResult(
            passed=False,
            checks=checks,
            issues=issues,
        )

    checks.append("Response is an ExecutionResponse.")

    if not isinstance(response.content, str):
        issues.append("Response content is not a string.")
    else:
        checks.append("Response content is a string.")
        if response.content.strip():
            checks.append("Response content is not empty or whitespace-only.")
        else:
            issues.append("Response content is empty or whitespace-only.")

    return VerificationResult(
        passed=not issues,
        checks=checks,
        issues=issues,
    )
