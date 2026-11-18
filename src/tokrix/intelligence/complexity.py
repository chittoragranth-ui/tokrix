"""Rule-based prompt complexity scoring."""

import re
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    """Coarse complexity bands for a prompt."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ComplexityResult:
    """Heuristic score and the signals that contributed to it."""

    score: int
    level: ComplexityLevel
    reasons: list[str]


_TECHNICAL_TERMS = (
    "algorithm",
    "api",
    "architecture",
    "database",
    "dependency",
    "distributed",
    "interface",
    "modular",
    "optimization",
    "protocol",
    "schema",
    "sql",
)

_PROGRAMMING_MATH_DATA_TERMS = (
    "calculate",
    "code",
    "csv",
    "data",
    "dataset",
    "equation",
    "function",
    "implement",
    "javascript",
    "json",
    "math",
    "proof",
    "python",
    "query",
    "script",
    "statistics",
)

_CONSTRAINT_TERMS = (
    "at least",
    "avoid",
    "exactly",
    "include",
    "must",
    "no more than",
    "only",
    "preserve",
    "required",
    "without",
)

_ACTION_TERMS = (
    "add",
    "analyze",
    "build",
    "calculate",
    "compare",
    "create",
    "describe",
    "design",
    "document",
    "explain",
    "generate",
    "implement",
    "include",
    "inspect",
    "preserve",
    "refactor",
    "solve",
    "sort",
    "test",
    "validate",
    "write",
)


def _contains_term(prompt: str, terms: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"\b{re.escape(term)}\b", prompt, re.IGNORECASE)
        for term in terms
    )


def _length_points(prompt: str) -> tuple[int, str | None]:
    length = len(prompt)
    if length <= 120:
        return 0, None
    if length <= 300:
        return 1, "Prompt length is over 120 characters (+1)."
    if length <= 800:
        return 2, "Prompt length is over 300 characters (+2)."
    return 3, "Prompt length is over 800 characters (+3)."


def _multi_step_points(prompt: str) -> tuple[int, str | None]:
    instruction_count = sum(
        len(re.findall(rf"\b{re.escape(term)}\b", prompt, re.IGNORECASE))
        for term in _ACTION_TERMS
    )
    if instruction_count >= 5:
        return 3, "Five or more actionable instructions are present (+3)."
    if instruction_count >= 3:
        return 2, "Three or four actionable instructions are present (+2)."
    if instruction_count == 2:
        return 1, "Two actionable instructions are present (+1)."
    return 0, None


def _constraint_points(prompt: str) -> tuple[int, str | None]:
    count = sum(
        len(re.findall(rf"\b{re.escape(term)}\b", prompt, re.IGNORECASE))
        for term in _CONSTRAINT_TERMS
    )
    if count >= 3:
        return 2, "Three or more constraint terms are present (+2)."
    if count:
        return 1, "One or two constraint terms are present (+1)."
    return 0, None


def _level_for_score(score: int) -> ComplexityLevel:
    if score <= 3:
        return ComplexityLevel.LOW
    if score <= 6:
        return ComplexityLevel.MEDIUM
    return ComplexityLevel.HIGH


def analyze_complexity(prompt: str) -> ComplexityResult:
    """Estimate prompt complexity using deterministic text heuristics.

    The score reflects visible prompt characteristics, not token usage or
    computational requirements.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt must not be empty or whitespace-only")

    score = 0
    reasons: list[str] = []

    points, reason = _length_points(prompt)
    score += points
    if reason:
        reasons.append(reason)

    points, reason = _multi_step_points(prompt)
    score += points
    if reason:
        reasons.append(reason)

    if _contains_term(prompt, _TECHNICAL_TERMS):
        score += 1
        reasons.append("Technical or architectural terminology is present (+1).")

    points, reason = _constraint_points(prompt)
    score += points
    if reason:
        reasons.append(reason)

    if _contains_term(prompt, _PROGRAMMING_MATH_DATA_TERMS):
        score += 1
        reasons.append("Programming, mathematics, or data terminology is present (+1).")

    score = min(score, 10)
    if not reasons:
        reasons.append("No additional complexity indicators were detected.")

    return ComplexityResult(score=score, level=_level_for_score(score), reasons=reasons)