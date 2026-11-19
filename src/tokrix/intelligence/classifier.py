"""Deterministic, rule-based prompt task classification."""

import re
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Supported high-level prompt task categories."""

    CODING = "coding"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    GENERAL = "general"


@dataclass
class ClassificationResult:
    """Task category, heuristic confidence, and supporting evidence."""

    task_type: TaskType
    confidence: float
    reasons: list[str]


# A phrase match is stronger evidence than a single broad keyword.
_RULES: dict[TaskType, tuple[tuple[str, int], ...]] = {
    TaskType.CODING: (
        ("write code", 3),
        ("write a program", 3),
        ("implement", 2),
        ("debug", 2),
        ("fix the bug", 3),
        ("code", 1),
        ("function", 1),
        ("programming", 2),
        ("python", 1),
        ("javascript", 1),
    ),
    TaskType.REASONING: (
        ("step by step", 3),
        ("reason through", 3),
        ("solve", 2),
        ("prove", 2),
        ("derive", 2),
        ("compare", 1),
        ("evaluate", 1),
        ("why", 1),
        ("logic", 1),
    ),
    TaskType.SUMMARIZATION: (
        ("summarize", 3),
        ("summary", 3),
        ("summarise", 3),
        ("key points", 2),
        ("main points", 2),
        ("tldr", 3),
        ("brief overview", 2),
    ),
    TaskType.QUESTION_ANSWERING: (
        ("what is", 2),
        ("who is", 2),
        ("when did", 2),
        ("where is", 2),
        ("how do i", 2),
        ("how does", 2),
        ("answer the question", 3),
        ("explain what", 2),
        ("explain how", 2),
    ),
    TaskType.CREATIVE_WRITING: (
        ("write a story", 3),
        ("write a poem", 3),
        ("creative writing", 3),
        ("fictional", 2),
        ("short story", 3),
        ("poem", 2),
        ("dialogue", 2),
        ("lyrics", 2),
    ),
    TaskType.DATA_ANALYSIS: (
        ("analyze the data", 3),
        ("analyse the data", 3),
        ("analyze this dataset", 3),
        ("data analysis", 3),
        ("dataset", 2),
        ("statistics", 2),
        ("trend", 1),
        ("visualize", 2),
        ("visualise", 2),
        ("chart", 1),
    ),
}

_MINIMUM_EVIDENCE = 2


def _matched_rules(
    prompt: str, rules: tuple[tuple[str, int], ...]
) -> list[tuple[str, int]]:
    candidates = []
    for phrase, weight in rules:
        spans = [
            match.span()
            for match in re.finditer(
                rf"\b{re.escape(phrase)}\b", prompt, re.IGNORECASE
            )
        ]
        if spans:
            candidates.append((phrase, weight, spans))

    candidates.sort(key=lambda candidate: (-candidate[1], -len(candidate[0])))

    accepted: list[tuple[str, int]] = []
    accepted_spans: list[tuple[int, int]] = []
    for phrase, weight, spans in candidates:
        independent_spans = [
            span
            for span in spans
            if not any(
                span[0] < accepted_end and span[1] > accepted_start
                for accepted_start, accepted_end in accepted_spans
            )
        ]
        if independent_spans:
            accepted.append((phrase, weight))
            accepted_spans.extend(independent_spans)
    return accepted


def classify_task(prompt: str) -> ClassificationResult:
    """Classify a prompt using keywords and patterns.

    Confidence is a simple heuristic derived from rule-match strength and the
    margin over the next category; it is not a statistically calibrated
    probability.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt must not be empty or whitespace-only")

    scores: dict[TaskType, int] = {}
    evidence: dict[TaskType, list[str]] = {}
    for task_type, rules in _RULES.items():
        matched_rules = _matched_rules(prompt, rules)
        scores[task_type] = sum(weight for _, weight in matched_rules)
        evidence[task_type] = [phrase for phrase, _ in matched_rules]

    highest_score = max(scores.values())
    leaders = [
        task_type for task_type, score in scores.items() if score == highest_score
    ]
    if highest_score < _MINIMUM_EVIDENCE:
        return ClassificationResult(
            task_type=TaskType.GENERAL,
            confidence=0.0,
            reasons=["No task category had sufficient keyword or phrase evidence."],
        )
    if len(leaders) != 1:
        ambiguous_categories = ", ".join(task_type.value for task_type in leaders)
        return ClassificationResult(
            task_type=TaskType.GENERAL,
            confidence=0.0,
            reasons=[
                (
                    f"Evidence tied between categories ({ambiguous_categories}); "
                    "classified as general."
                )
            ],
        )

    task_type = leaders[0]
    other_scores = [
        score for category, score in scores.items() if category is not task_type
    ]
    second_score = max(other_scores, default=0)
    if highest_score - second_score < 2:
        second_categories = ", ".join(
            category.value
            for category, score in scores.items()
            if score == second_score and category is not task_type
        )
        return ClassificationResult(
            task_type=TaskType.GENERAL,
            confidence=0.0,
            reasons=[
                (
                    f"Evidence was too close between {task_type.value} "
                    f"({highest_score}) and {second_categories} ({second_score}); "
                    "classified as general."
                )
            ],
        )

    strength = min(highest_score / 6, 1.0)
    margin = (highest_score - second_score) / highest_score
    confidence = round(strength * margin, 2)

    matched_terms = ", ".join(evidence[task_type])
    return ClassificationResult(
        task_type=task_type,
        confidence=confidence,
        reasons=[
            f"Matched {task_type.value} indicators: {matched_terms}.",
            (
                f"Rule evidence score {highest_score}; next category score "
                f"{second_score}."
            ),
        ],
    )