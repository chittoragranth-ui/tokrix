"""Tests for the documented intelligence package import path."""

from tokrix import intelligence
from tokrix.intelligence.classifier import ClassificationResult, TaskType
from tokrix.intelligence.complexity import ComplexityLevel, ComplexityResult
from tokrix.intelligence.token_estimator import estimate_tokens


def test_package_version_is_v1() -> None:
    import tokrix

    assert tokrix.__version__ == "1.0.0"


def test_intelligence_package_exports_its_public_apis() -> None:
    assert set(intelligence.__all__) == {
        "ClassificationResult",
        "ComplexityLevel",
        "ComplexityResult",
        "TaskType",
        "analyze_complexity",
        "classify_task",
        "estimate_tokens",
    }
    assert intelligence.ClassificationResult is ClassificationResult
    assert intelligence.ComplexityLevel is ComplexityLevel
    assert intelligence.ComplexityResult is ComplexityResult
    assert intelligence.TaskType is TaskType
    assert intelligence.estimate_tokens is estimate_tokens


def test_intelligence_package_functions_are_usable() -> None:
    prompt = "Write code in Python to solve a problem."

    classification = intelligence.classify_task(prompt)
    complexity = intelligence.analyze_complexity(prompt)

    assert classification.task_type is TaskType.CODING
    assert isinstance(complexity, ComplexityResult)
    assert intelligence.estimate_tokens(prompt) > 0
