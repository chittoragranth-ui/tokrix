"""Curated synthetic cases for repeatable offline routing evaluation.

Expected outcomes are controlled benchmark data, not measurements of real
model performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, TypeAlias

from tokrix.intelligence.classifier import TaskType
from tokrix.routing.benchmark import RoutingBenchmarkCase
from tokrix.routing.router import ModelTier, RoutingDecision

if TYPE_CHECKING:
    from tokrix.feedback import FeedbackRecord

_TierOutcomes: TypeAlias = tuple[bool, bool, bool]

_TIER_SCORE = {
    ModelTier.LOW: 20,
    ModelTier.MEDIUM: 55,
    ModelTier.HIGH: 85,
}


@dataclass(frozen=True)
class BenchmarkDatasetEntry:
    """A stable identifier and prompt paired with a controlled routing case."""

    case_id: str
    prompt: str
    benchmark_case: RoutingBenchmarkCase

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str):
            raise TypeError("case_id must be a string")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty or whitespace-only")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        if not isinstance(self.benchmark_case, RoutingBenchmarkCase):
            raise TypeError("benchmark_case must be a RoutingBenchmarkCase")


def _feedback_record(
    task_type: TaskType,
    tier: ModelTier,
    passed: bool,
    attempt_number: int,
) -> FeedbackRecord:
    from tokrix.decision import Action
    from tokrix.feedback import FeedbackRecord

    return FeedbackRecord(
        attempt_number=attempt_number,
        task_type=task_type,
        model_tier=tier,
        verification_passed=passed,
        action=Action.ACCEPT if passed else Action.RETRY,
        execution_latency_ms=1.0,
        pipeline_accepted=passed,
        pipeline_stopped=not passed,
    )


def _feedback_for_upgrade(
    task_type: TaskType,
    current_tier: ModelTier,
    current_outcomes: tuple[bool, ...],
    next_tier: ModelTier,
    next_outcomes: tuple[bool, ...],
) -> tuple[FeedbackRecord, ...]:
    return (
        *(
            _feedback_record(task_type, current_tier, passed, index + 1)
            for index, passed in enumerate(current_outcomes)
        ),
        *(
            _feedback_record(task_type, next_tier, passed, index + 1)
            for index, passed in enumerate(next_outcomes)
        ),
    )


def _benchmark_case(
    task_type: TaskType,
    static_tier: ModelTier,
    outcomes: _TierOutcomes,
    feedback: tuple[FeedbackRecord, ...] = (),
) -> RoutingBenchmarkCase:
    return RoutingBenchmarkCase(
        task_type=task_type,
        static_decision=RoutingDecision(
            tier=static_tier,
            budget_score=_TIER_SCORE[static_tier],
            reason="Curated offline benchmark baseline.",
        ),
        historical_feedback=feedback,
        expected_outcomes=dict(zip(ModelTier, outcomes, strict=True)),
    )


# These are controlled synthetic outcomes, not measurements of real model
# performance. Historical feedback is separately constructed for policy input.
@lru_cache(maxsize=1)
def get_benchmark_dataset() -> tuple[BenchmarkDatasetEntry, ...]:
    """Return the deterministic built-in synthetic routing benchmark dataset."""
    return (
    BenchmarkDatasetEntry(
        "coding-low-upgrade",
        "Implement a Python function that merges two sorted lists.",
        _benchmark_case(
            TaskType.CODING,
            ModelTier.LOW,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.CODING,
                ModelTier.LOW,
                (True, False, False, False, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "coding-medium-retain",
        "Find the off-by-one error in this loop and suggest a minimal fix.",
        _benchmark_case(
            TaskType.CODING,
            ModelTier.MEDIUM,
            (True, True, False),
            _feedback_for_upgrade(
                TaskType.CODING,
                ModelTier.MEDIUM,
                (True, True, True, True, False),
                ModelTier.HIGH,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "reasoning-high-upgrade",
        "Reason through whether the conclusion follows from the two premises.",
        _benchmark_case(
            TaskType.REASONING,
            ModelTier.MEDIUM,
            (False, True, False),
            _feedback_for_upgrade(
                TaskType.REASONING,
                ModelTier.MEDIUM,
                (True, False, False, False, False),
                ModelTier.HIGH,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "reasoning-low-success",
        "Compare two short arguments and identify which one uses stronger evidence.",
        _benchmark_case(
            TaskType.REASONING,
            ModelTier.LOW,
            (True, True, True),
            _feedback_for_upgrade(
                TaskType.REASONING,
                ModelTier.LOW,
                (True, True, True, True, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "summary-all-success",
        "Summarize the main points of this brief project update in three bullets.",
        _benchmark_case(
            TaskType.SUMMARIZATION,
            ModelTier.HIGH,
            (True, True, True),
        ),
    ),
    BenchmarkDatasetEntry(
        "summary-insufficient",
        "Give a one-sentence summary of the meeting notes.",
        _benchmark_case(
            TaskType.SUMMARIZATION,
            ModelTier.LOW,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.SUMMARIZATION,
                ModelTier.LOW,
                (False, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "qa-no-improvement",
        "What is the difference between a list and a tuple in Python?",
        _benchmark_case(
            TaskType.QUESTION_ANSWERING,
            ModelTier.LOW,
            (True, False, False),
            _feedback_for_upgrade(
                TaskType.QUESTION_ANSWERING,
                ModelTier.LOW,
                (True, False, False, False, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "qa-medium-success",
        "How does a hash table provide average constant-time lookup?",
        _benchmark_case(
            TaskType.QUESTION_ANSWERING,
            ModelTier.MEDIUM,
            (True, True, True),
        ),
    ),
    BenchmarkDatasetEntry(
        "creative-high-only",
        "Write a short scene where a lost traveler finds a welcoming inn.",
        _benchmark_case(
            TaskType.CREATIVE_WRITING,
            ModelTier.LOW,
            (False, False, True),
        ),
    ),
    BenchmarkDatasetEntry(
        "creative-upgrade-no-gain",
        "Write a four-line poem about the first rain after a dry summer.",
        _benchmark_case(
            TaskType.CREATIVE_WRITING,
            ModelTier.LOW,
            (True, True, False),
            _feedback_for_upgrade(
                TaskType.CREATIVE_WRITING,
                ModelTier.LOW,
                (True, False, False, False, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "data-analysis-upgrade",
        "Analyze this small table and describe the clearest month-over-month trend.",
        _benchmark_case(
            TaskType.DATA_ANALYSIS,
            ModelTier.LOW,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.DATA_ANALYSIS,
                ModelTier.LOW,
                (True, False, False, False, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "data-analysis-medium-to-high",
        "Explain whether these summary statistics support the stated conclusion.",
        _benchmark_case(
            TaskType.DATA_ANALYSIS,
            ModelTier.MEDIUM,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.DATA_ANALYSIS,
                ModelTier.MEDIUM,
                (True, False, False, False, False),
                ModelTier.HIGH,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "general-all-fail",
        "Help me organize a few ideas for the weekend.",
        _benchmark_case(
            TaskType.GENERAL,
            ModelTier.LOW,
            (False, False, False),
        ),
    ),
    BenchmarkDatasetEntry(
        "general-all-success",
        "Suggest a clear title for a personal notes folder.",
        _benchmark_case(
            TaskType.GENERAL,
            ModelTier.MEDIUM,
            (True, True, True),
        ),
    ),
    BenchmarkDatasetEntry(
        "general-high-baseline",
        "Offer a concise, practical way to prioritize several unrelated errands.",
        _benchmark_case(
            TaskType.GENERAL,
            ModelTier.HIGH,
            (False, False, False),
        ),
    ),
)
