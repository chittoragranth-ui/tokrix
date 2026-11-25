"""Curated synthetic cases for repeatable offline routing evaluation.

Expected outcomes are controlled benchmark data, not measurements of real
model performance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
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


class WorkloadGroup(Enum):
    """Representative scenario families in the built-in workload catalog."""

    SIMPLE_FACTUAL = "simple_factual"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    LONG_DOCUMENT_SUMMARIZATION = "long_document_summarization"
    AMBIGUOUS_REQUEST = "ambiguous_request"
    COMPLEX_MULTI_CONSTRAINT = "complex_multi_constraint"
    RESOURCE_CONSTRAINED = "resource_constrained"
    STRUCTURED_DATA_ANALYSIS = "structured_data_analysis"
    CREATIVE_GENERATION = "creative_generation"


class ResourceDemand(Enum):
    """Qualitative resource demand modeled for an evaluation scenario."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class BenchmarkDatasetEntry:
    """Validated workload scenario paired with a controlled routing case."""

    case_id: str
    prompt: str
    benchmark_case: RoutingBenchmarkCase
    complexity_level: int = 3
    workload_group: WorkloadGroup = WorkloadGroup.SIMPLE_FACTUAL
    resource_demand: ResourceDemand = ResourceDemand.MEDIUM

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
        if (
            not isinstance(self.complexity_level, int)
            or isinstance(self.complexity_level, bool)
        ):
            raise TypeError("complexity_level must be an integer")
        if not 1 <= self.complexity_level <= 5:
            raise ValueError("complexity_level must be between 1 and 5")
        if not isinstance(self.workload_group, WorkloadGroup):
            raise TypeError("workload_group must be a WorkloadGroup")
        if not isinstance(self.resource_demand, ResourceDemand):
            raise TypeError("resource_demand must be a ResourceDemand")
        if not isinstance(self.benchmark_case.task_type, TaskType):
            raise TypeError("benchmark_case.task_type must be a TaskType")
        if not isinstance(self.benchmark_case.static_decision.tier, ModelTier):
            raise TypeError("benchmark_case static tier must be a ModelTier")
        if set(self.benchmark_case.expected_outcomes) != set(ModelTier):
            raise ValueError(
                "benchmark_case expected_outcomes must contain every ModelTier"
            )

    @property
    def task_type(self) -> TaskType:
        """Task type evaluated by the paired benchmark case."""
        return self.benchmark_case.task_type

    @property
    def expected_outcomes(self) -> Mapping[ModelTier, bool]:
        """Per-tier synthetic verification outcomes for the scenario."""
        return self.benchmark_case.expected_outcomes


def validate_benchmark_dataset(
    entries: Iterable[BenchmarkDatasetEntry],
) -> None:
    """Validate identifier uniqueness and representative built-in coverage."""
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        raise TypeError("entries must be an iterable of BenchmarkDatasetEntry values")
    dataset = tuple(entries)
    if not dataset:
        raise ValueError("benchmark dataset must not be empty")
    if any(not isinstance(entry, BenchmarkDatasetEntry) for entry in dataset):
        raise TypeError("dataset entries must be BenchmarkDatasetEntry instances")

    case_ids = [entry.case_id for entry in dataset]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("benchmark dataset case_id values must be unique")
    task_coverage = {entry.task_type for entry in dataset}
    if task_coverage != set(TaskType):
        raise ValueError("benchmark dataset must cover every TaskType")
    if len({entry.complexity_level for entry in dataset}) < 3:
        raise ValueError("benchmark dataset must cover at least three complexity levels")
    if {entry.resource_demand for entry in dataset} != set(ResourceDemand):
        raise ValueError("benchmark dataset must cover every ResourceDemand")
    if {entry.workload_group for entry in dataset} != set(WorkloadGroup):
        raise ValueError("benchmark dataset must cover every WorkloadGroup")
    if any(
        len(entry.expected_outcomes) != len(ModelTier)
        or set(entry.expected_outcomes) != set(ModelTier)
        for entry in dataset
    ):
        raise ValueError("every scenario must define outcomes for every ModelTier")


def _validate_scenario_metadata(
    scenario_ids: Iterable[str],
    metadata: Mapping[
        str, tuple[int, WorkloadGroup, ResourceDemand]
    ],
) -> None:
    scenario_id_set = set(scenario_ids)
    metadata_id_set = set(metadata)
    missing = scenario_id_set - metadata_id_set
    unused = metadata_id_set - scenario_id_set
    if missing or unused:
        raise ValueError(
            "scenario metadata keys must exactly match scenario IDs "
            f"(missing={sorted(missing)}, unused={sorted(unused)})"
        )


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
    """Return the validated, deterministic synthetic scenario catalog."""
    dataset = (
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
        "Summarize a long project update covering milestones, risks, owners, "
        "dates, and unresolved decisions in three concise bullets.",
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
    BenchmarkDatasetEntry(
        "coding-constrained-debugging",
        "Given a short Python loop with an off-by-one error, identify the faulty "
        "condition and provide the smallest correction.",
        _benchmark_case(
            TaskType.CODING,
            ModelTier.LOW,
            (True, False, False),
        ),
    ),
    BenchmarkDatasetEntry(
        "coding-multi-constraint-generation",
        "Implement a streaming CSV summarizer that validates required columns, "
        "handles malformed rows, uses bounded memory, and includes focused tests.",
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
        "reasoning-ambiguous-constraints",
        "A project must be fast, cheap, and highly reliable. Which trade-off "
        "should we choose?",
        _benchmark_case(
            TaskType.REASONING,
            ModelTier.LOW,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.REASONING,
                ModelTier.LOW,
                (True, False, False, False, False),
                ModelTier.MEDIUM,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "reasoning-multi-step-proof",
        "For a finite set of integers, prove or disprove that an odd sum of "
        "squares must contain an odd number of odd terms, and explain each step.",
        _benchmark_case(
            TaskType.REASONING,
            ModelTier.MEDIUM,
            (False, False, True),
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
        "summary-long-document",
        "Summarize a long policy document, preserving its exceptions, dates, "
        "responsible teams, and unresolved decisions in a concise structured brief.",
        _benchmark_case(
            TaskType.SUMMARIZATION,
            ModelTier.MEDIUM,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.SUMMARIZATION,
                ModelTier.MEDIUM,
                (True, False, False, False, False),
                ModelTier.HIGH,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "summary-resource-constrained",
        "Summarize this short status update in one sentence without losing its "
        "single action item.",
        _benchmark_case(
            TaskType.SUMMARIZATION,
            ModelTier.LOW,
            (True, True, True),
        ),
    ),
    BenchmarkDatasetEntry(
        "qa-simple-fact",
        "What is the boiling point of pure water at standard atmospheric pressure?",
        _benchmark_case(
            TaskType.QUESTION_ANSWERING,
            ModelTier.LOW,
            (True, False, False),
        ),
    ),
    BenchmarkDatasetEntry(
        "qa-underspecified-request",
        "What is the best database for my application?",
        _benchmark_case(
            TaskType.QUESTION_ANSWERING,
            ModelTier.LOW,
            (False, True, True),
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
        "creative-multi-constraint",
        "Write a short mystery scene for younger readers, with no violence, "
        "three speaking characters, a fair clue, and a hopeful ending.",
        _benchmark_case(
            TaskType.CREATIVE_WRITING,
            ModelTier.MEDIUM,
            (False, True, True),
            _feedback_for_upgrade(
                TaskType.CREATIVE_WRITING,
                ModelTier.MEDIUM,
                (True, False, False, False, False),
                ModelTier.HIGH,
                (True, True, True, True, True),
            ),
        ),
    ),
    BenchmarkDatasetEntry(
        "creative-resource-constrained",
        "Write a two-line thank-you note for a neighbor who watered the plants.",
        _benchmark_case(
            TaskType.CREATIVE_WRITING,
            ModelTier.LOW,
            (True, False, False),
        ),
    ),
    BenchmarkDatasetEntry(
        "data-structured-analysis",
        "Given a table of weekly sales by region, calculate the percentage "
        "change, flag missing values, and summarize the strongest trend.",
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
        "data-higher-tier-regression",
        "Inspect a small, clean table and report its largest value and the "
        "difference between the two displayed totals.",
        _benchmark_case(
            TaskType.DATA_ANALYSIS,
            ModelTier.LOW,
            (True, False, False),
        ),
    ),
    BenchmarkDatasetEntry(
        "general-resource-constrained",
        "List two quick ways to remember a short shopping list.",
        _benchmark_case(
            TaskType.GENERAL,
            ModelTier.LOW,
            (True, True, False),
        ),
    ),
    BenchmarkDatasetEntry(
        "general-ambiguous-multi-constraint",
        "Help me plan the launch.",
        _benchmark_case(
            TaskType.GENERAL,
            ModelTier.MEDIUM,
            (False, False, True),
            _feedback_for_upgrade(
                TaskType.GENERAL,
                ModelTier.MEDIUM,
                (True, False, False, False, False),
                ModelTier.HIGH,
                (True, True, True, True, True),
            ),
        ),
    ),
    )

    scenario_metadata = {
        "coding-low-upgrade": (3, WorkloadGroup.CODE_GENERATION, ResourceDemand.MEDIUM),
        "coding-medium-retain": (2, WorkloadGroup.DEBUGGING, ResourceDemand.LOW),
        "reasoning-high-upgrade": (
            4,
            WorkloadGroup.MULTI_STEP_REASONING,
            ResourceDemand.HIGH,
        ),
        "reasoning-low-success": (
            2,
            WorkloadGroup.MULTI_STEP_REASONING,
            ResourceDemand.LOW,
        ),
        "summary-all-success": (
            4,
            WorkloadGroup.LONG_DOCUMENT_SUMMARIZATION,
            ResourceDemand.HIGH,
        ),
        "summary-insufficient": (
            1,
            WorkloadGroup.RESOURCE_CONSTRAINED,
            ResourceDemand.LOW,
        ),
        "qa-no-improvement": (2, WorkloadGroup.SIMPLE_FACTUAL, ResourceDemand.LOW),
        "qa-medium-success": (2, WorkloadGroup.SIMPLE_FACTUAL, ResourceDemand.LOW),
        "creative-high-only": (
            3,
            WorkloadGroup.CREATIVE_GENERATION,
            ResourceDemand.MEDIUM,
        ),
        "creative-upgrade-no-gain": (
            2,
            WorkloadGroup.CREATIVE_GENERATION,
            ResourceDemand.LOW,
        ),
        "data-analysis-upgrade": (
            3,
            WorkloadGroup.STRUCTURED_DATA_ANALYSIS,
            ResourceDemand.MEDIUM,
        ),
        "data-analysis-medium-to-high": (
            4,
            WorkloadGroup.STRUCTURED_DATA_ANALYSIS,
            ResourceDemand.HIGH,
        ),
        "general-all-fail": (
            3,
            WorkloadGroup.AMBIGUOUS_REQUEST,
            ResourceDemand.MEDIUM,
        ),
        "general-all-success": (1, WorkloadGroup.SIMPLE_FACTUAL, ResourceDemand.LOW),
        "general-high-baseline": (
            4,
            WorkloadGroup.COMPLEX_MULTI_CONSTRAINT,
            ResourceDemand.HIGH,
        ),
        "coding-constrained-debugging": (
            1,
            WorkloadGroup.RESOURCE_CONSTRAINED,
            ResourceDemand.LOW,
        ),
        "coding-multi-constraint-generation": (
            5,
            WorkloadGroup.COMPLEX_MULTI_CONSTRAINT,
            ResourceDemand.HIGH,
        ),
        "reasoning-ambiguous-constraints": (
            3,
            WorkloadGroup.AMBIGUOUS_REQUEST,
            ResourceDemand.MEDIUM,
        ),
        "reasoning-multi-step-proof": (
            5,
            WorkloadGroup.MULTI_STEP_REASONING,
            ResourceDemand.HIGH,
        ),
        "summary-long-document": (
            5,
            WorkloadGroup.LONG_DOCUMENT_SUMMARIZATION,
            ResourceDemand.HIGH,
        ),
        "summary-resource-constrained": (
            1,
            WorkloadGroup.RESOURCE_CONSTRAINED,
            ResourceDemand.LOW,
        ),
        "qa-simple-fact": (1, WorkloadGroup.SIMPLE_FACTUAL, ResourceDemand.LOW),
        "qa-underspecified-request": (
            3,
            WorkloadGroup.AMBIGUOUS_REQUEST,
            ResourceDemand.MEDIUM,
        ),
        "creative-multi-constraint": (
            5,
            WorkloadGroup.COMPLEX_MULTI_CONSTRAINT,
            ResourceDemand.HIGH,
        ),
        "creative-resource-constrained": (
            1,
            WorkloadGroup.RESOURCE_CONSTRAINED,
            ResourceDemand.LOW,
        ),
        "data-structured-analysis": (
            4,
            WorkloadGroup.STRUCTURED_DATA_ANALYSIS,
            ResourceDemand.HIGH,
        ),
        "data-higher-tier-regression": (
            2,
            WorkloadGroup.STRUCTURED_DATA_ANALYSIS,
            ResourceDemand.LOW,
        ),
        "general-resource-constrained": (
            1,
            WorkloadGroup.RESOURCE_CONSTRAINED,
            ResourceDemand.LOW,
        ),
        "general-ambiguous-multi-constraint": (
            4,
            WorkloadGroup.AMBIGUOUS_REQUEST,
            ResourceDemand.HIGH,
        ),
    }
    _validate_scenario_metadata(
        (entry.case_id for entry in dataset),
        scenario_metadata,
    )
    scenarios = tuple(
        BenchmarkDatasetEntry(
            case_id=entry.case_id,
            prompt=entry.prompt,
            benchmark_case=entry.benchmark_case,
            complexity_level=scenario_metadata[entry.case_id][0],
            workload_group=scenario_metadata[entry.case_id][1],
            resource_demand=scenario_metadata[entry.case_id][2],
        )
        for entry in dataset
    )
    validate_benchmark_dataset(scenarios)
    return scenarios
