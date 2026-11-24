from dataclasses import FrozenInstanceError, replace

import pytest

from tokrix.intelligence.classifier import TaskType
from tokrix.routing import (
    BenchmarkDatasetEntry,
    ModelTier,
    ResourceDemand,
    WorkloadGroup,
    evaluate_routing,
    get_benchmark_dataset,
    validate_benchmark_dataset,
)
from tokrix.routing import dataset as dataset_module


def test_dataset_has_unique_nonempty_ids_and_full_task_coverage() -> None:
    dataset = get_benchmark_dataset()

    assert len(dataset) >= 25
    case_ids = [entry.case_id for entry in dataset]
    assert all(case_id.strip() for case_id in case_ids)
    assert len(set(case_ids)) == len(case_ids)
    assert {entry.task_type for entry in dataset} == set(TaskType)


def test_every_entry_has_a_nonempty_prompt_and_benchmark_case() -> None:
    for entry in get_benchmark_dataset():
        assert isinstance(entry, BenchmarkDatasetEntry)
        assert isinstance(entry.prompt, str)
        assert entry.prompt.strip()
        assert entry.benchmark_case.expected_outcomes


def test_dataset_represents_every_task_type() -> None:
    represented = {
        entry.benchmark_case.task_type for entry in get_benchmark_dataset()
    }

    assert represented == set(TaskType)


def test_dataset_has_complexity_resource_and_workload_diversity() -> None:
    dataset = get_benchmark_dataset()

    assert len({entry.complexity_level for entry in dataset}) >= 4
    assert min(entry.complexity_level for entry in dataset) == 1
    assert max(entry.complexity_level for entry in dataset) == 5
    assert {entry.resource_demand for entry in dataset} == set(ResourceDemand)
    assert {entry.workload_group for entry in dataset} == set(WorkloadGroup)
    assert all(
        sum(entry.task_type is task_type for entry in dataset) >= 3
        for task_type in TaskType
    )


def test_scenario_model_validates_metadata() -> None:
    case = get_benchmark_dataset()[0].benchmark_case

    with pytest.raises(TypeError, match="complexity_level"):
        BenchmarkDatasetEntry(
            "bad-complexity",
            "Describe a workload.",
            case,
            complexity_level=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="between 1 and 5"):
        BenchmarkDatasetEntry(
            "bad-complexity",
            "Describe a workload.",
            case,
            complexity_level=6,
        )
    with pytest.raises(TypeError, match="WorkloadGroup"):
        BenchmarkDatasetEntry(
            "bad-group",
            "Describe a workload.",
            case,
            workload_group="coding",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="ResourceDemand"):
        BenchmarkDatasetEntry(
            "bad-demand",
            "Describe a workload.",
            case,
            resource_demand="high",  # type: ignore[arg-type]
        )


def test_dataset_validator_rejects_empty_and_duplicate_identifiers() -> None:
    dataset = get_benchmark_dataset()

    with pytest.raises(ValueError, match="must not be empty"):
        validate_benchmark_dataset(())
    duplicate = BenchmarkDatasetEntry(
        dataset[0].case_id,
        "A distinct prompt.",
        dataset[1].benchmark_case,
        dataset[1].complexity_level,
        dataset[1].workload_group,
        dataset[1].resource_demand,
    )
    with pytest.raises(ValueError, match="must be unique"):
        validate_benchmark_dataset((*dataset, duplicate))


@pytest.mark.parametrize("entries", [None, "not a dataset"])
def test_dataset_validator_rejects_invalid_input_types(entries: object) -> None:
    with pytest.raises(TypeError, match="iterable"):
        validate_benchmark_dataset(entries)  # type: ignore[arg-type]


def test_dataset_validator_rejects_malformed_entries() -> None:
    with pytest.raises(TypeError, match="BenchmarkDatasetEntry"):
        validate_benchmark_dataset([object()])  # type: ignore[list-item]


def test_dataset_validator_rejects_missing_task_coverage() -> None:
    dataset = get_benchmark_dataset()
    incomplete = tuple(
        entry
        for entry in dataset
        if entry.task_type is not TaskType.CREATIVE_WRITING
    )

    with pytest.raises(ValueError, match="every TaskType"):
        validate_benchmark_dataset(incomplete)


def test_dataset_validator_rejects_missing_workload_group() -> None:
    dataset = get_benchmark_dataset()
    incomplete = tuple(
        replace(entry, workload_group=WorkloadGroup.DEBUGGING)
        if entry.workload_group is WorkloadGroup.CODE_GENERATION
        else entry
        for entry in dataset
    )

    assert {entry.task_type for entry in incomplete} == set(TaskType)
    assert len({entry.complexity_level for entry in incomplete}) >= 3
    assert {entry.resource_demand for entry in incomplete} == set(ResourceDemand)

    with pytest.raises(ValueError, match="every WorkloadGroup"):
        validate_benchmark_dataset(incomplete)


def test_dataset_validator_rejects_missing_resource_demand() -> None:
    dataset = get_benchmark_dataset()
    incomplete = tuple(
        replace(entry, resource_demand=ResourceDemand.LOW)
        if entry.resource_demand is ResourceDemand.MEDIUM
        else entry
        for entry in dataset
    )

    assert {entry.task_type for entry in incomplete} == set(TaskType)
    assert len({entry.complexity_level for entry in incomplete}) >= 3
    assert {entry.workload_group for entry in incomplete} == set(WorkloadGroup)

    with pytest.raises(ValueError, match="every ResourceDemand"):
        validate_benchmark_dataset(incomplete)


def test_dataset_validator_rejects_insufficient_complexity_diversity() -> None:
    dataset = tuple(
        replace(entry, complexity_level=3) for entry in get_benchmark_dataset()
    )

    with pytest.raises(ValueError, match="complexity levels"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize(
    ("scenario_ids", "metadata", "message"),
    [
        (("scenario-a",), {}, "missing=\\['scenario-a'\\]"),
        ((), {"unused": (1, WorkloadGroup.SIMPLE_FACTUAL, ResourceDemand.LOW)}, "unused"),
    ],
)
def test_scenario_metadata_must_match_scenario_ids(
    scenario_ids: tuple[str, ...],
    metadata: dict[str, tuple[int, WorkloadGroup, ResourceDemand]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataset_module._validate_scenario_metadata(scenario_ids, metadata)


def test_summary_all_success_has_long_document_scenario_metadata() -> None:
    entry = next(
        entry
        for entry in get_benchmark_dataset()
        if entry.case_id == "summary-all-success"
    )

    assert "long project update" in entry.prompt
    assert entry.workload_group is WorkloadGroup.LONG_DOCUMENT_SUMMARIZATION


def test_entries_and_dataset_are_immutable() -> None:
    dataset = get_benchmark_dataset()

    with pytest.raises(FrozenInstanceError):
        dataset[0].case_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        dataset[0] = dataset[1]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        dataset[0].benchmark_case.task_type = TaskType.GENERAL  # type: ignore[misc]


def test_dataset_retrieval_is_deterministic() -> None:
    first = get_benchmark_dataset()
    second = get_benchmark_dataset()

    assert first == second
    assert first is second
    assert [entry.case_id for entry in first] == [
        entry.case_id for entry in second
    ]


def test_each_case_has_a_boolean_outcome_for_every_tier() -> None:
    tiers = set(ModelTier)

    for entry in get_benchmark_dataset():
        outcomes = entry.benchmark_case.expected_outcomes
        assert set(outcomes) == tiers
        assert all(isinstance(outcomes[tier], bool) for tier in tiers)


def test_outcome_profiles_cover_distinct_tier_behaviors() -> None:
    profiles = {
        tuple(entry.expected_outcomes[tier] for tier in ModelTier)
        for entry in get_benchmark_dataset()
    }

    assert len(profiles) >= 6
    assert (True, False, False) in profiles  # LOW is sufficient; higher is worse.
    assert (False, True, True) in profiles  # MEDIUM provides a meaningful gain.
    assert (False, False, True) in profiles  # HIGH is necessary in this scenario.
    assert (True, True, True) in profiles  # Higher tiers add no benefit.


def test_each_case_is_compatible_with_the_existing_evaluator() -> None:
    for entry in get_benchmark_dataset():
        result = evaluate_routing([entry.benchmark_case])
        assert result.case_count == 1
        assert len(result.case_results) == 1


def test_evaluator_reports_valid_aggregate_metrics_for_dataset() -> None:
    dataset = get_benchmark_dataset()
    result = evaluate_routing(entry.benchmark_case for entry in dataset)

    assert result.case_count == len(dataset)
    for metrics in (result.static_metrics, result.adaptive_metrics):
        assert 0 <= metrics.verification_successes <= result.case_count
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 1.0 <= metrics.average_tier_cost_proxy <= 3.0
        assert set(metrics.tier_distribution) == set(ModelTier)
        assert sum(metrics.tier_distribution.values()) == result.case_count
    assert 0 <= result.adaptive_upgrade_count <= result.case_count
    assert 0.0 <= result.adaptive_upgrade_rate <= 1.0
    assert result.adaptive_upgrade_count < result.case_count


def test_adaptive_upgrades_have_mixed_controlled_outcomes() -> None:
    result = evaluate_routing(
        entry.benchmark_case for entry in get_benchmark_dataset()
    )

    upgrades = [
        case
        for case in result.case_results
        if case.adaptive_tier is not case.static_tier
    ]
    assert any(
        case.adaptive_verification_passed
        and not case.static_verification_passed
        for case in upgrades
    )
    assert any(
        not case.adaptive_verification_passed
        and case.static_verification_passed
        for case in upgrades
    )
    assert any(
        case.adaptive_verification_passed is case.static_verification_passed
        for case in upgrades
    )
