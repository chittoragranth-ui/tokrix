from dataclasses import FrozenInstanceError

import pytest

from tokrix.intelligence.classifier import TaskType
from tokrix.routing import (
    BenchmarkDatasetEntry,
    ModelTier,
    evaluate_routing,
    get_benchmark_dataset,
)


def test_dataset_has_fifteen_entries_with_unique_nonempty_ids() -> None:
    dataset = get_benchmark_dataset()

    assert len(dataset) == 15
    case_ids = [entry.case_id for entry in dataset]
    assert all(case_id.strip() for case_id in case_ids)
    assert len(set(case_ids)) == len(case_ids)


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
