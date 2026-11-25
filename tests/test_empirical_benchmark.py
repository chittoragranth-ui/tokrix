from dataclasses import FrozenInstanceError

import pytest

from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.routing import (
    MAX_BENCHMARK_REPETITIONS,
    BenchmarkTarget,
    EmpiricalBenchmarkCase,
    EmpiricalBenchmarkConfig,
    ModelTier,
    TokenPricing,
    empirical,
    run_empirical_benchmark,
)


class FakeProvider:
    def __init__(self, outcomes: list[ExecutionResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        self.calls.append(request)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _case(
    case_id: str = "case-a",
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> EmpiricalBenchmarkCase:
    return EmpiricalBenchmarkCase(
        case_id=case_id,
        prompt=f"Prompt for {case_id}",
        metadata=metadata,
    )


def _target(
    target_id: str,
    provider: FakeProvider,
    *,
    tier: ModelTier = ModelTier.LOW,
    model_name: str = "test-model",
    pricing: TokenPricing | None = None,
) -> BenchmarkTarget:
    return BenchmarkTarget(
        target_id=target_id,
        provider=provider,
        model_name=model_name,
        tier=tier,
        pricing=pricing,
    )


def test_case_metadata_is_defensively_copied_and_immutable() -> None:
    metadata = {"suite": "smoke", "index": 1}
    case = _case(metadata=metadata)
    metadata["suite"] = "changed"

    assert case.metadata is not None
    assert case.metadata["suite"] == "smoke"
    with pytest.raises(TypeError):
        case.metadata["suite"] = "mutated"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        case.case_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"case_id": ""}, ValueError, "case_id must not be empty"),
        ({"case_id": 1}, TypeError, "case_id must be a string"),
        ({"prompt": ""}, ValueError, "prompt must not be empty"),
        ({"prompt": None}, TypeError, "prompt must be a string"),
        ({"evaluation_context": 1}, TypeError, "evaluation_context"),
        ({"metadata": []}, TypeError, "metadata must be a mapping"),
        ({"metadata": {1: "invalid"}}, TypeError, "keys must be strings"),
        ({"metadata": {"invalid": object()}}, TypeError, "scalar values"),
        ({"metadata": {"invalid": float("inf")}}, ValueError, "must be finite"),
    ],
)
def test_empirical_case_rejects_malformed_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "case_id": "valid",
        "prompt": "A valid prompt",
    }
    values.update(kwargs)

    with pytest.raises(error, match=message):
        EmpiricalBenchmarkCase(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("input_cost", "output_cost", "error", "message"),
    [
        (True, 1, TypeError, "must be a number"),
        (float("nan"), 1, ValueError, "finite and non-negative"),
        (-1, 1, ValueError, "finite and non-negative"),
    ],
)
def test_token_pricing_rejects_invalid_values(
    input_cost: float,
    output_cost: float,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        TokenPricing(input_cost, output_cost)


def test_benchmark_config_snapshots_lists_without_calling_providers() -> None:
    provider = FakeProvider([ExecutionResponse("unused")])
    cases = [_case()]
    targets = [_target("provider-a", provider)]

    config = EmpiricalBenchmarkConfig(cases, targets)  # type: ignore[arg-type]
    cases.clear()
    targets.clear()

    assert config.cases == (_case(),)
    assert len(config.targets) == 1
    assert provider.calls == []
    with pytest.raises(FrozenInstanceError):
        config.repetitions = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"repetitions": True}, TypeError, "repetitions must be an integer"),
        ({"repetitions": 0}, ValueError, "between 1 and"),
        (
            {"repetitions": MAX_BENCHMARK_REPETITIONS + 1},
            ValueError,
            "between 1 and",
        ),
        ({"cases": [object()]}, TypeError, "EmpiricalBenchmarkCase"),
        ({"targets": [object()]}, TypeError, "BenchmarkTarget"),
        ({"targets": []}, ValueError, "at least one benchmark target"),
        ({"quality_evaluator": object()}, TypeError, "must be callable"),
    ],
)
def test_benchmark_config_rejects_invalid_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    provider = FakeProvider([ExecutionResponse("unused")])
    values: dict[str, object] = {
        "cases": (_case(),),
        "targets": (_target("provider-a", provider),),
    }
    values.update(kwargs)

    with pytest.raises(error, match=message):
        EmpiricalBenchmarkConfig(**values)  # type: ignore[arg-type]


def test_benchmark_config_rejects_duplicate_case_and_target_ids() -> None:
    provider = FakeProvider([ExecutionResponse("unused")])
    target = _target("provider-a", provider)

    with pytest.raises(ValueError, match="case identifiers must be unique"):
        EmpiricalBenchmarkConfig((_case(), _case()), (target,))
    with pytest.raises(ValueError, match="target identifiers must be unique"):
        EmpiricalBenchmarkConfig(
            (_case(),),
            (target, _target("provider-a", provider)),
        )


def test_target_requires_provider_and_valid_tier() -> None:
    with pytest.raises(TypeError, match="callable execute"):
        BenchmarkTarget("invalid", object())
    with pytest.raises(TypeError, match="tier must be a ModelTier"):
        BenchmarkTarget(
            "invalid-tier",
            FakeProvider([ExecutionResponse("unused")]),
            tier="low",  # type: ignore[arg-type]
        )


def test_runner_records_latency_usage_cost_quality_and_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        [
            ExecutionResponse(
                "first",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                provider_reported_cost=0.005,
            ),
            ExecutionResponse(
                "second",
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                provider_reported_cost=0.01,
            ),
        ]
    )
    target = _target(
        "provider-a",
        provider,
        pricing=TokenPricing(10, 20),
    )
    timestamps = iter([1_000_000, 3_000_000, 5_000_000, 8_000_000])
    monkeypatch.setattr(empirical.time, "perf_counter_ns", lambda: next(timestamps))
    config = EmpiricalBenchmarkConfig(
        (_case(),),
        (target,),
        repetitions=2,
        quality_evaluator=lambda case, response: len(response.content) * 10,
    )

    result = run_empirical_benchmark(config)

    summary = result.providers[0]
    assert len(provider.calls) == 2
    assert result.case_count == 1
    assert result.repetitions == 2
    assert [run.repetition for run in summary.runs] == [1, 2]
    assert [run.latency_ms for run in summary.runs] == [2.0, 3.0]
    assert all(run.execution_succeeded for run in summary.runs)
    assert [run.quality_score for run in summary.runs] == [50.0, 60.0]
    assert summary.latency.average == 2.5
    assert summary.input_tokens.average == 150
    assert summary.output_tokens.average == 75
    assert summary.total_tokens.average == 225
    assert summary.provider_reported_cost.average == pytest.approx(0.0075)
    assert summary.estimated_cost.average == pytest.approx(0.003)
    assert summary.quality.average == 55.0
    assert all(run.target_id == "provider-a" for run in summary.runs)
    assert all(run.model_name == "test-model" for run in summary.runs)
    assert all(run.tier is ModelTier.LOW for run in summary.runs)


def test_runner_records_provider_failure_and_missing_metrics() -> None:
    provider = FakeProvider(
        [
            RuntimeError("provider unavailable"),
            ExecutionResponse("partial usage", input_tokens=12),
        ]
    )
    config = EmpiricalBenchmarkConfig(
        (_case(), _case("case-b")),
        (_target("provider-a", provider),),
        quality_evaluator=lambda case, response: 75,
    )

    summary = run_empirical_benchmark(config).providers[0]

    assert summary.failed_runs == 1
    assert summary.successful_runs == 1
    assert summary.runs[0].error_type == "RuntimeError"
    assert summary.runs[0].error_message == "provider unavailable"
    assert summary.runs[0].quality_score is None
    assert summary.runs[1].execution_succeeded
    assert summary.runs[1].input_tokens == 12
    assert summary.runs[1].provider_reported_cost is None
    assert summary.input_tokens.average == 12
    assert summary.input_tokens.available_count == 1
    assert summary.input_tokens.total_count == 2
    assert summary.quality.average == 75.0
    assert summary.quality.available_count == 1
    assert summary.quality.total_count == 2
    assert summary.estimated_cost.average is None


@pytest.mark.parametrize(
    ("score", "error_text"),
    [
        (True, "TypeError: quality evaluator must return a numeric score"),
        (-1, "ValueError: quality score must be finite and between 0 and 100"),
        (101, "ValueError: quality score must be finite and between 0 and 100"),
        (float("nan"), "ValueError: quality score must be finite and between 0 and 100"),
    ],
)
def test_invalid_quality_scores_remain_explicit_and_unavailable(
    score: object,
    error_text: str,
) -> None:
    provider = FakeProvider([ExecutionResponse("response")])
    result = run_empirical_benchmark(
        EmpiricalBenchmarkConfig(
            (_case(),),
            (_target("provider-a", provider),),
            quality_evaluator=lambda case, response: score,  # type: ignore[return-value]
        )
    )

    summary = result.providers[0]
    assert summary.runs[0].quality_score is None
    assert summary.runs[0].quality_error == error_text
    assert summary.quality.average is None
    assert summary.quality.available_count == 0


def test_quality_evaluator_exception_is_separate_from_execution_failure() -> None:
    provider = FakeProvider([ExecutionResponse("response")])

    def fail_quality(case: EmpiricalBenchmarkCase, response: ExecutionResponse) -> int:
        raise RuntimeError("evaluation unavailable")

    summary = run_empirical_benchmark(
        EmpiricalBenchmarkConfig(
            (_case(),),
            (_target("provider-a", provider),),
            quality_evaluator=fail_quality,
        )
    ).providers[0]

    assert summary.successful_runs == 1
    assert summary.failed_runs == 0
    assert summary.runs[0].quality_error == "RuntimeError: evaluation unavailable"


def test_empty_cases_and_provider_comparison_preserve_availability() -> None:
    first = FakeProvider([ExecutionResponse("unused")])
    second = FakeProvider([ExecutionResponse("unused")])
    result = run_empirical_benchmark(
        EmpiricalBenchmarkConfig(
            (),
            (
                _target("provider-a", first),
                _target("provider-b", second, tier=ModelTier.MEDIUM),
            ),
        )
    )

    assert result.case_count == 0
    assert len(result.providers) == 2
    for summary in result.providers:
        assert summary.runs == ()
        assert summary.successful_runs == 0
        assert summary.failed_runs == 0
        assert summary.latency.average is None
        assert summary.latency.available_count == 0
        assert summary.latency.total_count == 0
        assert summary.provider_reported_cost.average is None
        assert summary.quality.average is None
    assert first.calls == []
    assert second.calls == []


def test_provider_summaries_compare_independent_targets() -> None:
    first = FakeProvider(
        [ExecutionResponse("first", input_tokens=10, output_tokens=5)]
    )
    second = FakeProvider(
        [ExecutionResponse("second", input_tokens=20, output_tokens=10)]
    )
    result = run_empirical_benchmark(
        EmpiricalBenchmarkConfig(
            (_case(),),
            (
                _target("provider-a", first, tier=ModelTier.LOW),
                _target("provider-b", second, tier=ModelTier.HIGH),
            ),
        )
    )

    by_target = {summary.target_id: summary for summary in result.providers}
    assert by_target["provider-a"].input_tokens.average == 10
    assert by_target["provider-a"].total_tokens.average is None
    assert by_target["provider-b"].input_tokens.average == 20
    assert by_target["provider-a"].runs[0].tier is ModelTier.LOW
    assert by_target["provider-b"].runs[0].tier is ModelTier.HIGH


def test_runner_rejects_invalid_configuration_type() -> None:
    with pytest.raises(TypeError, match="EmpiricalBenchmarkConfig"):
        run_empirical_benchmark(object())  # type: ignore[arg-type]


def test_response_rejects_invalid_usage_and_provider_cost() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        ExecutionResponse("response", input_tokens=-1)
    with pytest.raises(ValueError, match="provider_reported_cost"):
        ExecutionResponse("response", provider_reported_cost=float("inf"))
