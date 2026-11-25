"""Explicit OpenAI empirical benchmark; running it may incur API charges."""

from tokrix.execution.providers import OpenAIProvider
from tokrix.routing import (
    BenchmarkTarget,
    EmpiricalBenchmarkCase,
    EmpiricalBenchmarkConfig,
    run_empirical_benchmark,
)

MODEL_NAME = "gpt-4o-mini"


def main() -> None:
    provider = OpenAIProvider(model=MODEL_NAME)
    config = EmpiricalBenchmarkConfig(
        cases=(
            EmpiricalBenchmarkCase(
                case_id="short-factual",
                prompt="In one sentence, what is a hash table?",
            ),
            EmpiricalBenchmarkCase(
                case_id="short-summary",
                prompt="Summarize: Reliable systems validate inputs and report errors.",
            ),
        ),
        targets=(
            BenchmarkTarget(
                target_id=f"openai:{MODEL_NAME}",
                provider=provider,
                model_name=MODEL_NAME,
            ),
        ),
        repetitions=1,
    )
    result = run_empirical_benchmark(config)

    for summary in result.providers:
        print(f"Target: {summary.target_id}")
        print(f"Successful/failed: {summary.successful_runs}/{summary.failed_runs}")
        print(f"Average latency (ms): {summary.latency.average}")
        print(f"Average total tokens: {summary.total_tokens.average}")
        print(f"Provider-reported cost: {summary.provider_reported_cost.average}")
        print(f"Estimated cost: {summary.estimated_cost.average}")
        print(f"Average evaluated quality: {summary.quality.average}")
        print("None means the provider/evaluator did not supply that metric.")


if __name__ == "__main__":
    main()
