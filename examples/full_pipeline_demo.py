"""Demonstrate the full Tokrix pipeline with deterministic offline providers."""

from tokrix.budget import BudgetMode
from tokrix.execution import ExecutionRequest, ExecutionResponse
from tokrix.feedback import FeedbackCollector
from tokrix.pipeline import PipelineResult, TokrixPipeline
from tokrix.routing import AdaptiveRoutingPolicy, ModelTier
from tokrix.telemetry import MetricsAggregator

PROMPT = "What is photosynthesis?"
HISTORICAL_RUNS = 5


class DeterministicProvider:
    """Return one fixed response without network or provider dependencies."""

    def __init__(self, content: str) -> None:
        self._content = content

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        del request
        return ExecutionResponse(content=self._content)


def _show_run(label: str, result: PipelineResult) -> None:
    print(f"\n{label}")
    print(
        f"  task={result.classification_result.task_type.value}, "
        f"complexity={result.complexity_result.score}/10, "
        f"estimated_tokens={result.estimated_tokens}"
    )
    print(
        f"  budget={result.budget_decision.budget_score}/100, "
        f"static_tier={result.routing_decision.tier.value}"
    )
    print(
        f"  recommended_tier={result.recommended_tier.value}, "
        f"final_tier={result.selected_tier.value}"
    )
    for attempt in result.attempt_history or ():
        print(
            f"  attempt {attempt.attempt_number}: "
            f"{attempt.model_tier.value} "
            f"verification={'passed' if attempt.verification_result.passed else 'failed'}, "
            f"decision={attempt.decision.action.value}, "
            f"latency={attempt.execution_latency_ms:.3f} ms"
        )
    explanation = result.adaptive_routing_explanation
    if explanation is not None:
        print(
            f"  adaptive routing: {explanation.reason_code.value} "
            f"({explanation.reason})"
        )


def main() -> None:
    """Run a short deterministic workflow showing escalation and adaptation."""
    feedback = FeedbackCollector()
    telemetry = MetricsAggregator()
    pipeline = TokrixPipeline(
        {
            ModelTier.LOW: DeterministicProvider(""),
            ModelTier.MEDIUM: DeterministicProvider(
                "Photosynthesis converts light into chemical energy."
            ),
            ModelTier.HIGH: DeterministicProvider("Unused in this demonstration."),
        },
        feedback_collector=feedback,
        adaptive_policy=AdaptiveRoutingPolicy(),
    )

    print("Tokrix full pipeline demonstration (deterministic and offline)")
    print(f"Prompt: {PROMPT}")
    print(
        "The LOW provider returns an empty response to demonstrate verification "
        "failure, a same-tier retry, and escalation."
    )

    first_result: PipelineResult | None = None
    for _ in range(HISTORICAL_RUNS):
        result = pipeline.run(
            PROMPT,
            mode=BudgetMode.COST_OPTIMIZED,
            max_retries=1,
        )
        telemetry.record(result)
        if first_result is None:
            first_result = result

    if first_result is not None:
        _show_run("First run: static routing and escalation", first_result)

    result = pipeline.run(
        PROMPT,
        mode=BudgetMode.COST_OPTIMIZED,
        max_retries=1,
    )
    telemetry.record(result)
    _show_run("After feedback: adaptive routing", result)

    snapshot = telemetry.snapshot()
    print("\nTelemetry across the six demonstration runs")
    print(
        f"  runs={snapshot.total_runs}, accepted={snapshot.accepted_runs}, "
        f"retries={snapshot.total_retries}, escalations={snapshot.total_escalations}"
    )
    for tier in ModelTier:
        metrics = snapshot.tier_metrics[tier]
        print(
            f"  {tier.value}: attempts={metrics.total_attempts}, "
            f"success_rate={metrics.success_rate:.0%}"
        )

    print(f"\nFeedback records collected: {len(feedback.snapshot())}")
    if result.adaptive_routing_explanation is not None:
        print(
            "Task-specific sample counts: "
            f"{dict(result.adaptive_routing_explanation.tier_sample_counts)}"
        )


if __name__ == "__main__":
    main()
