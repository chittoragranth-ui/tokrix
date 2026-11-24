"""Unified orchestration for Tokrix prompt analysis and execution."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tokrix.budget.adaptive import BudgetDecision, BudgetMode, allocate_budget
from tokrix.decision.engine import Action, DecisionResult, decide_action
from tokrix.execution.engine import (
    ExecutionEngine,
    ExecutionRequest,
    ExecutionResponse,
    ModelProvider,
)
from tokrix.intelligence.classifier import (
    ClassificationResult,
    classify_task,
)
from tokrix.intelligence.complexity import ComplexityResult, analyze_complexity
from tokrix.intelligence.token_estimator import estimate_tokens
from tokrix.routing.adaptive import (
    AdaptiveRoutingExplanation,
    AdaptiveRoutingPolicy,
    AdaptiveRoutingReason,
    AdaptiveRoutingResult,
    recommend_tier,
)
from tokrix.routing.router import ModelTier, RoutingDecision, route_model
from tokrix.verification.verifier import VerificationResult, verify_response

if TYPE_CHECKING:
    from tokrix.feedback.collector import FeedbackRecord


class _TimedProvider:
    """Measure only the underlying provider's execute call."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        self.execution_latency_ms = 0.0

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        started = time.perf_counter()
        try:
            return self._provider.execute(request)
        finally:
            self.execution_latency_ms = max(
                0.0,
                (time.perf_counter() - started) * 1000,
            )


@dataclass
class PipelineAttempt:
    """One model execution attempt within a pipeline run."""

    attempt_number: int
    model_tier: ModelTier
    verification_result: VerificationResult
    decision: DecisionResult
    execution_latency_ms: float = 0.0

    @property
    def decision_taken(self) -> Action:
        """The action selected for this attempt."""
        return self.decision.action

    @property
    def decision_reason(self) -> str:
        """The explanation for this attempt's decision."""
        return self.decision.reason


@dataclass
class PipelineResult:
    """Structured decision data and response returned by a pipeline run."""

    response: ExecutionResponse
    complexity_result: ComplexityResult
    classification_result: ClassificationResult
    estimated_tokens: int
    budget_decision: BudgetDecision
    routing_decision: RoutingDecision
    selected_tier: ModelTier
    verification_result: VerificationResult | None = None
    attempt_history: list[PipelineAttempt] | None = None
    final_decision: DecisionResult | None = None
    total_duration_ms: float = 0.0
    adaptive_routing_result: AdaptiveRoutingResult | None = None
    adaptive_routing_explanation: AdaptiveRoutingExplanation | None = None

    @property
    def execution_response(self) -> ExecutionResponse:
        """Alias for the response field."""
        return self.response

    @property
    def complexity(self) -> ComplexityResult:
        """Alias for the complexity analysis field."""
        return self.complexity_result

    @property
    def classification(self) -> ClassificationResult:
        """Alias for the classification field."""
        return self.classification_result

    @property
    def tier(self) -> ModelTier:
        """Alias for the selected model tier."""
        return self.selected_tier

    @property
    def final_tier(self) -> ModelTier:
        """Final tier that was selected or executed."""
        return self.selected_tier

    @property
    def recommended_tier(self) -> ModelTier:
        """Tier recommended before execution, prior to any escalation."""
        if self.adaptive_routing_result is None:
            return self.routing_decision.tier
        return self.adaptive_routing_result.recommended_tier

    @property
    def final_verification(self) -> VerificationResult | None:
        """Verification result from the final execution attempt."""
        return self.verification_result


class _FeedbackStore(Protocol):
    """Feedback lifecycle consumed by a pipeline instance."""

    def snapshot(self) -> Iterable[FeedbackRecord]: ...

    def record(self, result: PipelineResult) -> None: ...


class TokrixPipeline:
    """Analyze, allocate, route, and execute a single prompt."""

    def __init__(
        self,
        providers: Mapping[ModelTier, ModelProvider],
        feedback_collector: _FeedbackStore | None = None,
        adaptive_routing_enabled: bool = True,
        adaptive_policy: AdaptiveRoutingPolicy | None = None,
    ) -> None:
        if not isinstance(providers, Mapping):
            raise TypeError("providers must be a mapping from ModelTier to providers")
        if feedback_collector is not None and (
            not callable(getattr(feedback_collector, "snapshot", None))
            or not callable(getattr(feedback_collector, "record", None))
        ):
            raise TypeError(
                "feedback_collector must implement snapshot and record"
            )
        if not isinstance(adaptive_routing_enabled, bool):
            raise TypeError("adaptive_routing_enabled must be a bool")
        if adaptive_policy is not None and not isinstance(
            adaptive_policy, AdaptiveRoutingPolicy
        ):
            raise TypeError("adaptive_policy must be an AdaptiveRoutingPolicy or None")

        validated: dict[ModelTier, ModelProvider] = {}
        for tier, provider in providers.items():
            if not isinstance(tier, ModelTier):
                raise TypeError("provider mapping keys must be ModelTier values")
            if not callable(getattr(provider, "execute", None)):
                raise TypeError(
                    f"provider for {tier.value} tier must implement execute"
                )
            validated[tier] = provider
        self._providers = validated
        self._feedback_collector = feedback_collector
        self._adaptive_routing_enabled = adaptive_routing_enabled
        self._adaptive_policy = adaptive_policy or AdaptiveRoutingPolicy()

    def run(
        self,
        prompt: str,
        mode: BudgetMode = BudgetMode.BALANCED,
        max_retries: int = 1,
    ) -> PipelineResult:
        """Run the complete deterministic Tokrix pipeline for a prompt."""
        pipeline_started = time.perf_counter()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        if not isinstance(mode, BudgetMode):
            raise TypeError("mode must be a BudgetMode")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        complexity = analyze_complexity(prompt)
        classification = classify_task(prompt)
        estimated_tokens = estimate_tokens(prompt)
        budget = allocate_budget(
            complexity_score=complexity.score,
            task_type=classification.task_type,
            estimated_tokens=estimated_tokens,
            mode=mode,
        )
        routing = route_model(budget)
        adaptive_routing_result = None
        if not self._adaptive_routing_enabled:
            explanation = AdaptiveRoutingExplanation(
                original_tier=routing.tier,
                recommended_tier=routing.tier,
                task_type=classification.task_type,
                adaptive_routing_enabled=False,
                feedback_available=self._feedback_collector is not None,
                evidence_sufficient=None,
                tier_sample_counts={},
                tier_success_rates={},
                policy=self._adaptive_policy,
                upgrade_made=False,
                reason_code=AdaptiveRoutingReason.ADAPTIVE_ROUTING_DISABLED,
                reason="Adaptive routing is disabled; static routing is used.",
            )
        elif self._feedback_collector is None:
            explanation = AdaptiveRoutingExplanation(
                original_tier=routing.tier,
                recommended_tier=routing.tier,
                task_type=classification.task_type,
                adaptive_routing_enabled=True,
                feedback_available=False,
                evidence_sufficient=None,
                tier_sample_counts={},
                tier_success_rates={},
                policy=self._adaptive_policy,
                upgrade_made=False,
                reason_code=AdaptiveRoutingReason.FEEDBACK_COLLECTOR_UNAVAILABLE,
                reason=(
                    "No feedback collector is configured; static routing is used."
                ),
            )
        else:
            historical_feedback = self._feedback_collector.snapshot()
            adaptive_routing_result = recommend_tier(
                routing,
                classification.task_type,
                historical_feedback,
                self._adaptive_policy,
            )
            explanation = AdaptiveRoutingExplanation(
                original_tier=routing.tier,
                recommended_tier=adaptive_routing_result.recommended_tier,
                task_type=classification.task_type,
                adaptive_routing_enabled=True,
                feedback_available=True,
                evidence_sufficient=adaptive_routing_result.evidence_sufficient,
                tier_sample_counts=adaptive_routing_result.tier_sample_counts,
                tier_success_rates=adaptive_routing_result.tier_success_rates,
                policy=self._adaptive_policy,
                upgrade_made=adaptive_routing_result.adjusted,
                reason_code=adaptive_routing_result.reason_code,
                reason=adaptive_routing_result.reason,
            )
        tier = (
            adaptive_routing_result.recommended_tier
            if adaptive_routing_result is not None
            else routing.tier
        )

        provider = self._providers.get(tier)
        if provider is None:
            raise ValueError(
                f"No provider configured for the {tier.value.upper()} model tier"
            )

        attempt_history: list[PipelineAttempt] = []
        retry_count = 0
        final_decision: DecisionResult | None = None
        response: ExecutionResponse
        verification: VerificationResult

        while True:
            timed_provider = _TimedProvider(provider)
            response = ExecutionEngine(timed_provider).execute(
                ExecutionRequest(prompt=prompt)
            )
            verification = verify_response(response)
            decision = decide_action(
                verification,
                tier,
                retry_count=retry_count,
                max_retries=max_retries,
            )
            attempt_history.append(
                PipelineAttempt(
                    attempt_number=len(attempt_history) + 1,
                    model_tier=tier,
                    verification_result=verification,
                    decision=decision,
                    execution_latency_ms=timed_provider.execution_latency_ms,
                )
            )

            if decision.action is Action.ACCEPT:
                final_decision = decision
                break
            if decision.action is Action.RETRY:
                retry_count += 1
                continue
            if decision.action is Action.ESCALATE:
                next_tier = decision.next_tier
                if next_tier is None:
                    final_decision = DecisionResult(
                        action=Action.STOP,
                        current_tier=tier,
                        next_tier=None,
                        reason=f"Escalation from {tier.value.upper()} stopped.",
                    )
                    break
                next_provider = self._providers.get(next_tier)
                if next_provider is None:
                    final_decision = DecisionResult(
                        action=Action.STOP,
                        current_tier=tier,
                        next_tier=None,
                        reason=(
                            f"Escalation from {tier.value.upper()} to "
                            f"{next_tier.value.upper()} stopped because no "
                            f"{next_tier.value.upper()} provider is configured."
                        ),
                    )
                    break
                tier = next_tier
                provider = next_provider
                retry_count = 0
                continue

            final_decision = decision
            break

        total_duration_ms = max(
            0.0,
            (time.perf_counter() - pipeline_started) * 1000,
        )
        result = PipelineResult(
            response=response,
            complexity_result=complexity,
            classification_result=classification,
            estimated_tokens=estimated_tokens,
            budget_decision=budget,
            routing_decision=routing,
            selected_tier=tier,
            verification_result=verification,
            attempt_history=attempt_history,
            final_decision=final_decision,
            total_duration_ms=total_duration_ms,
            adaptive_routing_result=adaptive_routing_result,
            adaptive_routing_explanation=explanation,
        )
        if self._feedback_collector is not None:
            self._feedback_collector.record(result)
            result.total_duration_ms = max(
                0.0,
                (time.perf_counter() - pipeline_started) * 1000,
            )
        return result