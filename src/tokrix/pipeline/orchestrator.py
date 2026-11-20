"""Unified orchestration for Tokrix prompt analysis and execution."""

from collections.abc import Mapping
from dataclasses import dataclass

from tokrix.budget.adaptive import BudgetDecision, BudgetMode, allocate_budget
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
from tokrix.routing.router import ModelTier, RoutingDecision, route_model


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


class TokrixPipeline:
    """Analyze, allocate, route, and execute a single prompt."""

    def __init__(self, providers: Mapping[ModelTier, ModelProvider]) -> None:
        if not isinstance(providers, Mapping):
            raise TypeError("providers must be a mapping from ModelTier to providers")

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

    def run(
        self,
        prompt: str,
        mode: BudgetMode = BudgetMode.BALANCED,
    ) -> PipelineResult:
        """Run the complete deterministic Tokrix pipeline for a prompt."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        if not isinstance(mode, BudgetMode):
            raise TypeError("mode must be a BudgetMode")

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
        tier = routing.tier

        provider = self._providers.get(tier)
        if provider is None:
            raise ValueError(
                f"No provider configured for the {tier.value.upper()} model tier"
            )

        response = ExecutionEngine(provider).execute(
            ExecutionRequest(prompt=prompt)
        )
        return PipelineResult(
            response=response,
            complexity_result=complexity,
            classification_result=classification,
            estimated_tokens=estimated_tokens,
            budget_decision=budget,
            routing_decision=routing,
            selected_tier=tier,
        )