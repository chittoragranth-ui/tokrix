"""Minimal example of Tokrix's complete prompt pipeline."""

from tokrix.budget.adaptive import BudgetMode
from tokrix.execution.providers import OpenAIProvider
from tokrix.pipeline import TokrixPipeline
from tokrix.routing import ModelTier

# Change these model names to use different models for each routed tier.
LOW_MODEL = "gpt-4o-mini"
MEDIUM_MODEL = "gpt-4o"
HIGH_MODEL = "gpt-4.1"

PROMPT = "Explain the difference between a list and a dictionary in Python."


def main() -> None:
    providers = {
        ModelTier.LOW: OpenAIProvider(model=LOW_MODEL),
        ModelTier.MEDIUM: OpenAIProvider(model=MEDIUM_MODEL),
        ModelTier.HIGH: OpenAIProvider(model=HIGH_MODEL),
    }
    pipeline = TokrixPipeline(providers)
    result = pipeline.run(PROMPT, mode=BudgetMode.BALANCED)

    print("Response:")
    print(result.response.content)
    print()
    print("Pipeline metadata:")
    print(f"Task classification: {result.classification_result.task_type.value}")
    print(f"Complexity score: {result.complexity_result.score}")
    print(f"Estimated tokens: {result.estimated_tokens}")
    print(f"Budget score: {result.budget_decision.budget_score}")
    print(f"Selected model tier: {result.selected_tier.value}")


if __name__ == "__main__":
    main()
