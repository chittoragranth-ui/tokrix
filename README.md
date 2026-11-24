Tokrix
====

Adaptive inference and resource allocation infrastructure for Python.

Basic pipeline example
----------------------

Install the project and the optional OpenAI provider dependency:

```bash
python -m pip install -e ".[openai]"
```

Set the API key using the environment variable used by the official OpenAI SDK:

```bash
export OPENAI_API_KEY="your-api-key"
```

Run the example from the project root:

```bash
python examples/basic_pipeline.py
```

The model names in `examples/basic_pipeline.py` are constants and can be changed
to models available to your OpenAI account.

Feedback-driven adaptive routing
--------------------------------

Adaptive routing uses historical, task-specific verification outcomes when a
`FeedbackCollector` is configured. Its default policy requires at least five
samples per tier, limits the current tier's success rate to 60%, requires a
higher-tier success rate of at least 80%, and requires at least a 20 percentage
point improvement. Recommendations only upgrade by one tier; LOW and MEDIUM may
upgrade, HIGH cannot, and routing never downgrades. Without a feedback
collector, the pipeline uses static routing.

The immutable `AdaptiveRoutingPolicy` exposes those thresholds. Pass it as
`adaptive_policy`. Adaptive routing is enabled by default when a collector is
present; set `adaptive_routing_enabled=False` to retain static tier selection
while still collecting completed-run feedback:

```python
from fractions import Fraction

from tokrix.feedback import FeedbackCollector
from tokrix.pipeline import TokrixPipeline
from tokrix.routing import AdaptiveRoutingPolicy

feedback = FeedbackCollector()
policy = AdaptiveRoutingPolicy(
    minimum_samples_per_tier=8,
    max_current_success_rate=Fraction(1, 2),
    min_higher_success_rate=Fraction(9, 10),
    min_success_rate_improvement=Fraction(3, 10),
)
pipeline = TokrixPipeline(
    providers,
    feedback_collector=feedback,
    adaptive_policy=policy,
)
result = pipeline.run("Explain photosynthesis.")

print(result.routing_decision.tier)  # Original budget-based tier
print(result.recommended_tier)  # Adaptive recommendation
print(result.selected_tier)  # Tier actually used after any escalation
```

To disable adaptive recommendations while keeping feedback collection:

```python
pipeline = TokrixPipeline(
    providers,
    feedback_collector=feedback,
    adaptive_routing_enabled=False,
)
```

The pipeline snapshots existing feedback before routing, then records the
completed run's per-attempt feedback afterward. A request therefore cannot use
its own outcome to choose its initial tier. With adaptive routing disabled, the
original static routing tier is used and no recommendation is computed, but an
available collector still records the run. Insufficient task-specific evidence
retains the static tier. The collector stores feedback only in memory.

Offline routing benchmarks
--------------------------

The offline routing benchmark compares static routing with Tokrix's adaptive
routing policy using a deterministic, built-in dataset of synthetic cases. It
uses historical feedback as input to adaptive recommendations and controlled
expected outcomes to score the selected tiers. It does not execute models,
make provider calls, or require API keys.

From the repository root, install Tokrix in editable mode and run the example:

```bash
python3 -m pip install -e .
python3 examples/routing_benchmark.py
```

The report includes verification success rates for static and adaptive routing,
their difference in percentage points, average tier-cost proxies, relative cost
change, and changes in LOW/MEDIUM/HIGH tier counts. It also counts cases where
adaptive verification improved, degraded, or stayed unchanged compared with
static routing; the outcome percentages use all evaluated cases as their
denominator.

All reported differences are adaptive minus static. A positive success-rate
difference means adaptive routing succeeded more often; a negative one means it
succeeded less often. A positive cost difference means a higher relative tier
cost, while a negative one means a lower relative tier cost. The tier-cost
proxy assigns LOW/MEDIUM/HIGH the relative values 1/2/3; it is not provider
pricing, token usage, or a monetary estimate.

The expected outcomes are synthetic benchmark data, not measurements from real
models. Treat the results as a reproducible demonstration of the evaluation
workflow, not as evidence or a prediction of real-world model performance.