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

Offline routing benchmarks
--------------------------

`tokrix.routing.benchmark.evaluate_routing` compares static routing with the
existing adaptive recommendation policy using controlled benchmark cases.
Historical feedback is used only for adaptive recommendations; separately
provided expected outcomes score the selected tiers. Evaluation is offline and
does not execute models or make provider calls. Its LOW/MEDIUM/HIGH cost proxies
(1/2/3) are relative units, not pricing or token-cost estimates, and benchmark
results are not claims about real-world model performance.