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