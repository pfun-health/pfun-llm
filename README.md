# pfun-llm

LLM backend interfaces for pfun.

Provides a common interface for interacting with generative models across
multiple providers, with a pluggable backend architecture.

## Backends

| Backend     | Module                       | Configuration                              |
|-------------|------------------------------|--------------------------------------------|
| Ollama      | `pfun_llm.backend.ollama`    | `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_API_KEY` |
| Google      | `pfun_llm.backend.google`    | `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT_ID`, `GOOGLE_CLOUD_LOCATION` |
| Perplexity  | `pfun_llm.backend.perplexity`| `PERPLEXITY_API_KEY`                       |
| OpenAI      | `pfun_llm.backend.openai`    | `OPENAI_KEY`                               |

All settings are read from `pfun_common.settings` (environment variables or a
`.env` file). The "Ollama" backend is the primary production dependency.

## Usage

### Ollama

```python
import asyncio

from pfun_llm.backend.ollama import OllamaGenerativeModel


async def main():
    model = OllamaGenerativeModel()  # or pass model="<name>" explicitly
    response = await model.generate_content("Hello!")
    print(response)


asyncio.run(main())
```

The generated responses and backends follow the same conventions as
[`pfun-cma-model`](https://github.com/pfun-health/pfun-cma-model).

## Installation

```bash
uv sync --extra ollama           # base install with the Ollama backend
uv sync --extra ollama --extra dev   # include test dependencies
```

## Development

Run the test suite:

```bash
uv run pytest
```

The test suite focuses on the essential production functionality of the
Ollama backend: message schemas, message formatting helpers, client
setup, chat/streaming behavior, and error-recovery (auto-pull on `404`).

## Links

- Source: <https://github.com/pfun-health/pfun-llm>
- Docs: <https://pfun-health.github.io/pfun-llm/>
- Releases: <https://pfun-health.github.io/pfun-llm/release-notes/>