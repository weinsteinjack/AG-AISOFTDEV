# Productionizing the `utils` Package

The `utils` package powers every lab notebook, but it is also designed to support production workloads. This guide explains how to configure timeouts, retries, logging, and rate limiting when you promote classroom prototypes into real applications.

---

## 1. Loading Configuration

Most helpers lazily call `load_environment()` from `utils.settings`. When running outside a notebook, call it explicitly during startup:

```python
from utils import load_environment

# Loads .env (if present) and returns a dictionary of key/value pairs
config = load_environment()
```

The function automatically falls back to OS environment variables if `.env` is missing. Store secrets (API keys, database URLs) in environment variables in production and reserve `.env` files for local development.

---

## 2. HTTP Timeouts and Retries

The shared HTTP client (`utils.http.request`) exposes sensible defaults:

| Environment Variable | Default | Description |
| --- | --- | --- |
| `UTILS_TIMEOUT_CONNECT` | `10` seconds | Maximum time to establish a connection. |
| `UTILS_TIMEOUT_READ` | `60` seconds | Maximum time to wait for a response. |
| `UTILS_MAX_RETRIES` | `3` | Number of retry attempts for transient HTTP errors. |

Example usage:

```python
from utils.http import request

response = request("GET", "https://api.example.com/status")
response.raise_for_status()
data = response.json()
```

Override defaults per deployment:

```bash
export UTILS_TIMEOUT_CONNECT=5
export UTILS_TIMEOUT_READ=30
export UTILS_MAX_RETRIES=5
```

These settings apply to all helpers that rely on the shared HTTP session (LLM providers, embedding services, etc.).

---

## 3. Rate Limiting Provider Calls

Set provider-specific query-per-second (QPS) limits to prevent accidental throttling:

```bash
export UTILS_RATE_LIMIT_QPS_OPENAI=1
export UTILS_RATE_LIMIT_QPS_ANTHROPIC=0.5
```

In code, call `rate_limit()` before issuing a provider request:

```python
from utils import get_completion
from utils.rate_limit import rate_limit

rate_limit("openai", api_key, "gpt-4o")
completion = get_completion(prompt, client, model_name="gpt-4o", api_provider="openai")
```

Each combination of provider + API key + model maintains its own token bucket. The limiter persists for the life of the Python process, making it safe for web servers and background workers alike.

---

## 4. Structured Logging

Use the built-in logger to capture consistent metadata across notebooks, scripts, and servers:

```python
from utils.logging import get_logger

logger = get_logger()
logger.info("Starting onboarding agent", extra={"stage": "init"})
```

Configuration tips:

* Control verbosity with `UTILS_LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, etc.).
* When running inside Uvicorn, pass the logger into your FastAPI app so request/response cycles appear alongside agent events.
* Ship logs to a centralised platform (ELK, Datadog, CloudWatch) by configuring your logging handlers—`get_logger()` returns a standard Python logger so all familiar patterns apply.

---

## 5. Working with Async Helpers

When calling async functions such as `async_get_completion()` or `async_transcribe_audio()`, reuse clients and event loops to avoid connection overhead:

```python
import asyncio
from utils import async_setup_llm_client, async_get_completion

async def generate_updates(prompts):
    client, model, provider = await async_setup_llm_client(preferred_provider="openai")
    tasks = [async_get_completion(p, client, model, provider) for p in prompts]
    return await asyncio.gather(*tasks)

asyncio.run(generate_updates(["Summarise day 1", "Summarise day 2"]))
```

The async helpers respect the same timeout and rate-limit environment variables described above. Review `async_tests/test_async_llm.py` for patterns that scale to production pipelines.

---

## 6. Handling Legacy Compatibility Wrappers

Some labs reference older helper signatures that returned `(result, error_message)` tuples. These wrappers remain available (e.g., `get_completion_compat`, `async_get_image_generation_completion_compat`). They translate tuple semantics into the new exception-based API.

* Prefer the modern helpers (`get_completion`, `get_image_generation_completion`, etc.).
* If you must call a `_compat` function, always check the second element of the tuple before using the result.
* Plan to migrate legacy code soon—the wrappers are marked for future removal.

---

## 7. Deployment Checklist

Use this quick reference when shipping utilities-backed applications:

- [ ] Call `load_environment()` once during startup (web server, CLI, or notebook).
- [ ] Set timeout and retry environment variables to match your provider SLAs.
- [ ] Configure `UTILS_RATE_LIMIT_QPS_*` for each provider/API key combination used in production.
- [ ] Initialise a structured logger via `get_logger()` and ensure logs are shipped off-host.
- [ ] Guard provider calls with `try`/`except` and surface actionable error messages to users.
- [ ] When saving generated files, continue using `utils.artifacts` to enforce sandboxing.

---

By tuning these settings you can safely graduate from notebook experiments to production services without rewriting the helper layer. Combine this guide with the Docker and Deployment references to deliver robust AI-enhanced applications.
