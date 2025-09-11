# Productionizing Utils

This guide covers configuration options for the `utils` helpers when moving from
experiments to production environments.

## Timeouts and Retries

HTTP helpers use sensible defaults that can be tuned with environment
variables:

- `UTILS_TIMEOUT_CONNECT` – connection timeout in seconds (default `10`)
- `UTILS_TIMEOUT_READ` – read timeout in seconds (default `60`)
- `UTILS_MAX_RETRIES` – number of retry attempts for transient errors (default `3`)

Example:

```python
from utils.http import request

response = request("GET", "https://httpbin.org/get")
print(response.status_code)
```

## Rate Limiting

Rate limiting is enforced per provider/API key/model using a token bucket. Enable
it by setting `UTILS_RATE_LIMIT_QPS_<PROVIDER>` to the allowed queries per
second.

```python
import os
from utils.rate_limit import rate_limit

os.environ["UTILS_RATE_LIMIT_QPS_OPENAI"] = "1"
rate_limit("openai", "fake-key", "gpt-4o")
```

## Logging

Use `utils.logging.get_logger` to emit structured logs. The log level can be
controlled with `UTILS_LOG_LEVEL`.

```python
from utils.logging import get_logger

logger = get_logger()
logger.info("hello from utils")
```
