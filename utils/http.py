"""HTTP helpers with connection pooling, retries, and timeouts."""
from __future__ import annotations

import os
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CONNECT_TIMEOUT = float(os.getenv("UTILS_TIMEOUT_CONNECT", "10"))
READ_TIMEOUT = float(os.getenv("UTILS_TIMEOUT_READ", "60"))
DEFAULT_TIMEOUT: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT)
TOTAL_TIMEOUT: float = sum(DEFAULT_TIMEOUT)
MAX_RETRIES = int(os.getenv("UTILS_MAX_RETRIES", "3"))


class _JitterRetry(Retry):
    """Retry class with jittered exponential backoff."""

    def get_backoff_time(self) -> float:  # pragma: no cover - deterministic
        backoff = super().get_backoff_time()
        if backoff <= 0:
            return 0
        return random.uniform(0, backoff)


def _create_session() -> requests.Session:
    session = requests.Session()
    retry = _JitterRetry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _create_session()


def get_session() -> requests.Session:
    """Return a module-level ``requests`` session."""

    return _SESSION


def request(method: str, url: str, **kwargs) -> requests.Response:
    """Wrapper around ``Session.request`` applying default timeout."""

    if "timeout" not in kwargs:
        kwargs["timeout"] = DEFAULT_TIMEOUT
    return _SESSION.request(method, url, **kwargs)


__all__ = [
    "DEFAULT_TIMEOUT",
    "TOTAL_TIMEOUT",
    "get_session",
    "request",
]
