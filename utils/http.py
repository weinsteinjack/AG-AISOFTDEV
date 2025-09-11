"""HTTP helpers used across provider implementations."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_session(retries: int = 3,
                backoff_factor: float = 0.3,
                status_forcelist: tuple[int, ...] = (500, 502, 504)) -> requests.Session:
    """Return a ``requests`` session with retry behaviour configured."""
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

__all__ = ["get_session"]
