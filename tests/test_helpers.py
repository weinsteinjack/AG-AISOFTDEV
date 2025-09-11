from typing import Any, cast

import pytest

from utils.errors import (
    CLIENT_NOT_INITIALIZED,
    UNSUPPORTED_PROVIDER,
    ProviderOperationError,
)
from utils.helpers import ensure_provider, normalize_prompt
from utils.providers import PROVIDERS


def test_normalize_prompt() -> None:
    assert normalize_prompt("  hi  ") == "hi"


def test_ensure_provider_success() -> None:
    client = object()
    dummy: Any = object()
    PROVIDERS["dummy"] = cast(Any, dummy)
    try:
        assert ensure_provider(client, "dummy", "m", "op") is dummy
    finally:
        del PROVIDERS["dummy"]


def test_ensure_provider_missing_client() -> None:
    with pytest.raises(ProviderOperationError) as exc:
        ensure_provider(None, "p", "m", "op")  # type: ignore[arg-type]
    assert CLIENT_NOT_INITIALIZED in str(exc.value)


def test_ensure_provider_unknown_provider() -> None:
    with pytest.raises(ProviderOperationError) as exc:
        ensure_provider(object(), "unknown", "m", "op")
    assert UNSUPPORTED_PROVIDER in str(exc.value)


def test_get_completion_stub() -> None:
    from utils import llm

    class Stub:
        def text_completion(
            self, client: Any, prompt: str, model_name: str, temperature: float = 0.7
        ) -> str:
            return prompt

    client = object()
    PROVIDERS["stub"] = cast(Any, Stub())
    try:
        result = llm.get_completion(" hi ", client, "m", "stub")
        assert result == "hi"
    finally:
        del PROVIDERS["stub"]


@pytest.mark.asyncio
async def test_async_get_completion_stub() -> None:
    from utils import llm

    class Stub:
        async def async_text_completion(
            self, client: Any, prompt: str, model_name: str, temperature: float = 0.7
        ) -> str:
            return prompt

    client = object()
    PROVIDERS["stub"] = cast(Any, Stub())
    try:
        result = await llm.async_get_completion(" hi ", client, "m", "stub")
        assert result == "hi"
    finally:
        del PROVIDERS["stub"]
