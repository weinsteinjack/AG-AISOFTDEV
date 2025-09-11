import asyncio
import time
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import async_get_completion
from utils import llm as llm_module


class DummyProvider:
    @staticmethod
    async def async_text_completion(client, prompt, model_name, temperature=0.7):
        await asyncio.sleep(0.1)
        return f"resp-{prompt}"


def setup_dummy_provider(monkeypatch):
    monkeypatch.setitem(llm_module.PROVIDERS, "dummy", DummyProvider)


@pytest.mark.asyncio
async def test_async_get_completion_parallel(monkeypatch):
    setup_dummy_provider(monkeypatch)
    client = object()
    prompts = [f"p{i}" for i in range(5)]

    start = time.perf_counter()
    for p in prompts:
        await async_get_completion(p, client, "model", "dummy")
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    await asyncio.gather(
        *(async_get_completion(p, client, "model", "dummy") for p in prompts)
    )
    parallel = time.perf_counter() - start

    assert parallel < sequential / 2
