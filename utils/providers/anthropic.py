from __future__ import annotations

import os
from typing import Any, Tuple

from ..errors import ProviderOperationError


def setup_client(model_name: str, config: dict[str, Any]):
    from anthropic import Anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file.")
    return Anthropic(api_key=api_key)


def text_completion(client: Any, prompt: str, model_name: str, temperature: float = 0.7) -> str:
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:  # pragma: no cover - network dependent
        raise ProviderOperationError("anthropic", model_name, "completion", str(e))


def vision_completion(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("anthropic", kwargs.get("model_name", ""), "vision", "Not implemented")


def image_generation(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("anthropic", kwargs.get("model_name", ""), "image generation", "Not implemented")


def image_edit(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("anthropic", kwargs.get("model_name", ""), "image edit", "Not implemented")


def transcribe_audio(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("anthropic", kwargs.get("model_name", ""), "audio transcription", "Not implemented")
