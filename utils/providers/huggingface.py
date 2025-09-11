from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from typing import Any, Tuple

from ..errors import ProviderOperationError
from ..http import TOTAL_TIMEOUT
from ..rate_limit import rate_limit


def setup_client(model_name: str, config: dict[str, Any]) -> Any:
    from huggingface_hub import InferenceClient

    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in .env file.")
    return InferenceClient(model=model_name, token=api_key)


async def async_setup_client(model_name: str, config: dict[str, Any]) -> Any:
    return await asyncio.to_thread(setup_client, model_name, config)


def text_completion(
    client: Any, prompt: str, model_name: str, temperature: float = 0.7
) -> str:
    try:
        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        rate_limit("huggingface", api_key, model_name)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=max(0.1, temperature),
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:  # pragma: no cover - network dependent
        raise ProviderOperationError("huggingface", model_name, "completion", str(e))


async def async_text_completion(
    client: Any, prompt: str, model_name: str, temperature: float = 0.7
) -> str:
    return await asyncio.to_thread(
        text_completion, client, prompt, model_name, temperature
    )


def vision_completion(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    raise ProviderOperationError(
        "huggingface", kwargs.get("model_name", ""), "vision", "Not implemented"
    )


async def async_vision_completion(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    return await asyncio.to_thread(vision_completion, *args, **kwargs)


def image_generation(client: Any, prompt: str, model_name: str) -> Tuple[str, str]:
    api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    rate_limit("huggingface", api_key, model_name)
    pil_image = client.text_to_image(prompt, timeout=TOTAL_TIMEOUT)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), "image/png"


async def async_image_generation(
    client: Any, prompt: str, model_name: str
) -> Tuple[str, str]:
    return await asyncio.to_thread(image_generation, client, prompt, model_name)


def image_edit(*args: Any, **kwargs: Any) -> Tuple[str, str]:  # pragma: no cover
    raise ProviderOperationError(
        "huggingface", kwargs.get("model_name", ""), "image edit", "Not implemented"
    )


async def async_image_edit(
    *args: Any, **kwargs: Any
) -> Tuple[str, str]:  # pragma: no cover
    return await asyncio.to_thread(image_edit, *args, **kwargs)


def transcribe_audio(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    raise ProviderOperationError(
        "huggingface",
        kwargs.get("model_name", ""),
        "audio transcription",
        "Not implemented",
    )


async def async_transcribe_audio(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    return await asyncio.to_thread(transcribe_audio, *args, **kwargs)
