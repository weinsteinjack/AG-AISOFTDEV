from __future__ import annotations

import os
from io import BytesIO
import base64
from typing import Any, Tuple

from ..errors import ProviderOperationError


def setup_client(model_name: str, config: dict[str, Any]):
    from huggingface_hub import InferenceClient
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in .env file.")
    return InferenceClient(model=model_name, token=api_key)


def text_completion(client: Any, prompt: str, model_name: str, temperature: float = 0.7) -> str:
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=max(0.1, temperature),
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:  # pragma: no cover - network dependent
        raise ProviderOperationError("huggingface", model_name, "completion", str(e))


def vision_completion(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("huggingface", kwargs.get("model_name", ""), "vision", "Not implemented")


def image_generation(client: Any, prompt: str, model_name: str) -> Tuple[str, str]:
    pil_image = client.text_to_image(prompt)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), "image/png"


def image_edit(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("huggingface", kwargs.get("model_name", ""), "image edit", "Not implemented")


def transcribe_audio(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("huggingface", kwargs.get("model_name", ""), "audio transcription", "Not implemented")
