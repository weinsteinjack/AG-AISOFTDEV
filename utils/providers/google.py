from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Tuple

from ..errors import ProviderOperationError
from ..http import TOTAL_TIMEOUT
from ..rate_limit import rate_limit


def _is_image_model(model_name: str) -> bool:
    """Return ``True`` if ``model_name`` uses Google's image generation stack."""
    lowered = model_name.lower()
    return "imagen" in lowered or "image" in lowered


def _extract_generated_image(response: Any, model_name: str) -> Tuple[str, str]:
    """Normalize Google image responses to ``(base64, mime_type)``."""
    generated = getattr(response, "generated_images", None) or []
    if not generated:
        raise ProviderOperationError(
            "google", model_name, "image generation", "No image data returned by API"
        )
    image_obj = getattr(generated[0], "image", None)
    if not image_obj:
        raise ProviderOperationError(
            "google", model_name, "image generation", "Missing image payload in response"
        )
    mime_type = getattr(image_obj, "mime_type", None) or "image/png"
    image_bytes = getattr(image_obj, "image_bytes", None)
    if isinstance(image_bytes, bytes) and image_bytes:
        return base64.b64encode(image_bytes).decode("utf-8"), mime_type
    inline_b64 = getattr(image_obj, "bytes_base64", None)
    if isinstance(inline_b64, str) and inline_b64:
        return inline_b64, mime_type
    raise ProviderOperationError(
        "google",
        model_name,
        "image generation",
        "Google response did not include image bytes",
    )


def setup_client(model_name: str, config: dict[str, Any]) -> Any:
    if config.get("audio_transcription"):
        from google.cloud import speech

        return speech.SpeechClient()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    if _is_image_model(model_name):
        from google import genai as google_genai

        return google_genai.Client(api_key=api_key)
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


async def async_setup_client(model_name: str, config: dict[str, Any]) -> Any:
    return await asyncio.to_thread(setup_client, model_name, config)


def text_completion(
    client: Any, prompt: str, model_name: str, temperature: float = 0.7
) -> str:
    try:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        rate_limit("google", api_key, model_name)
        # google.generativeai's GenerativeModel does not accept a `timeout` kwarg.
        # Use `request_options={"timeout": ...}` instead. Also pass temperature via
        # generation_config so callers can control creativity similar to other providers.
        response = client.generate_content(
            prompt,
            request_options={"timeout": TOTAL_TIMEOUT},
            generation_config={"temperature": temperature},
        )
        return response.text
    except Exception as e:  # pragma: no cover - network dependent
        raise ProviderOperationError("google", model_name, "completion", str(e))


async def async_text_completion(
    client: Any, prompt: str, model_name: str, temperature: float = 0.7
) -> str:
    return await asyncio.to_thread(
        text_completion, client, prompt, model_name, temperature
    )


def vision_completion(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    raise ProviderOperationError(
        "google", kwargs.get("model_name", ""), "vision", "Not implemented"
    )


async def async_vision_completion(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    return await asyncio.to_thread(vision_completion, *args, **kwargs)


def image_generation(client: Any, prompt: str, model_name: str) -> Tuple[str, str]:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    rate_limit("google", api_key, model_name)
    if _is_image_model(model_name) and hasattr(client, "models"):
        response = client.models.generate_images(model=model_name, prompt=prompt)
        return _extract_generated_image(response, model_name)

    if _is_image_model(model_name) and hasattr(client, "generate_content"):
        response = client.generate_content(
            prompt,
            request_options={"timeout": TOTAL_TIMEOUT},
        )
        for part in getattr(response, "parts", []):
            inline_data = getattr(part, "inline_data", None)
            if not inline_data:
                continue
            data = getattr(inline_data, "data", None)
            mime_type = getattr(inline_data, "mime_type", None) or "image/png"
            if isinstance(data, bytes) and data:
                return base64.b64encode(data).decode("utf-8"), mime_type
            if isinstance(data, str) and data:
                return data, mime_type
        raise ProviderOperationError(
            "google",
            model_name,
            "image generation",
            "generate_content did not return inline image data",
        )

    raise ProviderOperationError(
        "google",
        model_name,
        "image generation",
        "Not implemented for this model",
    )


async def async_image_generation(
    client: Any, prompt: str, model_name: str
) -> Tuple[str, str]:
    return await asyncio.to_thread(image_generation, client, prompt, model_name)


def image_edit(*args: Any, **kwargs: Any) -> Tuple[str, str]:  # pragma: no cover
    raise ProviderOperationError(
        "google", kwargs.get("model_name", ""), "image edit", "Not implemented"
    )


async def async_image_edit(
    *args: Any, **kwargs: Any
) -> Tuple[str, str]:  # pragma: no cover
    return await asyncio.to_thread(image_edit, *args, **kwargs)


def transcribe_audio(
    client: Any, audio_path: str, model_name: str, language_code: str = "en-US"
) -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    rate_limit("google", api_key, model_name)
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    audio = {"content": content}
    config = {"language_code": language_code}
    response = client.recognize(config=config, audio=audio, timeout=TOTAL_TIMEOUT)
    if response.results:
        return response.results[0].alternatives[0].transcript
    raise ProviderOperationError(
        "google",
        model_name,
        "audio transcription",
        "No transcription result from Google Speech-to-Text.",
    )


async def async_transcribe_audio(
    client: Any, audio_path: str, model_name: str, language_code: str = "en-US"
) -> str:
    return await asyncio.to_thread(
        transcribe_audio, client, audio_path, model_name, language_code
    )
