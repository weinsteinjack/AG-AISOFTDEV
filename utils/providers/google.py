from __future__ import annotations

import asyncio
import base64
import os
import random
import time
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

    for candidate in generated:
        image_obj = getattr(candidate, "image", None)
        if not image_obj:
            continue
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


_GENAI_IMPORTS: tuple[Any, Any] | None = None


def _get_google_genai_imports() -> tuple[Any, Any]:
    """Lazily import google.genai helpers so tests can stub behavior."""
    global _GENAI_IMPORTS
    if _GENAI_IMPORTS is None:
        try:
            from google.genai import errors as genai_errors  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            _GENAI_IMPORTS = (None, None)
        else:
            _GENAI_IMPORTS = (genai_errors, genai_types)
    return _GENAI_IMPORTS


def _should_retry_with_v1(error: Exception) -> bool:
    genai_errors, _ = _get_google_genai_imports()
    if not genai_errors or not isinstance(error, genai_errors.ClientError):
        return False
    if getattr(error, "code", None) == 404:
        return True
    message = (getattr(error, "message", "") or str(error) or "").lower()
    return "api version" in message or "predict" in message


def _is_client_not_found(error: Exception) -> bool:
    genai_errors, _ = _get_google_genai_imports()
    return bool(
        genai_errors
        and isinstance(error, genai_errors.ClientError)
        and getattr(error, "code", None) == 404
    )


def image_generation(
    client: Any, prompt: str, model_name: str
) -> tuple[bytes, str]:
    """
    Generate an image using a Google image generation model.

    Args:
        client: The google.genai.Client object.
        prompt (str): The text prompt for image generation.
        model_name (str): The name of the image generation model.

    Returns:
        A tuple containing the image bytes and its MIME type.
    """
    _, genai_types = _get_google_genai_imports()
    if not genai_types:
        raise ProviderOperationError(
            "google",
            model_name,
            "image_generation",
            "google.generativeai is not installed",
        )

    try:
        # Generate the image content
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        # Extract the image data from the response
        parts = response.candidates[0].content.parts
        for part in parts:
            blob = getattr(part, "inline_data", None)
            if blob and getattr(blob, "data", None):
                return blob.data, blob.mime_type

        raise ProviderOperationError(
            "google", model_name, "image_generation", "No image data found in response"
        )

    except Exception as e:
        # Wrap exceptions in ProviderOperationError for consistent error handling
        raise ProviderOperationError(
            "google", model_name, "image_generation", f"API call failed: {e}"
        )


def setup_client(model_name: str, config: dict[str, Any]) -> Any:
    """Set up the appropriate Google client based on the model's task."""
    if config.get("audio_transcription"):
        from google.cloud import speech

        return speech.SpeechClient()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    from google import genai

    # For text models, configure the API key and return the model object.
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
