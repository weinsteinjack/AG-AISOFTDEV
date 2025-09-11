from __future__ import annotations

import os
import asyncio
from typing import Any, Tuple

from ..errors import ProviderOperationError
from ..http import TOTAL_TIMEOUT
from ..rate_limit import rate_limit


def setup_client(model_name: str, config: dict[str, Any]):
    if config.get("audio_transcription"):
        from google.cloud import speech
        return speech.SpeechClient()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    if "imagen" in model_name:
        from google import genai as google_genai
        return google_genai.Client(api_key=api_key)
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


async def async_setup_client(model_name: str, config: dict[str, Any]):
    return await asyncio.to_thread(setup_client, model_name, config)


def text_completion(client: Any, prompt: str, model_name: str, temperature: float = 0.7) -> str:
    try:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        rate_limit("google", api_key, model_name)
        response = client.generate_content(
            prompt, timeout=TOTAL_TIMEOUT
        )
        return response.text
    except Exception as e:  # pragma: no cover - network dependent
        raise ProviderOperationError("google", model_name, "completion", str(e))


async def async_text_completion(client: Any, prompt: str, model_name: str, temperature: float = 0.7) -> str:
    return await asyncio.to_thread(text_completion, client, prompt, model_name, temperature)


def vision_completion(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("google", kwargs.get("model_name", ""), "vision", "Not implemented")


async def async_vision_completion(*args, **kwargs):  # pragma: no cover
    return await asyncio.to_thread(vision_completion, *args, **kwargs)


def image_generation(client: Any, prompt: str, model_name: str) -> Tuple[str, str]:
    if "imagen" in model_name:
        from google import genai as google_genai
        from google.genai import types as google_types

        api_key = os.getenv("GOOGLE_API_KEY", "")
        rate_limit("google", api_key, model_name)
        response = client.models.generate_images(
            model=model_name, prompt=prompt, request_options={"timeout": TOTAL_TIMEOUT}
        )
        image_data_base64 = response.generated_images[0].bytes_base64
        return image_data_base64, "image/png"
    raise ProviderOperationError("google", model_name, "image generation", "Not implemented for this model")


async def async_image_generation(client: Any, prompt: str, model_name: str) -> Tuple[str, str]:
    return await asyncio.to_thread(image_generation, client, prompt, model_name)


def image_edit(*args, **kwargs):  # pragma: no cover
    raise ProviderOperationError("google", kwargs.get("model_name", ""), "image edit", "Not implemented")


async def async_image_edit(*args, **kwargs):  # pragma: no cover
    return await asyncio.to_thread(image_edit, *args, **kwargs)


def transcribe_audio(client: Any, audio_path: str, model_name: str, language_code: str = "en-US") -> str:
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
