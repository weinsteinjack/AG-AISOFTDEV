from __future__ import annotations

import os
from typing import Any

from .errors import ProviderOperationError
from .models import RECOMMENDED_MODELS
from .providers import PROVIDERS


def transcribe_audio(audio_path: str, client: Any, model_name: str, api_provider: str, language_code: str = "en-US") -> str:
    if not client:
        raise ProviderOperationError(api_provider, model_name, "audio transcription", "API client not initialized.")
    if not RECOMMENDED_MODELS.get(model_name, {}).get("audio_transcription"):
        raise ProviderOperationError(api_provider, model_name, "audio transcription", f"Model '{model_name}' does not support audio transcription.")
    if not os.path.exists(audio_path):
        raise ProviderOperationError(api_provider, model_name, "audio transcription", f"Audio file not found at {audio_path}")
    provider_module = PROVIDERS.get(api_provider)
    if not provider_module:
        raise ProviderOperationError(api_provider, model_name, "audio transcription", "Unsupported provider")
    return provider_module.transcribe_audio(client, audio_path, model_name, language_code)


def transcribe_audio_compat(audio_path: str, client: Any, model_name: str, api_provider: str, language_code: str = "en-US"):
    try:
        return transcribe_audio(audio_path, client, model_name, api_provider, language_code), None
    except ProviderOperationError as e:
        return None, str(e)


__all__ = ['transcribe_audio', 'transcribe_audio_compat']
