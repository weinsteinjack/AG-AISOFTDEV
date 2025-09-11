from __future__ import annotations

import base64
import mimetypes
import time
from typing import Any, Tuple

from .errors import ProviderOperationError
from .logging import get_logger
from .providers import PROVIDERS
from .artifacts import save_artifact

logger = get_logger()


def _save_image(image_data_base64: str, image_mime: str) -> Tuple[str, str]:
    ext = mimetypes.guess_extension(image_mime) or ".png"
    filename = f"image_{int(time.time())}{ext}"
    file_path = save_artifact(base64.b64decode(image_data_base64), filename, subdir="screens")
    image_url = f"data:{image_mime};base64,{image_data_base64}"
    return str(file_path), image_url


def get_image_generation_completion(prompt: str, client: Any, model_name: str, api_provider: str) -> Tuple[str, str]:
    if not client:
        raise ProviderOperationError(api_provider, model_name, "image generation", "API client not initialized.")
    provider_module = PROVIDERS.get(api_provider)
    if not provider_module:
        raise ProviderOperationError(api_provider, model_name, "image generation", "Unsupported provider")
    image_data_base64, image_mime = provider_module.image_generation(client, prompt, model_name)
    return _save_image(image_data_base64, image_mime)


def get_image_generation_completion_compat(prompt: str, client: Any, model_name: str, api_provider: str):
    try:
        return get_image_generation_completion(prompt, client, model_name, api_provider), None
    except ProviderOperationError as e:
        return None, str(e)


def get_image_edit_completion(prompt: str, image_path: str, client: Any, model_name: str, api_provider: str, **edit_params: Any) -> Tuple[str, str]:
    if not client:
        raise ProviderOperationError(api_provider, model_name, "image edit", "API client not initialized.")
    provider_module = PROVIDERS.get(api_provider)
    if not provider_module:
        raise ProviderOperationError(api_provider, model_name, "image edit", "Unsupported provider")
    image_data_base64, image_mime = provider_module.image_edit(client, prompt, image_path, model_name, **edit_params)
    return _save_image(image_data_base64, image_mime)


def get_image_edit_completion_compat(prompt: str, image_path: str, client: Any, model_name: str, api_provider: str, **edit_params: Any):
    try:
        return get_image_edit_completion(prompt, image_path, client, model_name, api_provider, **edit_params), None
    except ProviderOperationError as e:
        return None, str(e)


__all__ = [
    'get_image_generation_completion', 'get_image_generation_completion_compat',
    'get_image_edit_completion', 'get_image_edit_completion_compat'
]
