from __future__ import annotations

import os
import base64
from typing import Any, Tuple

from ..errors import ProviderOperationError
from ..http import TOTAL_TIMEOUT, request
from ..rate_limit import rate_limit


def setup_client(model_name: str, config: dict[str, Any]):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key)


def text_completion(client: Any, prompt: str, model_name: str, temperature: float = 0.7) -> str:
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        rate_limit("openai", api_key, model_name)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=TOTAL_TIMEOUT,
            )
            return response.choices[0].message.content
        except Exception as api_error:
            if "v1/responses" in str(api_error):
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    temperature=temperature,
                    timeout=TOTAL_TIMEOUT,
                )
                if hasattr(response, "text"):
                    return response.text
                return response.choices[0].text
            raise api_error
    except Exception as e:  # pragma: no cover - network dependent
        raise ProviderOperationError("openai", model_name, "completion", str(e))


def vision_completion(*args, **kwargs):  # pragma: no cover - heavy network
    raise ProviderOperationError("openai", kwargs.get("model_name", ""), "vision", "Not implemented in this environment")


def image_generation(client: Any, prompt: str, model_name: str) -> Tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    rate_limit("openai", api_key, model_name)
    params = {"model": model_name, "prompt": prompt, "n": 1, "size": "1024x1024"}
    if model_name != "gpt-image-1":
        params["response_format"] = "b64_json"
    response = client.images.generate(timeout=TOTAL_TIMEOUT, **params)
    if model_name == "gpt-image-1" and response.data[0].url:
        img_resp = request("GET", response.data[0].url)
        img_resp.raise_for_status()
        image_data_base64 = base64.b64encode(img_resp.content).decode("utf-8")
    else:
        image_data_base64 = response.data[0].b64_json
    return image_data_base64, "image/png"


def image_edit(client: Any, prompt: str, image_path: str, model_name: str, **edit_params: Any) -> Tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    rate_limit("openai", api_key, model_name)
    with open(image_path, "rb") as image_file:
        response = client.images.edit(
            model=model_name,
            image=image_file,
            prompt=prompt,
            timeout=TOTAL_TIMEOUT,
            **edit_params,
        )
    return response.data[0].b64_json, "image/png"


def transcribe_audio(client: Any, audio_path: str, model_name: str, language_code: str = "en-US") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    rate_limit("openai", api_key, model_name)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            timeout=TOTAL_TIMEOUT,
        )
    return transcription.text
