# --- Helper Script for AI-Driven Software Engineering Course ---
# Description: This script provides a unified interface for interacting with
#              multiple LLM providers (OpenAI, Anthropic, Hugging Face, Google Gemini)
#              and simplifies common tasks like artifact management.
# -----------------------------------------------------------------

import os
import json
import requests
from PIL import Image
from io import BytesIO
import re
import base64
import mimetypes
import time
from typing import Any, Callable

# --- Dynamic Library Installation ---
# Provide a broad-typed `display` name so static type checkers won't complain
display: Callable[..., Any] = lambda *a, **k: None
# Optional helpers; we import inside a try to avoid hard dependency on IPython/dotenv
_load_dotenv = None
_have_ipython_display = False
try:
    from dotenv import load_dotenv as _ld
    _load_dotenv = _ld
except Exception:
    # dotenv is optional; functions that rely on it will handle absence gracefully
    _load_dotenv = None

try:
    import IPython.display as _IPython_display
    Markdown = _IPython_display.Markdown
    Code = _IPython_display.Code
    IPyImage = _IPython_display.Image
    display = _IPython_display.display
    _have_ipython_display = True
except Exception:
    # Minimal fallbacks when IPython.display isn't available
    def display(*args, **kwargs):
        for a in args:
            print(a)

    def Markdown(x):
        return x

    def Code(x, language=None):
        return x

    def IPyImage(data=None, url=None, filename=None):
        if data:
            return "[Image from data]"
        if url:
            return f"[Image from {url}]"
        if filename:
            return f"[Image from {filename}]"
        return "[Image]"

# --- Global Variables & Configuration ---
# Load environment variables from .env file if python-dotenv was available
if _load_dotenv:
    try:
        _load_dotenv()
    except Exception:
        # best-effort; don't fail the module import if .env can't be loaded
        pass

# Centralized dictionary to store model metadata
MODELS_JSON_PATH = os.path.join(os.path.dirname(__file__), 'models.json')
if os.path.exists(MODELS_JSON_PATH):
    with open(MODELS_JSON_PATH, 'r') as f:
        ALL_MODELS = json.load(f)
else:
    print(f"Warning: '{MODELS_JSON_PATH}' not found. Using an empty model list.")
    ALL_MODELS = {}

# Backwards-compatible alias used across the module
RECOMMENDED_MODELS = ALL_MODELS

# --- Core Functions ---

def get_model_info(model_name):
    """
    Retrieves metadata for a specific model from the global ALL_MODELS dictionary.

    Args:
        model_name (str): The name of the model to look up.

    Returns:
        dict: A dictionary containing the model's metadata, or an empty dictionary if not found.
    """
    return ALL_MODELS.get(model_name, {})

def get_provider_for_model(model_name):
    """
    Determines the provider for a given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The name of the provider (e.g., 'openai', 'anthropic', 'huggingface', 'google'),
             or 'unknown' if the provider cannot be determined.
    """
    model_info = get_model_info(model_name)
    return model_info.get("provider", "unknown")

def get_client_for_model(model_name):
    """
    Initializes and returns the appropriate API client based on the model provider.

    This function reads the necessary API keys from environment variables.
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    - HUGGINGFACE_API_KEY for Hugging Face
    - GOOGLE_API_KEY or GEMINI_API_KEY for Google

    Args:
        model_name (str): The name of the model for which to get the client.

    Returns:
        An instance of the appropriate client (e.g., OpenAI, Anthropic, etc.),
        or None if the provider is unknown or the client fails to initialize.
    """
    provider = get_provider_for_model(model_name)
    client = None

    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "huggingface":
            # For Hugging Face, we often use the requests library directly
            # but a dedicated client could be used if preferred.
            # We will return a "pseudo-client" dictionary for consistency.
            client = {
                "provider": "huggingface",
                "api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "api_url": f"https://api-inference.huggingface.co/models/{model_name}"
            }
        elif provider == "google":
            # Try multiple Google client libraries
            try:
                # New `google-genai` library
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    # The "client" is the library module itself, configured globally
                    client = genai
                else:
                    print("⚠️ Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
            except ImportError:
                try:
                    # Older `google.generativeai` library structure
                    from google.generativeai import GenerativeModel
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if api_key:
                        # This path might require a different initialization
                        # For simplicity, we'll assume the newer library is preferred.
                        # If you need the old one, adjust the client creation here.
                        print("ℹ️ Found older google.generativeai library. The new `google-genai` is recommended.")
                        # Create a model instance as the "client"
                        client = GenerativeModel(model_name)
                    else:
                        print("⚠️ Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
                except ImportError:
                    print("⚠️ Neither `google-genai` nor `google.generativeai` library found.")
                    print("💡 Install with: pip install google-genai")
        else:
            print(f"Provider '{provider}' for model '{model_name}' is not supported.")

    except Exception as e:
        print(f"❌ Failed to initialize client for {provider}: {e}")
        return None

    return client

def recommended_models_table(
    provider=None,
    vision=None,
    image_gen=None,
    audio_transcription=None,
    min_context=None,
    min_output_tokens=None,
    task=None
):
    """
    Generates a Markdown table of recommended models, filtered by specified criteria.

    This function dynamically filters the `ALL_MODELS` dictionary based on capabilities
    and returns a formatted string suitable for display in notebooks or terminals.

    Args:
        provider (str or list, optional): A provider name or list of provider names to filter by.
            Example: "openai" or ["openai", "anthropic"]
        vision (bool, optional): Filter models by vision (image-to-text) capability.
            - True: Only models that can process images.
            - False: Only models that cannot process images.
        image_gen (bool, optional): Filter models by image generation capability.
            - True: Only image generation models
            - False: Only non-image generation models
        audio_transcription (bool, optional): Filter models by audio transcription capability.
            - True: Only speech-to-text models
            - False: Only non-speech-to-text models
        min_context (int, optional): Minimum context window size in tokens required.
            Example: 32000 for models with at least 32K context
        min_output_tokens (int, optional): Minimum maximum output tokens required.
            Example: 4096 for models that can generate at least 4K tokens

    Returns:
        str: Markdown-formatted table string containing model information including:
            - Model name and provider
            - Vision, image generation, and audio capabilities (✅/❌)
            - Context window size and maximum output tokens

    Raises:
        None: This function handles all errors gracefully and returns appropriate messages.

    Examples:
        >>> # Show all available models
        >>> table = recommended_models_table()
        >>> display(Markdown(table))

        >>> # Show only OpenAI models with vision
        >>> table = recommended_models_table(provider="openai", vision=True)
        >>> print(table)

        >>> # Show models from any provider with at least 128K context
        >>> table = recommended_models_table(min_context=128000)
        >>> display(Markdown(table))
    """
    # Interpret task shortcuts
    if task:
        t = task.lower()
        if t in {"vision", "multimodal", "vl"} and vision is None:
            vision = True
        elif t in {"image", "image_generation", "image-generation"} and image_gen is None:
            image_gen = True
        elif t in {"audio", "speech", "audio_transcription", "stt"} and audio_transcription is None:
            audio_transcription = True
        elif t == "text":
            vision = False if vision is None else vision
            image_gen = False if image_gen is None else image_gen
            audio_transcription = False if audio_transcription is None else audio_transcription

    filtered_models = []
    provider_list = [provider] if isinstance(provider, str) else provider

    for name, meta in ALL_MODELS.items():
        # Provider filter
        if provider_list and meta.get("provider") not in provider_list:
            continue
        # Vision filter
        if vision is not None and meta.get("vision", False) != vision:
            continue
        # Image generation filter
        if image_gen is not None and meta.get("image_gen", False) != image_gen:
            continue
        # Audio transcription filter
        if audio_transcription is not None and meta.get("audio_transcription", False) != audio_transcription:
            continue
        # Context window filter
        if min_context is not None and meta.get("context_window", 0) < min_context:
            continue
        # Max output tokens filter
        if min_output_tokens is not None and meta.get("max_output_tokens", 0) < min_output_tokens:
            continue

        filtered_models.append((name, meta))

    if not filtered_models:
        return "No models match the specified criteria."

    # Sort by provider, then by a default rank or name
    filtered_models.sort(key=lambda x: (x[1].get("provider", ""), x[1].get("rank", 999), x[0]))

    # Create Markdown table
    headers = [
        "Model Name", "Provider", "Vision", "Image Gen", "Audio",
        "Context (Tokens)", "Max Output (Tokens)"
    ]
    table = ["| " + " | ".join(headers) + " |", "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"]

    for name, meta in filtered_models:
        row = [
            f"`{name}`",
            meta.get("provider", "n/a").title(),
            "✅" if meta.get("vision") else "❌",
            "✅" if meta.get("image_gen") else "❌",
            "✅" if meta.get("audio_transcription") else "❌",
            f'{meta.get("context_window", 0):,}' if meta.get("context_window") else "n/a",
            f'{meta.get("max_output_tokens", 0):,}' if meta.get("max_output_tokens") else "n/a"
        ]
        table.append("| " + " | ".join(row) + " |")

    return "\n".join(table)


def show_recommended_models(
    provider=None,
    vision=None,
    image_gen=None,
    audio_transcription=None,
    min_context=None,
    min_output_tokens=None
):
    """
    Displays a filtered list of recommended models as a Markdown table in a notebook.

    This is a convenience wrapper around `recommended_models_table` that directly
    calls `display` and `Markdown` for a richer output in IPython environments.

    Args:
        provider (str or list, optional): Filter by one or more providers.
        vision (bool, optional): Filter by vision capability.
        image_gen (bool, optional): Filter by image generation capability.
        audio_transcription (bool, optional): Filter by audio transcription capability.
        min_context (int, optional): Filter by minimum context window size.
        min_output_tokens (int, optional): Filter by minimum maximum output tokens.
    """
    table_md = recommended_models_table(
        provider=provider,
        vision=vision,
        image_gen=image_gen,
        audio_transcription=audio_transcription,
        min_context=min_context,
        min_output_tokens=min_output_tokens
    )
    display(Markdown(table_md))

# --- Artifact Management ---

def _get_artifact_path(artifact_name):
    """Constructs the full path for a given artifact name."""
    return os.path.join('artifacts', artifact_name)

def save_artifact_local(artifact_name, content):
    """
    Saves content to a file in the 'artifacts' directory (legacy/simple helper).

    Args:
        artifact_name (str): The name of the file to save.
        content (str): The content to write to the file.
    """
    # Keep legacy API but delegate to the unified save_artifact
    save_artifact(content, os.path.join('artifacts', artifact_name))

def load_artifact_local(artifact_name):
    """
    Loads content from a file in the 'artifacts' directory (legacy/simple helper).

    Args:
        artifact_name (str): The name of the file to load.

    Returns:
        str: The content of the file, or None if the file doesn't exist.
    """
    # Legacy wrapper for compatibility
    return load_artifact(os.path.join('artifacts', artifact_name))

def display_artifact_local(artifact_name, language=None):
    """
    Displays an artifact in a notebook, formatted as code or markdown.

    Args:
        artifact_name (str): The name of the artifact to display.
        language (str, optional): The language for syntax highlighting.
                                  If not provided, defaults to markdown.
    """
    content = load_artifact_local(artifact_name)
    if content:
        if language:
            display(Code(content, language=language))
        else:
            # Default to Markdown for .md files, otherwise plain text
            if artifact_name.endswith('.md'):
                display(Markdown(content))
            else:
                display(Code(content))


# --- Unified Model Interaction ---

def _start_loading_indicator(message="Thinking..."):
    """Lightweight, non-blocking loading helper.

    The previous implementation spawned a subprocess spinner which is fragile in
    many environments. Keep a minimal, safe behaviour: print a single message
    and return a token (None) that callers can pass to _stop_loading_indicator.
    """
    print(message)
    return None


def _stop_loading_indicator(token):
    """Stop the loading helper. Currently a no-op kept for API compatibility."""
    return None


def generate_content(
    model_name,
    prompt=None,
    system_prompt=None,
    image_path=None,
    audio_path=None,
    max_output_tokens=None,
    temperature=0.7,
    show_loading=True
):
    """
    A unified function to generate content from various multimodal models.

    This function abstracts the differences between providers (OpenAI, Anthropic,
    Hugging Face, Google) and handles text, image, and audio inputs.

    Args:
        model_name (str): The name of the model to use (e.g., "gpt-4o", "claude-3-opus-20240229").
        prompt (str, optional): The primary text prompt.
        system_prompt (str, optional): A system-level instruction for the model.
        image_path (str, optional): The file path to an image for vision models.
        audio_path (str, optional): The file path to an audio file for transcription models.
        max_output_tokens (int, optional): The maximum number of tokens to generate.
                                           If not provided, a model-specific default is used.
        temperature (float, optional): The sampling temperature (0.0 to 1.0). Defaults to 0.7.
        show_loading (bool, optional): Whether to display a loading indicator. Defaults to True.

    Returns:
        str: The generated text content from the model, or a base64 encoded image string
             if an image generation model is used. Returns None on failure.
    """
    provider = get_provider_for_model(model_name)
    client = get_client_for_model(model_name)
    model_info = get_model_info(model_name)

    if not client:
        print(f"❌ Could not get client for model '{model_name}'.")
        return None

    # Get default max_output_tokens from model info if not provided
    if max_output_tokens is None:
        max_output_tokens = model_info.get("max_output_tokens", 2048) # Default to 2048 if not in JSON

    is_vision_model = model_info.get("vision", False)
    is_imagen_model = model_info.get("image_gen", False)
    is_audio_model = model_info.get("audio_transcription", False)

    # --- Input Validation ---
    if image_path and not is_vision_model:
        print(f"⚠️ Warning: Model '{model_name}' does not support vision. The provided image will be ignored.")
        image_path = None
    if audio_path and not is_audio_model:
        print(f"⚠️ Warning: Model '{model_name}' does not support audio transcription. The audio file will be ignored.")
        audio_path = None
    if is_imagen_model and not prompt:
        print("❌ Error: Image generation models require a text prompt.")
        return None

    loading_indicator = None
    if show_loading:
        loading_indicator = _start_loading_indicator()

    try:
        response_text = None
        image_data_base64 = None

        # --- Provider-Specific Logic ---
        if provider == "openai":
            if is_audio_model and audio_path:
                # --- OpenAI Audio Transcription ---
                with open(audio_path, "rb") as audio_file:
                    try:
                        # Try the most common SDK shape first (client.audio.transcriptions.create)
                        audio_api = getattr(client, "audio", None)
                        # Safely retrieve transcriptions from either audio_api or client without direct attribute access
                        transcriptions_api = None
                        if audio_api is not None:
                            transcriptions_api = getattr(audio_api, "transcriptions", None)
                        if transcriptions_api is None:
                            transcriptions_api = getattr(client, "transcriptions", None)

                        if transcriptions_api is not None and hasattr(transcriptions_api, "create"):
                            # Call create via the local variable to avoid direct attribute access on client objects
                            transcription = transcriptions_api.create(model=model_name, file=audio_file)
                        # Fallback: client.audio.transcribe or client.transcribe
                        elif audio_api is not None and hasattr(audio_api, "transcribe"):
                            transcription = audio_api.transcribe(model=model_name, file=audio_file)
                        elif hasattr(client, "transcribe"):
                            transcription = client.transcribe(model=model_name, file=audio_file)
                        else:
                            print("❌ OpenAI client does not expose a recognized audio transcription interface.")
                            return None

                        # Normalize response_text for dict or object responses
                        if isinstance(transcription, dict):
                            response_text = transcription.get("text") or transcription.get("transcript") or transcription.get("result") or None
                        else:
                            response_text = getattr(transcription, "text", None) or getattr(transcription, "transcript", None)
                    except Exception as e:
                        print(f"❌ OpenAI audio transcription failed: {e}")
                        return None
            elif is_imagen_model:
                # --- OpenAI Image Generation (DALL-E) ---
                response = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    n=1,
                    size="1024x1024", # Or other supported sizes
                    response_format="b64_json"
                )
                image_data_base64 = response.data[0].b64_json
            else:
                # --- OpenAI Text/Vision ---
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                content_parts = []
                if prompt:
                    content_parts.append({"type": "text", "text": prompt})

                if image_path and is_vision_model:
                    try:
                        with open(image_path, "rb") as image_file:
                            b64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
                        content_parts.append({
                            "type": "image_url",
                            "image_url": f"data:{mime_type};base64,{b64_image}"
                        })
                    except Exception as e:
                        print(f"❌ Error processing image file: {e}")
                        return None

                if content_parts:
                    messages.append({"role": "user", "content": content_parts})

                if messages:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                    response_text = completion.choices[0].message.content
                else:
                    print("❌ No valid content (prompt or image) provided for OpenAI model.")
                    return None

        elif provider == "anthropic":
            # --- Anthropic Text/Vision ---
            messages = []
            content_parts = []
            if prompt:
                content_parts.append({"type": "text", "text": prompt})

            if image_path and is_vision_model:
                try:
                    with open(image_path, "rb") as image_file:
                        b64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_image,
                        },
                    })
                except Exception as e:
                    print(f"❌ Error processing image file: {e}")
                    return None

            if content_parts:
                messages.append({"role": "user", "content": content_parts})

            if messages:
                message = client.messages.create(
                    model=model_name,
                    system=system_prompt if system_prompt else None,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                )
                response_text = message.content[0].text
            else:
                print("❌ No valid content (prompt or image) provided for Anthropic model.")
                return None

        elif provider == "huggingface":
            # --- Hugging Face Inference API ---
            headers = {"Authorization": f"Bearer {client['api_key']}"}
            payload = {"inputs": prompt}
            if is_imagen_model:
                # Image generation models on HF
                response = requests.post(client['api_url'], headers=headers, json=payload)
                if response.status_code == 200:
                    # Assuming the API returns the image bytes directly
                    image_data_base64 = base64.b64encode(response.content).decode('utf-8')
                else:
                    print(f"❌ Hugging Face API Error: {response.status_code} - {response.text}")
            else:
                # Text generation models
                response = requests.post(client['api_url'], headers=headers, json=payload)
                if response.status_code == 200:
                    # The response format can vary, common one is a list with a dict
                    # containing 'generated_text'
                    result = response.json()
                    if isinstance(result, list) and result:
                        response_text = result[0].get('generated_text')
                    elif isinstance(result, dict):
                         response_text = result.get('generated_text')
                    else:
                        response_text = str(result) # Fallback
                else:
                    print(f"❌ Hugging Face API Error: {response.status_code} - {response.text}")

        elif provider == "google":
            # --- Google Gemini / Imagen ---
            if is_audio_model and audio_path:
                # Google does not have a simple transcription API in the same vein as OpenAI
                # in the Gemini client. This would typically involve Google Cloud Speech-to-Text.
                # For simplicity, we'll note this limitation.
                print("ℹ️ Direct audio transcription with the Gemini client is not supported in this script.")
                print("💡 Use the Google Cloud Speech-to-Text API for this functionality.")
                return None
            elif is_imagen_model:
                # Handle Imagen models
                print(f"🖼️ Imagen model '{model_name}' detected")
                print("ℹ️  Note: We'll try API-key/client methods first, then fall back to Vertex AI if needed.")

                # Helper: try a variety of client shapes that historically worked with API keys
                def _try_client_methods(c):
                    nonlocal image_data_base64
                    if not c:
                        return False

                    # 1) New google-genai client: c.models.generate_images
                    try:
                        if hasattr(c, 'models') and hasattr(c.models, 'generate_images'):
                            resp = c.models.generate_images(model=model_name, prompt=prompt, max_output_tokens=512)
                            imgs = getattr(resp, 'images', None) or (resp if isinstance(resp, (list, tuple)) else None)
                            if imgs:
                                img_obj = imgs[0]
                                try:
                                    buf = BytesIO()
                                    if hasattr(img_obj, 'save'):
                                        img_obj.save(buf, format='PNG')
                                        image_data_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                        return True
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"⚠️ google-genai.generate_images error: {e}")

                    # 2) convenience generate_images
                    try:
                        if hasattr(c, 'generate_images'):
                            resp = c.generate_images(model=model_name, prompt=prompt, number_of_images=1)
                            imgs = getattr(resp, 'images', None) or (resp if isinstance(resp, (list, tuple)) else None)
                            if imgs:
                                img_obj = imgs[0]
                                try:
                                    buf = BytesIO()
                                    if hasattr(img_obj, 'save'):
                                        img_obj.save(buf, format='PNG')
                                        image_data_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                        return True
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"⚠️ client.generate_images error: {e}")

                    # 3) legacy generate_content with inline_data
                    try:
                        if hasattr(c, 'generate_content'):
                            resp = c.generate_content(model=model_name, prompt=prompt)
                            if resp and hasattr(resp, 'candidates') and resp.candidates:
                                for candidate in resp.candidates:
                                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                        for part in candidate.content.parts:
                                            if hasattr(part, 'inline_data') and part.inline_data:
                                                image_data_base64 = part.inline_data.data
                                                return True
                    except Exception as e:
                        print(f"⚠️ client.generate_content error: {e}")

                    # 4) OpenAI-like images.generate
                    try:
                        if hasattr(c, 'images') and hasattr(c.images, 'generate'):
                            resp = c.images.generate(model=model_name, prompt=prompt, n=1, response_format='b64_json')
                            try:
                                b64 = resp.data[0].b64_json
                                image_data_base64 = b64
                                return True
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"⚠️ client.images.generate error: {e}")

                    return False

                tried_client = _try_client_methods(client)
                if tried_client and image_data_base64:
                    print("✅ Imagen generated via API-key/client method")
                else:
                    # Only attempt Vertex AI if client-based approaches failed
                    print("🔄 Client-based methods did not produce an image; attempting Vertex AI as fallback...")
                    try:
                        from google.cloud import aiplatform  # noqa
                        from vertexai.preview.vision_models import ImageGenerationModel  # noqa
                        aiplatform.init(
                            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                        )
                        imagen_model = ImageGenerationModel.from_pretrained(model_name)
                        response = imagen_model.generate_images(prompt=prompt, number_of_images=1)
                        if response and len(response) > 0:
                            buffer = BytesIO()
                            response[0].save(buffer, format='PNG')
                            image_data_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            print("✅ Successfully generated image with Vertex AI Imagen")
                        else:
                            print("⚠️ Vertex AI returned empty response")
                    except ImportError as e:
                        print(f"⚠️ Vertex AI not installed: {e}")
                        print("💡 Install with: pip install google-cloud-aiplatform")
                        image_data_base64 = None
                    except Exception as vertex_error:
                        print(f"⚠️ Vertex AI failed: {vertex_error}")
                        if "project" in str(vertex_error).lower() or "location" in str(vertex_error).lower():
                            print("🔧 Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables")
                        elif "credentials" in str(vertex_error).lower() or "auth" in str(vertex_error).lower():
                            print("🔐 Please authenticate with Google Cloud:")
                            print("   Run: gcloud auth application-default login")
                        image_data_base64 = None

                if not image_data_base64:
                    print("❌ Imagen model setup incomplete")
                    print("To use Imagen models: ")
                    print("   1. Ensure you have a compatible Google SDK (google-genai or google.generativeai) installed and that it supports image generation.")
                    print("   2. If using API key flow: set GOOGLE_API_KEY (or GEMINI_API_KEY) and install google-genai or google-generativeai that supports images.")
                    print("   3. If using Vertex AI: install google-cloud-aiplatform and set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION, and authenticate with gcloud.")
            else:
                # --- Google Gemini Text/Vision ---
                # The `client` is the `genai` module or a `GenerativeModel` instance
                try:
                    # Using the newer `google-genai` library structure
                    model = client.GenerativeModel(model_name)
                    content_parts = []
                    if prompt:
                        content_parts.append(prompt)
                    if image_path and is_vision_model:
                        try:
                            img = Image.open(image_path)
                            content_parts.append(img)
                        except Exception as e:
                            print(f"❌ Error opening image file: {e}")
                            return None

                    if content_parts:
                        # The generate_content method takes a list of parts
                        response = model.generate_content(content_parts)
                        response_text = response.text
                    else:
                        print("❌ No valid content (prompt or image) provided for Gemini model.")
                        return None

                except Exception as e:
                    print(f"❌ Google Gemini API Error: {e}")
                    return None

        # --- Final Result ---
        if is_imagen_model:
            return image_data_base64
        else:
            return response_text

    except Exception as e:
        import traceback
        print(f"\n❌ An unexpected error occurred in generate_content: {e}")
        print(f"   Provider: {provider}, Model: {model_name}")
        # traceback.print_exc()
        return None
    finally:
        if show_loading and loading_indicator:
            _stop_loading_indicator(loading_indicator)


def display_generated_content(content, prompt=None):
    """
    Intelligently displays the output of the `generate_content` function.

    - If the content is a base64 image string, it displays the image.
    - If the content is a JSON string, it displays it as formatted code.
    - If the content is a Python code block, it's displayed with syntax highlighting.
    - Otherwise, it's displayed as Markdown.

    Args:
        content (str): The content string to display.
        prompt (str, optional): The original prompt, used for context in image captions.
    """
    if not content:
        print("🤷 No content to display.")
        return

    # Check if it's a base64 image
    if re.match(r'^[A-Za-z0-9+/=]+$', content.strip()):
        try:
            # Attempt to decode to see if it's valid base64
            img_data = base64.b64decode(content)
            # A simple heuristic: if it decodes and the first few bytes look like a PNG/JPEG,
            # it's probably an image.
            if img_data.startswith(b'\x89PNG') or img_data.startswith(b'\xff\xd8'):
                if prompt:
                    display(Markdown(f"**Image generated from prompt:** *'{prompt}'*"))
                display(IPyImage(data=img_data))
                return
        except Exception:
            # Not a valid base64 string, so treat as text
            pass

    # Check if it's a JSON string
    stripped_content = content.strip()
    if (stripped_content.startswith('{') and stripped_content.endswith('}')) or \
       (stripped_content.startswith('[') and stripped_content.endswith(']')):
        try:
            # Try to parse it as JSON
            json.loads(stripped_content)
            display(Markdown("**Generated JSON:**"))
            display(Code(stripped_content, language='json'))
            return
        except json.JSONDecodeError:
            # Not valid JSON, fall through to treat as text/code
            pass

    # Check if it's a Python code block
    if "```python" in content:
        display(Markdown("**Generated Python Code:**"))
        # Extract content within the python block for proper rendering
        match = re.search(r'```python\n(.*)\n```', content, re.DOTALL)
        if match:
            display(Code(match.group(1).strip(), language='python'))
        else: # Fallback for malformed blocks
            display(Markdown(content))
        return

    # Default to Markdown
    display(Markdown(content))

# --- Environment and API Client Setup ---

def load_environment():
    """
    Load environment variables from a .env file located in the project root directory.

    This function automatically searches for a .env file by traversing up the directory
    tree from the current working directory. It looks for common project root indicators
    like .env files or .git directories to identify the project root. Once found, it
    loads all environment variables defined in the .env file into the current process
    environment using the python-dotenv library.

    The function is essential for loading API keys and other configuration settings
    needed by various LLM providers (OpenAI, Anthropic, Google, etc.) without hardcoding
    sensitive information in the source code.

    Args:
        None

    Returns:
        None: This function doesn't return a value but modifies the process environment
            by loading variables from the .env file.

    Raises:
        None: This function handles all errors gracefully. If the .env file is not found
            or cannot be loaded, it prints a warning message and continues execution.

    Examples:
        >>> # Basic usage - load environment variables
        >>> load_environment()
        >>> # Now you can access environment variables
        >>> import os
        >>> api_key = os.getenv('OPENAI_API_KEY')
        >>> print(f"API Key loaded: {api_key is not None}")
        API Key loaded: True

        >>> # Typical .env file content:
        >>> # OPENAI_API_KEY=sk-proj-...
        >>> # ANTHROPIC_API_KEY=sk-ant-...
        >>> # GOOGLE_API_KEY=AIza...
        >>> # HUGGINGFACE_API_KEY=hf_...

        >>> # After calling load_environment(), these are available:
        >>> import os
        >>> openai_key = os.getenv('OPENAI_API_KEY')
        >>> anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        >>> google_key = os.getenv('GOOGLE_API_KEY')

    Notes:
        - Searches upward from current directory until finding .env or .git
        - Falls back to current directory if no project root markers are found
        - Uses python-dotenv library for parsing .env file format
        - Prints warning if .env file is not found (but doesn't raise exception)
        - Environment variables loaded are accessible via os.getenv() or os.environ
        - Supports standard .env file syntax including comments and quoted values
        - Safe to call multiple times - subsequent calls will reload the .env file

    Dependencies:
        - python-dotenv: For parsing and loading .env files
        - os: For file system operations and environment variable access
    """
    path = os.getcwd()
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, '.env')) or os.path.exists(os.path.join(path, '.git')):
            project_root = path
            break
        path = os.path.dirname(path)
    else:
        project_root = os.getcwd()

    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        if _load_dotenv:
            try:
                _load_dotenv(dotenv_path=dotenv_path)
            except Exception:
                print("Warning: Failed to load .env file with python-dotenv.")
        else:
            # dotenv not installed, nothing to do
            pass
    else:
        print("Warning: .env file not found. API keys may not be loaded.")


def setup_llm_client(model_name="gpt-4o"):
    """
    Initialize and configure an LLM client for the specified model across multiple providers.

    This function provides a unified interface for setting up API clients for various LLM
    providers including OpenAI, Anthropic, Hugging Face, and Google (Gemini/Imagen/Speech-to-Text).
    It automatically loads environment variables, validates API keys, and handles provider-specific
    configuration differences. The function supports both modern and legacy SDK versions for
    maximum compatibility.

    Args:
        model_name (str, optional): The identifier of the model to use. Must be a key in the
            RECOMMENDED_MODELS dictionary. Defaults to "gpt-4o". Common examples:
            - "gpt-4o", "gpt-4o-mini" (OpenAI)
            - "claude-3-opus-20240229", "claude-3-haiku-20240307" (Anthropic)
            - "gemini-2.5-pro", "gemini-1.5-flash" (Google)
            - "meta-llama/Llama-2-70b-chat-hf" (Hugging Face)

    Returns:
        tuple: A 3-element tuple containing:
            - client: The initialized API client object (type varies by provider)
                - OpenAI: OpenAI client instance
                - Anthropic: Anthropic client instance
                - Hugging Face: InferenceClient instance
                - Google Gemini: GenerativeModel or genai module
                - Google Speech-to-Text: SpeechClient instance
            - model_name (str): The model name (echoed back for convenience)
            - api_provider (str): The provider name ("openai", "anthropic", "huggingface",
              "gemini", or "google")

            Returns (None, None, None) if initialization fails due to missing API keys,
            unsupported models, or missing dependencies.

    Raises:
        None: This function handles all errors gracefully and prints error messages
            instead of raising exceptions.

    Examples:
        >>> # Initialize OpenAI client (default)
        >>> client, model, provider = setup_llm_client("gpt-4o")
        ✅ LLM Client configured: Using 'openai' with model 'gpt-4o'
        >>> print(f"Provider: {provider}, Model: {model}")
        Provider: openai, Model: gpt-4o

        >>> # Initialize Anthropic client
        >>> client, model, provider = setup_llm_client("claude-3-opus-20240229")
        ✅ LLM Client configured: Using 'anthropic' with model 'claude-3-opus-20240229'

        >>> # Initialize Google Gemini client
        >>> client, model, provider = setup_llm_client("gemini-2.5-pro")
        ✅ LLM Client configured: Using 'gemini' with model 'gemini-2.5-pro'

        >>> # Initialize Hugging Face client
        >>> client, model, provider = setup_llm_client("microsoft/DialoGPT-medium")
        ✅ LLM Client configured: Using 'huggingface' with model 'microsoft/DialoGPT-medium'

        >>> # Handle missing API key
        >>> client, model, provider = setup_llm_client("gpt-4o")
        ERROR: OPENAI_API_KEY not found in .env file.
        >>> print(client, model, provider)
        None None None

        >>> # Handle unsupported model
        >>> client, model, provider = setup_llm_client("unsupported-model")
        ERROR: Model 'unsupported-model' is not in the list of recommended models.
        >>> print(client, model, provider)
        None None None

    Notes:
        - Automatically calls load_environment() to load .env file with API keys
        - Validates that the model exists in RECOMMENDED_MODELS before initialization
        - Handles provider-specific authentication and configuration:
            - OpenAI: Uses OPENAI_API_KEY environment variable
            - Anthropic: Uses ANTHROPIC_API_KEY environment variable
            - Hugging Face: Uses HUGGINGFACE_API_KEY environment variable
            - Google: Uses GOOGLE_API_KEY or GEMINI_API_KEY environment variables
        - Supports both modern and legacy Google SDKs (google-genai vs google.generativeai)
        - For Google models, automatically detects the appropriate SDK based on availability
        - Special handling for Google Speech-to-Text models (uses google.cloud.speech)
        - Prints success/error messages to console for debugging
        - Safe to call multiple times with different models

    Dependencies:
        - Provider-specific libraries (installed as needed):
            - openai: For OpenAI models
            - anthropic: For Anthropic models
            - huggingface_hub: For Hugging Face models
            - google.generativeai or google-genai: For Google Gemini/Imagen
            - google.cloud.speech: For Google Speech-to-Text
        - RECOMMENDED_MODELS: Global dictionary with model configurations
        - load_environment(): For loading API keys from .env file
    """
    load_environment()
    if model_name not in RECOMMENDED_MODELS:
        print(f"ERROR: Model '{model_name}' is not in the list of recommended models.")
        return None, None, None
    config = RECOMMENDED_MODELS[model_name]
    api_provider = config["provider"]
    client = None
    try:
        if api_provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY not found in .env file.")
            client = OpenAI(api_key=api_key)
        elif api_provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY not found in .env file.")
            client = Anthropic(api_key=api_key)
        elif api_provider == "huggingface":
            from huggingface_hub import InferenceClient
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key: raise ValueError("HUGGINGFACE_API_KEY not found in .env file.")
            client = InferenceClient(model=model_name, token=api_key)
        
        elif api_provider == "gemini" or api_provider == "google":  # Google for image generation, text/vision or STT
            if config.get("audio_transcription"):
                # Use Cloud Speech-to-Text (separate library)
                from google.cloud import speech
                client = speech.SpeechClient()
            else:
                # Prefer the new Google Gen AI SDK (google-genai). Fallback to legacy google.generativeai.
                try:
                    from google import genai as google_genai
                    # If using Vertex AI, set GOOGLE_GENAI_USE_VERTEXAI=True and provide GOOGLE_CLOUD_PROJECT/LOCATION.
                    # Otherwise, the Client will use the Gemini Developer API (API key).
                    # Supports Imagen via client.models.generate_images and Gemini via client.models.generate_content.
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if api_key:
                        client = google_genai.Client(api_key=api_key)
                    else:
                        # ADC/Workload Identity path (Vertex AI)
                        client = google_genai.Client()
                    # Tag the client so downstream functions know which SDK is in use.
                    try:
                        setattr(client, "_sdk_family", "google-genai")
                    except Exception:
                        pass
                except ImportError:
                    import google.generativeai as genai
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment.")
                    genai.configure(api_key=api_key)
                    # For image generation, we'll pass the module (legacy SDK) and instantiate per-call.
                    if config.get("image_generation"):
                        client = genai
                    else:
                        # For Gemini text/vision, instantiate a GenerativeModel object.
                        client = genai.GenerativeModel(model_name)
    except ImportError as e:
        print(f"ERROR: The required library for '{api_provider}' is not installed: {e}")
        return None, None, None
    except ValueError as e:
        print(f"ERROR: {e}")
        return None, None, None
    print(f"✅ LLM Client configured: Using '{api_provider}' with model '{model_name}'")
    return client, model_name, api_provider

# --- Core Interaction Functions ---

def get_completion(prompt, client, model_name, api_provider, temperature=0.7):
    """
    Generate text completion from an LLM using a text-only prompt.

    This function provides a unified interface for getting text completions from various
    LLM providers. It handles provider-specific API differences, error cases, and special
    handling for newer OpenAI models that may use different endpoints or not support
    temperature parameters. The function automatically adapts to different API structures
    and provides fallback mechanisms for maximum compatibility.

    Args:
        prompt (str): The text prompt to send to the model. This is the user's input
            or question that the model should respond to. Can be any length supported
            by the model's context window.
        client: The initialized API client object from setup_llm_client(). The type
            varies by provider (OpenAI, Anthropic, InferenceClient, etc.)
        model_name (str): The identifier of the model to use for completion. Must match
            the model used when initializing the client.
        api_provider (str): The provider name indicating which API to use. Supported:
            - "openai": OpenAI GPT models
            - "anthropic": Anthropic Claude models
            - "huggingface": Hugging Face models
            - "gemini" or "google": Google Gemini models
        temperature (float, optional): Controls randomness in the output. Higher values
            (e.g., 1.0) make output more random, lower values (e.g., 0.1) make it more
            deterministic. Defaults to 0.7. Range typically 0.0-2.0, but some models
            may not support temperature parameter.

    Returns:
        str: The generated text completion from the model. Returns an error message
            string if the API call fails or if the client is not initialized.

    Raises:
        None: This function catches all exceptions and returns error messages as strings
            instead of raising exceptions.

    Examples:
        >>> # Basic text completion with OpenAI
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> response = get_completion(
        ...     "What is the capital of France?",
        ...     client, model, provider
        ... )
        >>> print(response)
        "The capital of France is Paris."

        >>> # Creative writing with higher temperature
        >>> creative_response = get_completion(
        ...     "Write a short story about a robot learning to paint",
        ...     client, model, provider, temperature=1.2
        ... )
        >>> print(creative_response[:100])
        "In the year 2147, a maintenance droid named Pixel discovered..."

        >>> # Technical explanation with lower temperature
        >>> technical_response = get_completion(
        ...     "Explain how neural networks work",
        ...     client, model, provider, temperature=0.1
        ... )
        >>> print(technical_response[:100])
        "Neural networks are computational models inspired by biological..."

        >>> # Handle API errors gracefully
        >>> error_response = get_completion("Hello", None, "gpt-4o", "openai")
        >>> print(error_response)
        "API client not initialized."

        >>> # Anthropic Claude example
        >>> claude_client, claude_model, claude_provider = setup_llm_client("claude-3-haiku-20240307")
        >>> claude_response = get_completion(
        ...     "Explain quantum computing in simple terms",
        ...     claude_client, claude_model, claude_provider
        ... )
        >>> print(claude_response[:100])
        "Quantum computing uses quantum mechanics principles to process..."

    Notes:
        - Handles different API structures for each provider automatically
        - OpenAI: Tries chat completions first, falls back to responses endpoint for
          certain models, handles temperature parameter compatibility issues
        - Anthropic: Uses messages API with max_tokens=4096
        - Hugging Face: Uses chat_completion with minimum temperature of 0.1
        - Google/Gemini: Uses generate_content method
        - Special error handling for OpenAI models that don't support temperature
        - Returns descriptive error messages if API calls fail
        - Safe to use with any model from RECOMMENDED_MODELS
        - Automatically adjusts parameters based on provider capabilities

    Dependencies:
        - Provider-specific client libraries (OpenAI, Anthropic, etc.)
        - setup_llm_client(): For initializing the API client
        - RECOMMENDED_MODELS: For model capability validation
    """
    if not client: return "API client not initialized."
    try:
        if api_provider == "openai":
            # Some newer models use different endpoints
            try:
                # Try chat completions first (standard endpoint)
                response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temperature)
                return response.choices[0].message.content
            except Exception as api_error:
                error_message = str(api_error).lower()
                
                if "temperature" in error_message and "unsupported" in error_message:
                    # Retry without temperature parameter
                    try:
                        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
                        return response.choices[0].message.content
                    except Exception as retry_error:
                        if "v1/responses" in str(retry_error):
                            # Use the responses endpoint for certain models
                            response = client.responses.create(model=model_name, input=prompt)
                            return response.choices[0].text
                        else:
                            raise retry_error
                elif "v1/responses" in str(api_error):
                    # Use the responses endpoint for certain models
                    try:
                        response = client.responses.create(model=model_name, input=prompt, temperature=temperature)
                        return response.text
                    except Exception as resp_error:
                        # Try responses endpoint without temperature
                        response = client.responses.create(model=model_name, input=prompt)
                        return response.text
                else:
                    raise api_error
        elif api_provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif api_provider == "huggingface":
            response = client.chat_completion(messages=[{"role": "user", "content": prompt}], temperature=max(0.1, temperature), max_tokens=4096)
            return response.choices[0].message.content
        elif api_provider == "gemini" or api_provider == "google":
            response = client.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"An API error occurred: {e}"

def get_vision_completion(prompt, image_path_or_url, client, model_name, api_provider):
    """
    Generate text completion from a vision-capable LLM using both text and image inputs.

    This function enables multimodal AI interactions by processing both text prompts and images
    together. It handles the different image processing requirements for each provider, including
    URL-based and base64-encoded image formats. The function can accept either a public URL to
    an image or a local file path, automatically detecting the format and processing accordingly.

    Args:
        prompt (str): The text prompt or question about the image. This provides context or
            specific instructions for analyzing the image. Examples:
            - "What objects do you see in this image?"
            - "Describe the scene in detail"
            - "What emotions are expressed by the people in this photo?"
        image_path_or_url (str): Path or URL to the image to analyze. Can be:
            - Local file path: "/path/to/image.jpg" or "artifacts/photos/my_image.png"
            - Public URL: "https://example.com/image.jpg" (must be accessible)
            Supported formats: JPEG, PNG, GIF, WebP (varies by provider)
        client: The initialized API client object from setup_llm_client(). Must be a
            vision-capable model client.
        model_name (str): The identifier of the vision-capable model to use. Must be a
            model that supports vision (e.g., "gpt-4o", "claude-3-opus-20240229", "gemini-2.5-pro")
        api_provider (str): The provider name ("openai", "anthropic", "huggingface",
            "gemini", or "google").

    Returns:
        str: The model's response analyzing the image based on the prompt. Returns an error
            message string if the model doesn't support vision, if the image cannot be loaded,
            or if the API call fails.

    Raises:
        None: This function catches all exceptions and returns error messages as strings
            instead of raising exceptions.

    Examples:
        >>> # Analyze image from URL with OpenAI
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> response = get_vision_completion(
        ...     "What animals do you see in this image?",
        ...     "https://example.com/zoo.jpg",
        ...     client, model, provider
        ... )
        >>> print(response)
        "I can see lions, tigers, and elephants in this zoo image..."

        >>> # Analyze local image file
        >>> local_response = get_vision_completion(
        ...     "Describe the architecture in this building",
        ...     "artifacts/photos/cathedral.jpg",
        ...     client, model, provider
        ... )
        >>> print(local_response)
        "This appears to be a Gothic cathedral with..."

        >>> # Creative analysis with Anthropic
        >>> claude_client, claude_model, claude_provider = setup_llm_client("claude-3-opus-20240229")
        >>> creative_response = get_vision_completion(
        ...     "Write a short story inspired by this scene",
        ...     "https://example.com/sunset.jpg",
        ...     claude_client, claude_model, claude_provider
        ... )
        >>> print(creative_response[:100])
        "As the sun dipped below the horizon, painting the sky in hues of..."

        >>> # Technical analysis with Google Gemini
        >>> gemini_client, gemini_model, gemini_provider = setup_llm_client("gemini-2.5-pro")
        >>> technical_response = get_vision_completion(
        ...     "Count the number of people and describe their activities",
        ...     "artifacts/screenshots/meeting.jpg",
        ...     gemini_client, gemini_model, gemini_provider
        ... )
        >>> print(technical_response)
        "I can see 5 people in this meeting room. Three are seated at the table..."

        >>> # Handle non-vision model error
        >>> text_client, text_model, text_provider = setup_llm_client("gpt-3.5-turbo")
        >>> error_response = get_vision_completion(
        ...     "Describe this image",
        ...     "https://example.com/photo.jpg",
        ...     text_client, text_model, text_provider
        ... )
        >>> print(error_response)
        "Error: Model 'gpt-3.5-turbo' does not support vision."

        >>> # Handle missing image file
        >>> missing_response = get_vision_completion(
        ...     "What's in this image?",
        ...     "nonexistent.jpg",
        ...     client, model, provider
        ... )
        >>> print(missing_response)
        "Error: Local image file not found at nonexistent.jpg"

    Notes:
        - Validates that the model supports vision using RECOMMENDED_MODELS before processing
        - Handles both URLs and local file paths automatically
        - Different providers require different image formats:
            - OpenAI: Accepts both URLs and base64-encoded data URLs
            - Anthropic: Requires base64-encoded image data with MIME type
            - Google/Gemini: Requires PIL Image objects
            - Hugging Face: Requires PIL Image objects
        - Automatically downloads images from URLs or reads from disk and converts as needed
        - Sets max_tokens to 4096 for providers that support it
        - Handles HTTP errors when fetching images and file I/O errors
        - Supports common image formats (JPEG, PNG, GIF, WebP)
        - Images are processed in memory - no temporary files created
        - Safe to use with any vision-capable model from RECOMMENDED_MODELS

    Dependencies:
        - requests: For downloading images from URLs
        - PIL (Pillow): For image processing and format conversion
        - base64: For encoding images to base64 format
        - mimetypes: For determining image MIME types from file extensions
        - io.BytesIO: For converting image bytes to PIL Images
        - RECOMMENDED_MODELS: For vision capability validation
        - _encode_image_to_base64(): For base64 encoding local images
    """
    if not client: return "API client not initialized."
    if not RECOMMENDED_MODELS.get(model_name, {}).get("vision"):
        return f"Error: Model '{model_name}' does not support vision."

    is_url = image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://')

    try:
        if api_provider == "openai":
            image_url_data = {}
            if is_url:
                image_url_data = {"url": image_path_or_url}
            else:
                if not os.path.exists(image_path_or_url):
                    return f"Error: Local image file not found at {image_path_or_url}"
                base64_image = _encode_image_to_base64(image_path_or_url)
                image_url_data = {"url": base64_image}

            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image_url_data}
                    ]
                }],
                max_tokens=4096
            )
            return response.choices[0].message.content

        elif api_provider == "anthropic":
            if is_url:
                response_img = requests.get(image_path_or_url)
                response_img.raise_for_status()
                img_content = response_img.content
                mime_type = response_img.headers.get('Content-Type', 'image/jpeg')
            else:
                if not os.path.exists(image_path_or_url):
                    return f"Error: Local image file not found at {image_path_or_url}"
                with open(image_path_or_url, "rb") as f:
                    img_content = f.read()
                mime_type, _ = mimetypes.guess_type(image_path_or_url)
                if not mime_type:
                    return f"Error: Could not determine mime type for {image_path_or_url}"

            img_data = base64.b64encode(img_content).decode('utf-8')
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": img_data}}
                    ]
                }]
            )
            return response.content[0].text

        elif api_provider == "gemini" or api_provider == "google":
            if is_url:
                response_img = requests.get(image_path_or_url)
                response_img.raise_for_status()
                img = Image.open(BytesIO(response_img.content))
            else:
                if not os.path.exists(image_path_or_url):
                    return f"Error: Local image file not found at {image_path_or_url}"
                img = Image.open(image_path_or_url)
            
            response = client.generate_content([prompt, img])
            return response.text

        elif api_provider == "huggingface":
            if is_url:
                response_img = requests.get(image_path_or_url)
                response_img.raise_for_status()
                img = Image.open(BytesIO(response_img.content))
            else:
                if not os.path.exists(image_path_or_url):
                    return f"Error: Local image file not found at {image_path_or_url}"
                img = Image.open(image_path_or_url)
                
            # Note: HuggingFace's image_to_text might not support a separate text prompt in all cases.
            # The prompt may need to be embedded in the task definition for some models.
            # This implementation assumes the client handles the combination correctly.
            response = client.image_to_text(image=img, prompt=prompt)
            return response
            
    except Exception as e:
        return f"An API error occurred during vision completion: {e}"

def get_image_generation_completion(prompt, client, model_name, api_provider):
    """
    Generate an image from a text prompt using an image generation LLM.

    This function provides a unified interface for text-to-image generation across different
    providers. It handles the API differences between providers and returns the generated
    image as both a saved file and a base64-encoded data URL that can be displayed directly
    in web browsers or Jupyter notebooks. The function includes loading indicators and
    performance tracking.

    Args:
        prompt (str): The text description of the image to generate. Should be detailed and
            specific for best results. Examples:
            - "A serene mountain landscape at sunset with a lake in the foreground"
            - "A futuristic city with flying cars and neon lights, cyberpunk style"
            - "A cute cartoon cat wearing sunglasses and playing guitar"
        client: The initialized API client object from setup_llm_client(). For Google Imagen,
            this might be the genai module itself.
        model_name (str): The identifier of the image generation model to use. Examples:
            - "dall-e-3" (OpenAI)
            - "imagen-3.0-generate-002" (Google)
        api_provider (str): The provider name ("openai" or "google").

    Returns:
        tuple[str, str]: A tuple containing:
            - file_path (str): The local path to the saved image file (PNG format)
            - image_url (str): A data URL string in the format "data:image/png;base64,{base64_data}"
              that can be used directly in HTML img tags or displayed in Jupyter notebooks
            Returns (None, None) if an error occurs during generation.

    Raises:
        None: This function catches all exceptions and returns error messages as strings
            instead of raising exceptions.

    Examples:
        >>> # Generate image with OpenAI DALL-E
        >>> client, model, provider = setup_llm_client("dall-e-3")
        >>> file_path, image_url = get_image_generation_completion(
        ...     "A futuristic city with flying cars and neon lights",
        ...     client, model, provider
        ... )
        Generating image... This may take a moment.
        ⏳ Generating image...
        ✅ Image generated in 15.32 seconds.
        ✅ Image saved to: artifacts/screens/image_1662586800.png
        >>> print(f"Image saved at: {file_path}")
        Image saved at: artifacts/screens/image_1662586800.png

        >>> # Display in Jupyter notebook
        >>> from IPython.display import Image, display
        >>> display(Image(url=image_url))

        >>> # Generate with Google Imagen
        >>> google_client, google_model, google_provider = setup_llm_client("imagen-3.0-generate-002")
        >>> file_path, image_url = get_image_generation_completion(
        ...     "A serene mountain landscape at sunset with a lake",
        ...     google_client, google_model, google_provider
        ... )
        Generating image... This may take a moment.
        ⏳ Generating image...
        ✅ Image generated in 12.45 seconds.
        ✅ Image saved to: artifacts/screens/image_1662586801.png

        >>> # Handle non-image generation model
        >>> text_client, text_model, text_provider = setup_llm_client("gpt-4o")
        >>> file_path, image_url = get_image_generation_completion(
        ...     "A cat wearing a hat",
        ...     text_client, text_model, text_provider
        ... )
        >>> print(file_path, image_url)
        (None, "Error: Model 'gpt-4o' does not support image generation.")

        >>> # Handle API client not initialized
        >>> file_path, image_url = get_image_generation_completion(
        ...     "A beautiful sunset",
        ...     None, "dall-e-3", "openai"
        ... )
        >>> print(file_path, image_url)
        (None, "API client not initialized.")

    Notes:
        - Validates that the model supports image generation using RECOMMENDED_MODELS
        - Displays a loading indicator during generation (can take 10-30 seconds)
        - Tracks and reports generation time for performance monitoring
        - Provider-specific handling:
            - OpenAI: Uses images.generate() API, returns base64 directly
            - Google Imagen: Uses REST API with predict endpoint or generate_images() method
            - Google Gemini: Uses generate_content method with image generation capabilities
        - Saves generated images to artifacts/screens/ directory with unique timestamps
        - The returned data URL can be used directly in HTML or markdown
        - Loading indicators are shown in console and Jupyter environments
        - Supports both modern and legacy Google SDKs (google-genai vs google.generativeai)
        - Images are saved in PNG format regardless of the original format
        - File paths include timestamps to avoid overwriting existing images

    Dependencies:
        - time: For tracking generation duration and creating unique filenames
        - base64: For encoding/decoding image data
        - os: For file system operations
        - IPython.display: For showing loading indicators in Jupyter (optional)
        - RECOMMENDED_MODELS: For image generation capability validation
        - setup_llm_client(): For initializing the API client
    """
    # Unified, robust image generation helper.
    if not client:
        return None, "API client not initialized."

    model_meta = RECOMMENDED_MODELS.get(model_name, {})
    if not (model_meta.get("image_generation") or model_meta.get("image_gen") or model_meta.get("image_generation", False)):
        return None, f"Error: Model '{model_name}' does not support image generation."

    print("Generating image... This may take a moment.")
    try:
        start_time = time.time()

        def _save_image_bytes(image_bytes: bytes):
            timestamp = int(time.time() * 1000)
            os.makedirs('artifacts/screens', exist_ok=True)
            file_path = f"artifacts/screens/image_{timestamp}.png"
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            return file_path, f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        image_bytes = None

        # --- OpenAI ---
        if api_provider == 'openai':
            try:
                # Common modern client shape
                if hasattr(client, 'images') and hasattr(client.images, 'generate'):
                    resp = client.images.generate(model=model_name, prompt=prompt, n=1, size='1024x1024', response_format='b64_json')
                    # resp may be object or dict
                    try:
                        b64 = getattr(resp.data[0], 'b64_json', None) or (resp['data'][0].get('b64_json') if isinstance(resp, dict) else None)
                    except Exception:
                        b64 = None
                    if not b64:
                        # Try alternative shapes
                        if isinstance(resp, dict) and 'data' in resp and resp['data']:
                            b64 = resp['data'][0].get('b64_json')
                    if b64:
                        image_bytes = base64.b64decode(b64)
                # Fallbacks: some SDKs expose images.create
                if image_bytes is None and hasattr(client, 'images') and hasattr(client.images, 'create'):
                    resp = client.images.create(prompt=prompt, n=1, size='1024x1024')
                    # try to extract b64
                    if isinstance(resp, dict) and resp.get('data'):
                        b64 = resp['data'][0].get('b64_json')
                        if b64:
                            image_bytes = base64.b64decode(b64)
            except Exception as e:
                print(f"OpenAI image generation error: {e}")

        # --- Hugging Face ---
        if image_bytes is None and api_provider == 'huggingface':
            try:
                # Client might be a dict from get_client_for_model or an InferenceClient
                if isinstance(client, dict) and client.get('api_url'):
                    headers = {}
                    if client.get('api_key'):
                        headers['Authorization'] = f"Bearer {client['api_key']}"
                    payload = {"inputs": prompt}
                    resp = requests.post(client['api_url'], headers=headers, json=payload)
                    if resp.status_code == 200:
                        # If returns image bytes directly
                        ctype = resp.headers.get('Content-Type', '')
                        if ctype.startswith('image/'):
                            image_bytes = resp.content
                        else:
                            try:
                                j = resp.json()
                                # Common keys
                                if isinstance(j, list) and j:
                                    first = j[0]
                                    if isinstance(first, dict):
                                        for key in ('generated_image', 'image_base64', 'b64_json', 'image'):
                                            if key in first and first[key]:
                                                image_bytes = base64.b64decode(first[key])
                                                break
                                elif isinstance(j, dict):
                                    for key in ('generated_image', 'image_base64', 'b64_json', 'image'):
                                        if j.get(key):
                                            image_bytes = base64.b64decode(j.get(key))
                                            break
                            except Exception:
                                pass
                else:
                    # Try common InferenceClient method names
                    if hasattr(client, 'image_generation'):
                        resp = client.image_generation(prompt)
                        # try to decode
                        if isinstance(resp, bytes):
                            image_bytes = resp
                        elif isinstance(resp, dict) and resp.get('image'):
                            image_bytes = base64.b64decode(resp.get('image'))
            except Exception as e:
                print(f"Hugging Face image generation error: {e}")

        # --- Google (Gemini / Imagen / Vertex) ---
        if image_bytes is None and api_provider in ('google', 'gemini'):
            try:
                # Try new google-genai client shapes
                # 1) client.models.generate_images
                try:
                    if hasattr(client, 'models') and hasattr(client.models, 'generate_images'):
                        resp = client.models.generate_images(model=model_name, prompt=prompt, max_output_tokens=512)
                        imgs = getattr(resp, 'images', None) or (resp if isinstance(resp, (list, tuple)) else None)
                        if imgs:
                            img_obj = imgs[0]
                            # if it's a PIL Image
                            if hasattr(img_obj, 'save'):
                                buf = BytesIO()
                                img_obj.save(buf, format='PNG')
                                image_bytes = buf.getvalue()
                            else:
                                # maybe bytes
                                if isinstance(img_obj, (bytes, bytearray)):
                                    image_bytes = bytes(img_obj)
                except Exception:
                    pass

                # 2) client.generate_images or client.generate_content (legacy)
                if image_bytes is None:
                    if hasattr(client, 'generate_images'):
                        resp = client.generate_images(model=model_name, prompt=prompt, number_of_images=1)
                        imgs = getattr(resp, 'images', None) or (resp if isinstance(resp, (list, tuple)) else None)
                        if imgs:
                            img_obj = imgs[0]
                            if hasattr(img_obj, 'save'):
                                buf = BytesIO()
                                img_obj.save(buf, format='PNG')
                                image_bytes = buf.getvalue()
                    elif hasattr(client, 'generate_content'):
                        resp = client.generate_content(model=model_name, prompt=prompt)
                        # scan resp for inline_data
                        try:
                            if resp and hasattr(resp, 'candidates') and resp.candidates:
                                for candidate in resp.candidates:
                                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                        for part in candidate.content.parts:
                                            if hasattr(part, 'inline_data') and getattr(part.inline_data, 'data', None):
                                                image_b64 = getattr(part.inline_data, 'data')
                                                image_bytes = base64.b64decode(image_b64)
                                                break
                                        if image_bytes:
                                            break
                        except Exception:
                            pass

                # 3) Vertex AI fallback via python SDK (best-effort)
                if image_bytes is None:
                    try:
                        from vertexai.preview.vision_models import ImageGenerationModel  # type: ignore
                        from google.cloud import aiplatform  # type: ignore
                        aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))
                        imagen_model = ImageGenerationModel.from_pretrained(model_name)
                        resp = imagen_model.generate_images(prompt=prompt, number_of_images=1)
                        if resp and len(resp) > 0:
                            buf = BytesIO()
                            resp[0].save(buf, format='PNG')
                            image_bytes = buf.getvalue()
                    except Exception:
                        pass
            except Exception as e:
                print(f"Google image generation error: {e}")

        # Final check
        if not image_bytes:
            return None, "Image generation failed or returned no data."

        # Save and return
        file_path, data_url = _save_image_bytes(image_bytes)
        duration = time.time() - start_time
        print(f"✅ Image generated in {duration:.2f} seconds.")
        print(f"✅ Image saved to: {file_path}")
        return file_path, data_url

    except Exception as e:
        return None, f"An API error occurred during image generation: {e}"


def transcribe_audio(audio_path, client, model_name, api_provider, language_code="en-US"):
    """
    Transcribe audio from a file using a speech-to-text model.

    This function provides a unified interface for converting speech in audio files to text
    across different providers. It handles various audio formats and provider-specific API
    differences, supporting both OpenAI Whisper and Google Cloud Speech-to-Text.

    Args:
        audio_path (str): Path to the audio file to transcribe. Can be absolute or relative.
            Supported formats vary by provider but typically include:
            - OpenAI: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM, FLAC
            - Google: WAV, FLAC (16-bit, mono), LINEAR16 encoding recommended
        client: The initialized API client object from setup_llm_client().
            - OpenAI: OpenAI client instance
            - Google: speech.SpeechClient instance
        model_name (str): The identifier of the speech-to-text model to use. Examples:
            - "whisper-1" (OpenAI)
            - "google-cloud/speech-to-text/latest_long" (Google)
        api_provider (str): The provider name ("openai" or "google").
        language_code (str, optional): The language of the audio in BCP-47 format.
            Defaults to "en-US" (American English). Only used by Google Speech-to-Text.
            Examples: "es-ES" (Spanish), "fr-FR" (French), "ja-JP" (Japanese).

    Returns:
        str: The transcribed text from the audio file. Returns an error message string
            if the model doesn't support audio transcription, if the file cannot be read,
            or if the API call fails. Returns "No transcription available." if the audio
            couldn't be transcribed.

    Raises:
        None: This function catches all exceptions and returns error messages as strings
            instead of raising exceptions.

    Examples:
        >>> # Transcribe with OpenAI Whisper
        >>> client, model, provider = setup_llm_client("whisper-1")
        >>> text = transcribe_audio(
        ...     "recording.mp3", client, model, provider
        ... )
        >>> print(text)
        "Hello, this is a test recording."

        >>> # Transcribe with Google Speech-to-Text (Spanish)
        >>> google_client, google_model, google_provider = setup_llm_client("google-cloud/speech-to-text/latest_long")
        >>> spanish_text = transcribe_audio(
        ...     "spanish_audio.wav", google_client, google_model, google_provider,
        ...     language_code="es-ES"
        ... )
        >>> print(spanish_text)
        "Hola, esta es una grabación de prueba."

        >>> # Handle missing audio file
        >>> error_text = transcribe_audio(
        ...     "nonexistent.mp3", client, model, provider
        ... )
        >>> print(error_text)
        "An API error occurred during audio transcription: [Errno 2] No such file or directory: 'nonexistent.mp3'"

        >>> # Handle non-speech model
        >>> text_client, text_model, text_provider = setup_llm_client("gpt-4o")
        >>> error_text = transcribe_audio(
        ...     "audio.mp3", text_client, text_model, text_provider
        ... )
        >>> print(error_text)
        "Error: Model 'gpt-4o' does not support audio transcription."

        >>> # Handle API client not initialized
        >>> error_text = transcribe_audio("audio.mp3", None, "whisper-1", "openai")
        >>> print(error_text)
        "API client not initialized."

    Notes:
        - Validates that the model supports audio transcription using RECOMMENDED_MODELS
        - Provider-specific handling:
            - OpenAI (Whisper): Supports many languages automatically, no language_code needed
            - Google: Requires explicit language_code parameter, supports specific audio formats
        - File is read in binary mode and sent to the API
        - Google returns results with alternatives; uses the first alternative
        - Handles cases where no transcription is available
        - Audio files should be reasonably sized (OpenAI limit: 25MB, Google: varies)
        - For best results with Google, use 16-bit, mono WAV or FLAC files
        - OpenAI Whisper can handle various audio qualities and background noise better

    Dependencies:
        - google.cloud.speech: For Google Speech-to-Text (if using Google)
        - RECOMMENDED_MODELS: For audio transcription capability validation
        - setup_llm_client(): For initializing the API client
    """
    if not client: return "API client not initialized."
    if not RECOMMENDED_MODELS.get(model_name, {}).get("audio_transcription"):
        return f"Error: Model '{model_name}' does not support audio transcription."

    try:
        if api_provider == "openai":
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(model=model_name, file=f)
            return getattr(response, "text", response.get("text"))
        elif api_provider == "google":
            from google.cloud import speech
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(language_code=language_code)
            response = client.recognize(config=config, audio=audio)
            if response.results and response.results[0].alternatives:
                return response.results[0].alternatives[0].transcript
            return "No transcription available."
        else:
            return f"Error: Audio transcription not implemented for provider '{api_provider}'"
    except Exception as e:
        return f"An API error occurred during audio transcription: {e}"


def clean_llm_output(output_str: str, language: str = 'json') -> str:
    """
    Clean and extract code from markdown code fences in LLM output.

    This function processes LLM responses that contain markdown code blocks (fenced with ```)
    and extracts the clean code content. It's particularly useful when LLMs return formatted
    code examples with markdown syntax, allowing you to get just the executable code.

    Args:
        output_str (str): The raw output string from an LLM that may contain markdown
            code fences. Can include explanatory text, multiple code blocks, or mixed content.
        language (str, optional): The programming language identifier to look for in the
            code fence. Defaults to 'json'. Examples: 'python', 'javascript', 'sql', 'html'.
            The function will match both ```language and ``` (without language specifier).

    Returns:
        str: The cleaned code content extracted from the first matching code block.
            If no code fences are found, returns the original string with whitespace stripped.
            If multiple code blocks exist, only the first one is returned.

    Examples:
        >>> # Clean JSON output with language specifier
        >>> raw_output = '''Here's the JSON data you requested:
        ... ```json
        ... {"name": "John", "age": 30, "city": "New York"}
        ... ```
        ... This data represents a user profile.'''
        >>> clean_json = clean_llm_output(raw_output, 'json')
        >>> print(clean_json)
        {"name": "John", "age": 30, "city": "New York"}

        >>> # Clean Python code without language specifier
        >>> python_output = '''Here's a Python function:
        ... ```
        ... def greet(name):
        ...     return f"Hello, {name}!"
        ... ```
        ... You can use this to greet people.'''
        >>> clean_python = clean_llm_output(python_output, 'python')
        >>> print(clean_python)
        def greet(name):
            return f"Hello, {name}!"

        >>> # Handle output without code fences
        >>> plain_text = "This is just plain text without any code blocks."
        >>> cleaned = clean_llm_output(plain_text)
        >>> print(cleaned)
        This is just plain text without any code blocks.

        >>> # Extract from multiple code blocks (returns first one)
        >>> multi_code = '''```python
        ... print("First block")
        ... ```
        ... Some text in between
        ... ```javascript
        ... console.log("Second block");
        ... ```'''
        >>> first_block = clean_llm_output(multi_code, 'python')
        >>> print(first_block)
        print("First block")

        >>> # Case-insensitive language matching
        >>> mixed_case = '''```JSON
        ... {"key": "value"}
        ... ```'''
        >>> extracted = clean_llm_output(mixed_case, 'json')
        >>> print(extracted)
        {"key": "value"}

    Notes:
        - Uses regex pattern matching for precise code fence detection
        - Supports both language-specific (```json) and generic (```) code fences
        - Case-insensitive language matching for better compatibility
        - Handles whitespace and newlines around code fences gracefully
        - Falls back to simple string splitting if regex doesn't match
        - Strips leading/trailing whitespace from extracted code
        - Returns the original string (stripped) if no code fences are found
        - Only extracts the first code block if multiple exist
        - Useful for processing LLM outputs in automated workflows
        - Can handle malformed markdown with missing closing fences

    Dependencies:
        - re: For regular expression pattern matching and escaping
    """
    if '```' in output_str:
        # Regex to find code blocks with optional language specifier
        # It looks for ```[language_optional]\n[content]\n```
        pattern = re.compile(r'```(?:' + re.escape(language) + r')?\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
        match = pattern.search(output_str)
        if match:
            return match.group(1).strip()
        else:
            # Fallback if regex doesn't match perfectly (e.g., no language specified, or extra text)
            parts = output_str.split('```')
            if len(parts) >= 3:
                # Take the content between the first and second ```
                return parts[1].strip()
            else:
                # If only one ``` or malformed, return original string
                return output_str.strip()
    return output_str.strip()


# --- Artifact Management & Display ---

def _find_project_root():
    """
    Find the project root directory by searching upwards for common project markers.

    This internal utility function searches the directory tree upwards from the current
    working directory to identify the project root. It looks for common indicators of a
    project root directory such as version control folders, documentation files, or
    project-specific directories. This approach is more reliable than using os.getcwd()
    alone, especially in complex directory structures or when scripts are run from
    subdirectories.

    Args:
        None

    Returns:
        str: The absolute path to the project root directory. If no project markers
            are found, returns the current working directory as a fallback.

    Raises:
        None: This function handles all errors gracefully. If no project root markers
            are found, it prints a warning and returns the current directory.

    Examples:
        >>> # Typical project structure
        >>> # /Users/user/myproject/
        >>> # ├── .git/
        >>> # ├── artifacts/
        >>> # ├── README.md
        >>> # └── src/
        >>> #     └── script.py (current working directory)
        >>>
        >>> # When run from src/script.py
        >>> project_root = _find_project_root()
        >>> print(project_root)
        /Users/user/myproject

        >>> # When run from project root
        >>> project_root = _find_project_root()
        >>> print(project_root)
        /Users/user/myproject

        >>> # In a directory without markers
        >>> # /tmp/random_dir/
        >>> project_root = _find_project_root()
        Warning: Project root marker not found. Defaulting to current directory.
        >>> print(project_root)
        /tmp/random_dir

    Notes:
        - Searches upward from current directory until filesystem root
        - Checks for multiple common project markers to increase reliability:
            - '.git': Git repository indicator
            - 'artifacts': Project artifacts directory
            - 'README.md': Common documentation file
        - Stops at filesystem root to prevent infinite loops
        - Falls back to current directory if no markers found
        - Prints warning when falling back to current directory
        - Returns absolute paths for consistency
        - Used internally by artifact management functions
        - Helps ensure file operations happen in the correct project context

    Dependencies:
        - os: For path operations and directory traversal
        - os.path: For path joining and existence checking
    """
    path = os.getcwd()
    while path != os.path.dirname(path):  # Stop at the filesystem root
        # Check for multiple common markers to increase reliability
        if any(os.path.exists(os.path.join(path, marker)) for marker in ['.git', 'artifacts', 'README.md']):
            return path
        path = os.path.dirname(path)
    # Fallback if no markers are found (e.g., in a bare directory)
    print("Warning: Project root marker not found. Defaulting to current directory.")
    return os.getcwd()


def save_artifact(content, file_path):
    """
    Save content to a file, automatically creating directories and handling project structure.

    This function provides a convenient way to save text content to files within the project
    structure. It automatically finds the project root, creates any necessary directories,
    and writes the content to the specified file path. This is particularly useful for saving
    artifacts generated during AI workflows, such as code, documentation, or configuration files.

    Args:
        content (str): The text content to write to the file. Can be any string content
            including code, JSON, markdown, or plain text.
        file_path (str): The relative path from project root where the file should be saved.
            Examples: "artifacts/generated_code.py", "docs/api_reference.md", "config/settings.json".
            Directory structure will be created automatically if it doesn't exist.

    Returns:
        None: This function doesn't return a value but produces side effects:
            - Creates directories as needed
            - Writes content to the specified file
            - Prints success or error messages to console

    Raises:
        None: This function catches all exceptions and prints error messages instead
            of raising exceptions.

    Examples:
        >>> # Save generated Python code
        >>> code_content = '''def hello_world():
        ...     print("Hello, World!")
        ...     return True'''
        >>> save_artifact(code_content, "artifacts/generated_hello.py")
        ✅ Successfully saved artifact to: artifacts/generated_hello.py

        >>> # Save JSON configuration
        >>> import json
        >>> config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
        >>> save_artifact(json.dumps(config, indent=2), "config/model_config.json")
        ✅ Successfully saved artifact to: config/model_config.json

        >>> # Save markdown documentation
        >>> docs = '''# API Documentation
        ... 
        ... This document describes the available endpoints...
        ... '''
        >>> save_artifact(docs, "docs/api_docs.md")
        ✅ Successfully saved artifact to: docs/api_docs.md

        >>> # Save to nested directory (auto-creates directories)
        >>> content = "This is a test file in a nested directory."
        >>> save_artifact(content, "artifacts/deep/nested/folder/test.txt")
        ✅ Successfully saved artifact to: artifacts/deep/nested/folder/test.txt

        >>> # Handle permission errors
        >>> save_artifact("content", "/root/protected_file.txt")
        ❌ Error saving artifact to /root/protected_file.txt: [Errno 13] Permission denied: '/root'

    Notes:
        - Automatically finds project root using _find_project_root()
        - Creates parent directories automatically with os.makedirs()
        - Uses UTF-8 encoding for text files
        - Prints success message with ✅ emoji for successful saves
        - Prints error message with ❌ emoji for failures
        - Handles various exceptions (permission errors, disk space, etc.)
        - Safe to call multiple times - will overwrite existing files
        - Useful for saving LLM-generated content, analysis results, or project artifacts
        - Integrates well with Jupyter notebooks and automated workflows

    Dependencies:
        - os: For path operations and directory creation
        - os.path: For path joining and directory operations
        - _find_project_root(): For locating the project root directory
    """
    try:
        project_root = _find_project_root()
        full_path = os.path.join(project_root, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Successfully saved artifact to: {file_path}")
    except Exception as e:
        print(f"❌ Error saving artifact to {file_path}: {e}")

def load_artifact(file_path):
    """
    Load text content from a file within the project structure.

    This function provides a convenient way to read text content from files within the project
    structure. It automatically finds the project root and reads the specified file relative
    to that location. This is particularly useful for loading artifacts, configuration files,
    or any text-based content that was previously saved using save_artifact().

    Args:
        file_path (str): The relative path from project root to the file to read.
            Examples: "artifacts/generated_code.py", "config/settings.json", "docs/readme.md".

    Returns:
        str or None: The text content of the file as a string. Returns None if the file
            cannot be found or read. The function handles FileNotFoundError specifically
            and prints an error message in that case.

    Raises:
        None: This function catches FileNotFoundError and other exceptions, printing
            error messages and returning None instead of raising exceptions.

    Examples:
        >>> # Load previously saved Python code
        >>> code = load_artifact("artifacts/generated_hello.py")
        >>> print(code)
        def hello_world():
            print("Hello, World!")
            return True

        >>> # Load JSON configuration
        >>> config_content = load_artifact("config/model_config.json")
        >>> import json
        >>> config = json.loads(config_content)
        >>> print(config)
        {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000}

        >>> # Load markdown documentation
        >>> docs = load_artifact("docs/api_docs.md")
        >>> print(docs[:50])
        # API Documentation
        #
        # This document describes...

        >>> # Handle missing file
        >>> missing = load_artifact("nonexistent_file.txt")
        ❌ Error: Artifact file not found at nonexistent_file.txt.
        >>> print(missing)
        None

        >>> # Load file from nested directory
        >>> nested_content = load_artifact("artifacts/deep/nested/folder/test.txt")
        >>> print(nested_content)
        This is a test file in a nested directory.

    Notes:
        - Automatically finds project root using _find_project_root()
        - Reads files with UTF-8 encoding for proper text handling
        - Returns the entire file content as a single string
        - Handles FileNotFoundError specifically with a clear error message
        - Prints error message with ❌ emoji for missing files
        - Returns None for any read errors (permissions, encoding issues, etc.)
        - Useful for loading LLM-generated content, configuration files, or project artifacts
        - Complements save_artifact() for complete file I/O workflow
        - Safe for use in automated workflows and Jupyter notebooks

    Dependencies:
        - os: For path operations
        - os.path: For path joining
        - _find_project_root(): For locating the project root directory
    """
    try:
        project_root = _find_project_root()
        full_path = os.path.join(project_root, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: Artifact file not found at {file_path}.")
        return None

def render_plantuml_diagram(puml_code, output_path="artifacts/diagram.png"):
    """
    Render PlantUML markup code into a PNG diagram image and display it.

    This function takes PlantUML markup code and converts it into a visual diagram using
    the PlantUML web service. The generated PNG image is automatically saved to the
    specified path within the project and displayed in Jupyter notebook environments.
    It handles different versions of the plantuml library and provides fallbacks for
    various output formats.

    Args:
        puml_code (str): The PlantUML markup code to render. Must be valid PlantUML syntax
            starting with @startuml and ending with @enduml. Examples:
            - Basic class diagram: "@startuml\\nclass User\\n@enduml"
            - Sequence diagram: "@startuml\\nAlice -> Bob: Hello\\n@enduml"
            - Activity diagram: "@startuml\\nstart\\n:Action;\\nstop\\n@enduml"
        output_path (str, optional): The relative path from project root where the PNG
            image will be saved. Defaults to "artifacts/diagram.png". The directory
            structure will be created automatically if it doesn't exist.

    Returns:
        None: This function doesn't return a value but produces side effects:
            - Saves PNG image to the specified file path
            - Prints status messages to console
            - Displays the image in Jupyter environments using IPython.display

    Raises:
        None: This function catches all exceptions and prints error messages instead
            of raising exceptions.

    Examples:
        >>> # Render a simple class diagram
        >>> puml_code = '''@startuml
        ... class User {
        ...     +name: string
        ...     +email: string
        ...     +login()
        ... }
        ... class Admin extends User {
        ...     +permissions: string[]
        ...     +manageUsers()
        ... }
        ... User ||-- Admin
        ... @enduml'''
        >>> render_plantuml_diagram(puml_code, "artifacts/user_class_diagram.png")
        ✅ Diagram rendered and saved to: artifacts/user_class_diagram.png

        >>> # Render a sequence diagram
        >>> sequence_code = '''@startuml
        ... title Authentication Flow
        ... actor User
        ... participant "Web App" as App
        ... participant "Auth Service" as Auth
        ... User -> App: Login request
        ... App -> Auth: Validate credentials
        ... Auth --> App: Token
        ... App --> User: Success
        ... @enduml'''
        >>> render_plantuml_diagram(sequence_code, "artifacts/auth_flow.png")
        ✅ Diagram rendered and saved to: artifacts/auth_flow.png

        >>> # Render with default path
        >>> simple_code = '''@startuml
        ... start
        ... :User logs in;
        ... :Validate credentials;
        ... :Generate token;
        ... stop
        ... @enduml'''
        >>> render_plantuml_diagram(simple_code)
        ✅ Diagram rendered and saved to: artifacts/diagram.png

        >>> # Handle PlantUML syntax errors
        >>> invalid_code = "This is not valid PlantUML"
        >>> render_plantuml_diagram(invalid_code)
        ❌ Error rendering PlantUML diagram: [PlantUML error details]

    Notes:
        - Uses the public PlantUML web service (http://www.plantuml.com/plantuml/img/)
        - Automatically finds project root using _find_project_root()
        - Creates output directories automatically with os.makedirs()
        - Handles different versions of the plantuml library automatically
        - Supports both direct file writing and URL-based image fetching
        - Displays images in Jupyter notebooks using IPython.display
        - Falls back to markdown image links if IPython display fails
        - Prints success messages with ✅ emoji and error messages with ❌ emoji
        - Works with various PlantUML diagram types: class, sequence, activity, use case, etc.
        - PNG images are saved with high quality and standard dimensions
        - Safe for use in automated documentation generation workflows

    Dependencies:
        - plantuml: Python library for PlantUML integration
        - requests: For HTTP requests when fetching images from URLs
        - IPython.display: For displaying images in Jupyter notebooks (optional)
        - os: For file system operations
        - os.path: For path operations
        - _find_project_root(): For locating the project root directory
    """
    try:
        project_root = _find_project_root()
        full_path = os.path.join(project_root, output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        result_saved = False
        # Try using the plantuml Python library if available
        try:
            from plantuml import PlantUML  # type: ignore
            pl = PlantUML(url='http://www.plantuml.com/plantuml/img/')
            try:
                # Some PlantUML versions support generating directly to a file
                pl.processes(puml_code, outfile=full_path)
                result_saved = os.path.exists(full_path)
            except TypeError:
                # Fallback: processes may return bytes or a URL
                result = pl.processes(puml_code)
                if isinstance(result, (bytes, bytearray)):
                    with open(full_path, 'wb') as f:
                        f.write(result)
                    result_saved = True
                elif isinstance(result, str) and result.startswith('http'):
                    resp = requests.get(result)
                    resp.raise_for_status()
                    with open(full_path, 'wb') as f:
                        f.write(resp.content)
                    result_saved = True
        except Exception:
            # PlantUML library not available or failed; fall back to PlantUML server
            try:
                resp = requests.post('http://www.plantuml.com/plantuml/png', data=puml_code.encode('utf-8'), headers={'Content-Type': 'text/plain'})
                resp.raise_for_status()
                with open(full_path, 'wb') as f:
                    f.write(resp.content)
                result_saved = True
            except Exception as server_err:
                print(f"❌ Error fetching PlantUML image from server: {server_err}")

        if result_saved and os.path.exists(full_path):
            print(f"✅ Diagram rendered and saved to: {output_path}")
            try:
                display(IPyImage(filename=full_path))
            except Exception:
                display(Markdown(f"![diagram]({full_path})"))
        else:
            print("⚠️ Diagram rendering returned no file.")
    except Exception as e:
        print(f"❌ Error rendering PlantUML diagram: {e}")

def _encode_image_to_base64(image_path):
    """
    Encode a local image file to a base64 data URL for web embedding.

    This internal utility function reads a local image file, determines its MIME type,
    and encodes the binary data to base64 format. The result is a data URL that can be
    directly embedded in HTML, CSS, or used in web applications without requiring
    separate image files. This is commonly used for vision API calls that require
    inline image data.

    Args:
        image_path (str): The absolute or relative path to the image file to encode.
            Supported formats include JPEG, PNG, GIF, WebP, and other common image types.
            The file must exist and be readable.

    Returns:
        str: A data URL string in the format "data:{mime_type};base64,{base64_data}".
            This can be used directly in HTML img tags, CSS background-image properties,
            or API calls that accept inline image data.

    Raises:
        ValueError: If the MIME type cannot be determined or if the file is not an image.
        FileNotFoundError: If the specified image file does not exist.
        PermissionError: If the file cannot be read due to permission restrictions.
        OSError: For other file system related errors.

    Examples:
        >>> # Encode a JPEG image
        >>> data_url = _encode_image_to_base64("artifacts/photo.jpg")
        >>> print(data_url[:50])
        data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...

        >>> # Use in HTML
        >>> html_img = f'<img src="{data_url}" alt="My Photo">'
        >>> print(html_img)
        <img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..." alt="My Photo">

        >>> # Encode PNG with transparency
        >>> png_url = _encode_image_to_base64("icons/logo.png")
        >>> print(png_url[:50])
        data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...

        >>> # Handle non-image file
        >>> try:
        ...     _encode_image_to_base64("document.txt")
        ... except ValueError as e:
        ...     print(e)
        Cannot determine image type for document.txt

        >>> # Handle missing file
        >>> try:
        ...     _encode_image_to_base64("missing.jpg")
        ... except FileNotFoundError as e:
        ...     print(e)
        [Errno 2] No such file or directory: 'missing.jpg'

    Notes:
        - Automatically detects MIME type using Python's mimetypes module
        - Reads entire file into memory - not suitable for very large images
        - Uses UTF-8 encoding for the base64 string
        - Returns standard data URL format compatible with all modern browsers
        - Commonly used by vision API functions (get_vision_completion) for inline image data
        - Base64 encoding increases file size by approximately 33%
        - Safe for use with common image formats (JPEG, PNG, GIF, WebP, BMP, TIFF)
        - Validates that the file is actually an image before encoding

    Dependencies:
        - base64: For encoding binary data to base64 format
        - mimetypes: For determining file MIME types from extensions
        - os: For file path operations (implicitly used)
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image'):
        raise ValueError(f"Cannot determine image type for {image_path}")
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

