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
import subprocess
import time # For loading indicator

# --- Dynamic Library Installation ---
try:
    from dotenv import load_dotenv
    from IPython.display import display, Markdown, Code, Image as IPyImage
    from plantuml import PlantUML
except ImportError:
    # Provide safe fallbacks so the module can be imported even when optional deps are missing.
    print("Warning: Optional core dependencies not found. Some features will be degraded.")
    print("To enable full functionality run: pip install python-dotenv ipython plantuml")

    # noop load_dotenv fallback
    def load_dotenv(*args, **kwargs):
        print("Warning: python-dotenv not installed; .env will not be loaded.")

    # minimal IPython.display fallbacks
    def display(*args, **kwargs):
        # no-op in non-notebook environments
        return None

    def Markdown(text):
        return text

    def Code(text):
        return text

    class IPyImage:
        def __init__(self, *args, **kwargs):
            # placeholder for notebook image object
            pass

    # PlantUML fallback
    class PlantUML:
        def __init__(self, url=None):
            print("Warning: plantuml not installed; rendering disabled.")
        def processes(self, *args, **kwargs):
            print("PlantUML rendering skipped (plantuml not installed).")


# --- Model & Provider Configuration ---

RECOMMENDED_MODELS = {
    # =========================
    # OpenAI — Text + Vision
    # =========================
    "gpt-5-nano-2025-08-07": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 400_000, "output_tokens": 128_000
    },
    "gpt-5-mini-2025-08-07": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 400_000, "output_tokens": 128_000
    },
    "gpt-5-2025-08-07": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 400_000, "output_tokens": 128_000
    },
    "gpt-4o": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 128_000, "output_tokens": 16_384
    },
    "gpt-4o-mini": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 128_000, "output_tokens": 16_384
    },
    "gpt-4.1": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_000_000, "output_tokens": 32_000
    },
    "gpt-4.1-mini": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_000_000, "output_tokens": 32_000
    },
    "gpt-4.1-nano": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_000_000, "output_tokens": 32_000
    },
    "gpt-4.5": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 128_000, "output_tokens": 16_384
    },
    "o3": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 200_000, "output_tokens": 100_000
    },
    "o4-mini": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 200_000, "output_tokens": 100_000
    },
    "codex-mini-latest": {
        "provider": "openai", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 200_000, "output_tokens": 100_000
    },

    # =========================
    # OpenAI — Image / Audio
    # =========================
    "gpt-image-1": {
        "provider": "openai", "vision": True, "image_generation": True, "audio_transcription": False,
        "context_window_tokens": None, "output_tokens": None
    },
    "dall-e-3": {
        "provider": "openai", "vision": False, "image_generation": True, "audio_transcription": False,
        "context_window_tokens": None, "output_tokens": None
    },
    "whisper-1": {
        "provider": "openai", "vision": False, "image_generation": False, "audio_transcription": True,
        "context_window_tokens": None, "output_tokens": None
    },

    # =========================
    # Anthropic — Claude
    # =========================
    "claude-opus-4-1-20250805": {
        "provider": "anthropic", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 200_000, "output_tokens": 100_000
    },
    "claude-opus-4-20250514": {
        "provider": "anthropic", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 200_000, "output_tokens": 100_000
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_000_000, "output_tokens": 100_000
    },

    # ==========================================
    # Google — Gemini / Imagen / Speech-to-Text
    # ==========================================
    "gemini-2.5-pro": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 65_536
    },
    "gemini-2.5-flash": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 65_536
    },
    "gemini-2.5-flash-lite": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 65_536
    },
    "gemini-live-2.5-flash-preview": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 8_192
    },
    "gemini-2.5-flash-image-preview": {
        "provider": "google", "vision": True, "image_generation": True, "audio_transcription": False,
        "context_window_tokens": 32_768, "output_tokens": 32_768
    },
    "gemini-2.0-flash": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 8_192
    },
    "gemini-2.0-flash-lite": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 8_192
    },
    "gemini-2.0-flash-live-001": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_048_576, "output_tokens": 8_192
    },
    "gemini-veo-3": {
        "provider": "google", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": None, "output_tokens": None
    },
    "imagen-3.0-generate-002": {
        "provider": "google", "vision": False, "image_generation": True, "audio_transcription": False,
        "context_window_tokens": None, "output_tokens": None
    },
    "imagen-4.0-generate-001": {
        "provider": "google", "vision": False, "image_generation": True, "audio_transcription": False,
        "context_window_tokens": 480, "output_tokens": None
    },
    "google-cloud/speech-to-text/latest_long": {
        "provider": "google", "vision": False, "image_generation": False, "audio_transcription": True,
        "context_window_tokens": None, "output_tokens": None
    },
    "google-cloud/speech-to-text/latest_short": {
        "provider": "google", "vision": False, "image_generation": False, "audio_transcription": True,
        "context_window_tokens": None, "output_tokens": None
    },

    # =========================
    # Hugging Face — OSS
    # =========================
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "provider": "huggingface", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 10_000_000, "output_tokens": 100_000
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "provider": "huggingface", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 1_000_000, "output_tokens": 100_000
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "provider": "huggingface", "vision": False, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 4_096, "output_tokens": 1_024
    },
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5": {
        "provider": "huggingface", "vision": False, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 4_096, "output_tokens": 1_024
    },
    "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3": {
        "provider": "huggingface", "vision": False, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 4_096, "output_tokens": 1_024
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "provider": "huggingface", "vision": False, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 32_768, "output_tokens": 8_192
    },
    "deepseek-ai/DeepSeek-V3": {
        "provider": "huggingface", "vision": False, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 128_000, "output_tokens": 100_000
    },
    "deepseek-ai/DeepSeek-V3-Small": {
        "provider": "huggingface", "vision": False, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 128_000, "output_tokens": 100_000
    },
    "deepseek-ai/DeepSeek-VL2": {
        "provider": "huggingface", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 32_000, "output_tokens": 8_000
    },
    "deepseek-ai/DeepSeek-VL2-Small": {
        "provider": "huggingface", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 32_000, "output_tokens": 8_000
    },
    "deepseek-ai/DeepSeek-VL2-Tiny": {
        "provider": "huggingface", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 32_000, "output_tokens": 8_000
    },
    "deepseek-ai/Janus-Pro-7B": {
        "provider": "huggingface", "vision": True, "image_generation": False, "audio_transcription": False,
        "context_window_tokens": 0, "output_tokens": 0
    },
}

def recommended_models_table(task=None, provider=None, vision=None, image_generation=None,
                             audio_transcription=None, min_context=None, min_output_tokens=None):
    """
    Return a markdown table of recommended models, optionally filtered by attributes.

    Args:
        task (str, optional): High level task to filter models. Accepts values like
            'vision', 'image', 'audio', or 'text'. These set sensible defaults for
            the corresponding capability flags unless explicitly provided.
        provider (str, optional): Filter models by provider name (e.g. ``'openai'``).
        vision (bool, optional): If set, include only models that match vision capability.
        image_generation (bool, optional): If set, include only image generation models.
        audio_transcription (bool, optional): If set, include only models supporting
            audio transcription.
        min_context (int, optional): Minimum context window size required.
        min_output_tokens (int, optional): Minimum max output tokens required.

    Returns:
        str: Markdown formatted table.
    """
    # Interpret task shortcuts
    if task:
        t = task.lower()
        if t in {"vision", "multimodal", "vl"} and vision is None:
            vision = True
        elif t in {"image", "image_generation", "image-generation"} and image_generation is None:
            image_generation = True
        elif t in {"audio", "speech", "audio_transcription", "stt"} and audio_transcription is None:
            audio_transcription = True
        elif t == "text":
            vision = False if vision is None else vision
            image_generation = False if image_generation is None else image_generation
            audio_transcription = False if audio_transcription is None else audio_transcription

    rows = []
    for model_name in sorted(RECOMMENDED_MODELS.keys()):
        cfg = RECOMMENDED_MODELS[model_name]
        model_provider = (cfg.get("provider") or "").lower()
        model_vision = cfg.get("vision", False)
        model_image = cfg.get("image_generation", False)
        model_audio = cfg.get("audio_transcription", False)

        # Prefer canonical integer fields used in RECOMMENDED_MODELS
        context = cfg.get("context_window_tokens")
        if context is None:
            # Backwards-compat: allow older key name
            context = cfg.get("context_window")

        max_tokens = cfg.get("output_tokens")
        if max_tokens is None:
            max_tokens = cfg.get("max_output_tokens")

        # Normalize provider filter to be case-insensitive
        if provider and model_provider != provider.lower():
            continue
        if vision is not None and bool(model_vision) != bool(vision):
            continue
        if image_generation is not None and bool(model_image) != bool(image_generation):
            continue
        if audio_transcription is not None and bool(model_audio) != bool(audio_transcription):
            continue
        if min_context and (context is None or (isinstance(context, int) and context < min_context)):
            continue
        if min_output_tokens and (max_tokens is None or (isinstance(max_tokens, int) and max_tokens < min_output_tokens)):
            continue

        def _fmt_num(x):
            if x is None:
                return "-"
            try:
                # format large ints with commas
                return f"{int(x):,}"
            except Exception:
                return str(x)

        rows.append(
            f"| {model_name} | {model_provider or '-'} | {'✅' if model_vision else '❌'} | "
            f"{'✅' if model_image else '❌'} | {'✅' if model_audio else '❌'} | "
            f"{_fmt_num(context)} | {_fmt_num(max_tokens)} |"
        )

    if not rows:
        return "No models match the specified criteria."

    header = (
        "| Model | Provider | Vision | Image Gen | Audio Transcription | Context Window | Max Output Tokens |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    table = header + "\n".join(rows)
    display(Markdown(table))
    return table

# --- Environment and API Client Setup ---

def load_environment():
    """
    Loads environment variables from a .env file in the project root.
    
    This function searches upward from the current working directory to find the
    project root by looking for specific markers (.env file or .git directory).
    Once found, it loads all environment variables defined in the .env file,
    making them available to the application through os.getenv().
    
    Args:
        None
    
    Returns:
        None: This function doesn't return a value but has the side effect of
            loading environment variables into the process environment.
    
    Raises:
        None: This function handles all errors gracefully and prints warnings
            instead of raising exceptions.
    
    Notes:
        - Searches upward from current directory until it finds .env or .git
        - Falls back to current directory if no markers are found
        - Uses python-dotenv library to parse and load the .env file
        - Prints a warning if .env file is not found
        - Environment variables loaded are accessible via os.getenv()
    
    Example:
        >>> load_environment()
        >>> api_key = os.getenv('OPENAI_API_KEY')
        
        # Typical .env file content:
        # OPENAI_API_KEY=sk-...
        # ANTHROPIC_API_KEY=sk-ant-...
        # GOOGLE_API_KEY=AIza...
    
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
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print("Warning: .env file not found. API keys may not be loaded.")


def setup_llm_client(model_name="gpt-4o"):
    """
    Configures and returns an LLM client based on the specified model name.
    
    This function initializes the appropriate API client for the specified model,
    handling authentication and configuration for multiple providers including
    OpenAI, Anthropic, Hugging Face, and Google (Gemini/Imagen/Speech-to-Text).
    It automatically loads environment variables and validates API keys.
    
    Args:
        model_name (str, optional): The identifier of the model to use. Must be
            a key in the RECOMMENDED_MODELS dictionary. Defaults to "gpt-4o".
            Examples: "gpt-4o", "claude-3-opus-20240229", "gemini-2.5-pro"
    
    Returns:
        tuple: A 3-element tuple containing:
            - client: The initialized API client object (varies by provider)
                - OpenAI: OpenAI client instance
                - Anthropic: Anthropic client instance
                - Hugging Face: InferenceClient instance
                - Google: GenerativeModel, genai module, or SpeechClient
            - model_name (str): The model name (echoed back)
            - api_provider (str): The provider name ("openai", "anthropic", etc.)
            
            Returns (None, None, None) if initialization fails.
    
    Raises:
        None: This function handles all errors gracefully and prints error messages
            instead of raising exceptions.
    
    Notes:
        - Automatically calls load_environment() to load .env file
        - Validates that the model exists in RECOMMENDED_MODELS
        - Checks for required API keys in environment variables
        - Handles ImportError if provider libraries aren't installed
        - Special handling for Google models based on their capabilities:
            - Audio transcription models use google.cloud.speech
            - Image generation models return the genai module
            - Text/vision models return a GenerativeModel instance
        - Prints success/error messages to console
    
    Example:
        >>> # Initialize OpenAI client
        >>> client, model, provider = setup_llm_client("gpt-4o")
        ✅ LLM Client configured: Using 'openai' with model 'gpt-4o'
        
        >>> # Initialize Anthropic client
        >>> client, model, provider = setup_llm_client("claude-3-opus-20240229")
        ✅ LLM Client configured: Using 'anthropic' with model 'claude-3-opus-20240229'
        
        >>> # Handle missing API key
        >>> client, model, provider = setup_llm_client("gpt-4o")
        ERROR: OPENAI_API_KEY not found in .env file.
    
    Dependencies:
        - Provider-specific libraries (installed as needed):
            - openai: For OpenAI models
            - anthropic: For Anthropic models
            - huggingface_hub: For Hugging Face models
            - google.generativeai: For Google Gemini/Imagen
            - google.cloud.speech: For Google Speech-to-Text
        - RECOMMENDED_MODELS: Global dictionary with model configurations
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
        elif api_provider == "gemini" or api_provider == "google": # Google for image generation or STT
            if config.get("audio_transcription"):
                from google.cloud import speech
                client = speech.SpeechClient()
            else:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY") # Use GOOGLE_API_KEY for both Gemini text and Imagen
                if not api_key: raise ValueError("GOOGLE_API_KEY not found in .env file.")
                genai.configure(api_key=api_key)
                # For image generation, we'll pass the genai module itself
                # and instantiate the model within the generation function.
                if config.get("image_generation"):
                    client = genai
                else:
                    # For other Gemini models (text, vision), we instantiate a GenerativeModel.
                    client = genai.GenerativeModel(model_name)
    except ImportError:
        print(f"ERROR: The required library for '{api_provider}' is not installed.")
        return None, None, None
    except ValueError as e:
        print(f"ERROR: {e}")
        return None, None, None
    print(f"✅ LLM Client configured: Using '{api_provider}' with model '{model_name}'")
    return client, model_name, api_provider

# --- Core Interaction Functions ---

def get_completion(prompt, client, model_name, api_provider, temperature=0.7):
    """
    Sends a text-only prompt to the LLM and returns the completion.
    
    This function provides a unified interface for getting text completions from
    various LLM providers. It handles provider-specific API differences and
    error cases, including special handling for newer OpenAI models that may
    use different endpoints or not support temperature parameters.
    
    Args:
        prompt (str): The text prompt to send to the model. This is the user's
            input or question that the model should respond to.
        client: The initialized API client object from setup_llm_client().
            Type varies by provider (OpenAI, Anthropic, InferenceClient, etc.)
        model_name (str): The identifier of the model to use for completion.
        api_provider (str): The provider name ("openai", "anthropic", "huggingface",
            "gemini", or "google").
        temperature (float, optional): Controls randomness in the output. Higher
            values (e.g., 1.0) make output more random, lower values (e.g., 0.1)
            make it more deterministic. Defaults to 0.7. Range typically 0.0-2.0.
    
    Returns:
        str: The generated text completion from the model. Returns an error
            message string if the API call fails.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Handles different API structures for each provider
        - OpenAI: Tries chat completions first, falls back to responses endpoint
        - Anthropic: Uses messages API with max_tokens=4096
        - Hugging Face: Uses chat_completion with minimum temperature of 0.1
        - Google/Gemini: Uses generate_content method
        - Special error handling for OpenAI models that don't support temperature
        - Returns descriptive error messages if API calls fail
    
    Example:
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> response = get_completion(
        ...     "What is the capital of France?",
        ...     client, model, provider, temperature=0.5
        ... )
        >>> print(response)
        "The capital of France is Paris."
        
        >>> # Handle API errors gracefully
        >>> response = get_completion("Hello", None, "gpt-4o", "openai")
        >>> print(response)
        "API client not initialized."
    
    Dependencies:
        - Provider-specific client libraries
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
    Sends an image and a text prompt to a vision-capable LLM and returns the completion.
    
    This function enables multimodal AI interactions by processing both text and image
    inputs together. It handles the different image processing requirements for each
    provider, including URL-based and base64-encoded image formats. It can accept
    either a public URL to an image or a local file path.
    
    Args:
        prompt (str): The text prompt or question about the image. This provides
            context or specific instructions for analyzing the image.
        image_path_or_url (str): URL of the image to analyze or a local file path.
            If a URL, it must be publicly accessible.
        client: The initialized API client object from setup_llm_client().
            Type varies by provider.
        model_name (str): The identifier of the vision-capable model to use.
        api_provider (str): The provider name ("openai", "anthropic", "huggingface",
            "gemini", or "google").
    
    Returns:
        str: The model's response analyzing the image based on the prompt.
            Returns an error message string if the model doesn't support vision
            or if the API call fails.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Validates that the model supports vision using RECOMMENDED_MODELS
        - Handles both URLs and local file paths for images.
        - Different providers require different image formats:
            - OpenAI: Can use image URLs directly or base64-encoded data URLs.
            - Anthropic: Requires base64-encoded image data with MIME type.
            - Google/Gemini: Requires PIL Image object.
            - Hugging Face: Requires PIL Image object.
        - Automatically downloads images from URLs or reads from disk and converts as needed.
        - Sets max_tokens to 4096 for providers that support it.
        - Handles HTTP errors when fetching images and file I/O errors.
    
    Example:
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> # Using a URL
        >>> response_url = get_vision_completion(
        ...     "What objects do you see in this image?",
        ...     "https://example.com/image.jpg",
        ...     client, model, provider
        ... )
        >>> # Using a local file
        >>> response_local = get_vision_completion(
        ...     "Describe this local image.",
        ...     "artifacts/screens/my_image.png",
        ...     client, model, provider
        ... )
        
    Dependencies:
        - requests: For downloading images from URLs
        - PIL (Pillow): For image processing
        - base64: For encoding images
        - mimetypes: For determining image type from file extension
        - io.BytesIO: For converting image bytes to PIL Images
        - RECOMMENDED_MODELS: For vision capability validation
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
    Generates an image from a text prompt using an image generation LLM.
    
    This function provides a unified interface for text-to-image generation across
    different providers. It handles the API differences between providers and
    returns the generated image as a base64-encoded data URL that can be displayed
    directly in web browsers or Jupyter notebooks.
    
    Args:
        prompt (str): The text description of the image to generate. Should be
            detailed and specific for best results. Example: "A serene mountain
            landscape at sunset with a lake in the foreground".
        client: The initialized API client object from setup_llm_client().
            For Google Imagen, this might be the genai module itself.
        model_name (str): The identifier of the image generation model to use.
            Examples: "dall-e-3", "imagen-3.0-generate-002".
        api_provider (str): The provider name ("openai" or "google").
    
    Returns:
        tuple[str, str]: A tuple containing:
            - file_path (str): The local path to the saved image file.
            - image_url (str): A data URL string in the format "data:image/png;base64,{base64_data}"
              that can be used directly in HTML img tags or displayed in Jupyter.
            Returns (None, None) if an error occurs.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Validates that the model supports image generation using RECOMMENDED_MODELS
        - Displays a loading indicator during generation (can take 10-30 seconds)
        - Tracks and reports generation time
        - Provider-specific handling:
            - OpenAI: Uses images.generate() API, returns base64 directly
            - Google Imagen: Uses REST API with predict endpoint
            - Google Gemini: Uses generate_images() method
        - The returned data URL can be used directly in HTML or markdown
        - Loading indicators are shown in console and Jupyter environments
    
    Example:
        >>> client, model, provider = setup_llm_client("dall-e-3")
        >>> file_path, image_url = get_image_generation_completion(
        ...     "A futuristic city with flying cars and neon lights",
        ...     client, model, provider
        ... )
        Generating image... This may take a moment.
        ⏳ Generating image...
        ✅ Image generated in 15.32 seconds.
        ✅ Image saved to: artifacts/screens/image_1662586800.png
        >>> # Display in Jupyter: display(Image(url=image_url))
        
        >>> # Error handling
        >>> response = get_image_generation_completion(
        ...     "A cat", client, "gpt-4o", "openai"
        ... )
        >>> print(response)
        (None, "Error: Model 'gpt-4o' does not support image generation.")
    
    Dependencies:
        - time: For tracking generation duration
        - json: For handling Google API payloads
        - requests: For Google Imagen REST API calls
        - IPython.display: For showing loading indicators in Jupyter
        - RECOMMENDED_MODELS: For image generation capability validation
    """
    if not client: 
        return None, "API client not initialized."
        
    if not RECOMMENDED_MODELS.get(model_name, {}).get("image_generation"):
        return None, f"Error: Model '{model_name}' does not support image generation."

    # Display a loading indicator
    print("Generating image... This may take a moment.")
    display(Markdown("⏳ Generating image..."))
    start_time = time.time()

    try:
        image_data_base64 = None
        
        if api_provider == "openai":
            response = client.images.generate(
                model=model_name,
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )
            image_data_base64 = response.data[0].b64_json
        elif api_provider == "google":
            try:
                # The client is the genai module itself.
                response = client.generate_content(
                    model_name=model_name,
                    content=prompt
                )
                
                if response and hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.data:
                            img_bytes = part.inline_data.data
                            image_data_base64 = base64.b64encode(img_bytes).decode('utf-8')
                            break # Found the image, exit loop
                
                if not image_data_base64:
                    # Check for text response as a fallback
                    text_response = ""
                    if response and hasattr(response, 'candidates') and response.candidates:
                        for part in response.candidates[0].content.parts:
                            if part.text:
                                text_response = part.text
                                break
                    return None, f"The model '{model_name}' generated a text description instead of image data. Description: {text_response[:200]}..."

            except Exception as model_error:
                return None, f"Google image generation failed: {model_error}"

        if not image_data_base64:
            return None, "Image generation failed or returned no data."

        # Save and display the image
        duration = time.time() - start_time
        print(f"✅ Image generated in {duration:.2f} seconds.")
        
        image_bytes = base64.b64decode(image_data_base64)
        
        # Create a unique filename
        timestamp = int(time.time() * 1000)
        file_path = f"artifacts/screens/image_{timestamp}.png"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        print(f"✅ Image saved to: {file_path}")
        
        # Return the data URL
        return file_path, f"data:image/png;base64,{image_data_base64}"

    except Exception as e:
        return None, f"An API error occurred during image generation: {e}"


def transcribe_audio(audio_path, client, model_name, api_provider, language_code="en-US"):
    """
    Transcribes audio from a file using a speech-to-text model.
    
    This function provides a unified interface for converting speech in audio files
    to text across different providers. It handles various audio formats and
    provider-specific API differences.
    
    Args:
        audio_path (str): Path to the audio file to transcribe. Can be absolute
            or relative. Supported formats vary by provider but typically include
            MP3, WAV, M4A, and other common audio formats.
        client: The initialized API client object from setup_llm_client().
            - OpenAI: OpenAI client instance
            - Google: speech.SpeechClient instance
        model_name (str): The identifier of the speech-to-text model to use.
            Examples: "whisper-1", "google-cloud/speech-to-text/latest_long".
        api_provider (str): The provider name ("openai" or "google").
        language_code (str, optional): The language of the audio in BCP-47 format.
            Defaults to "en-US" (American English). Examples: "es-ES" (Spanish),
            "fr-FR" (French), "ja-JP" (Japanese). Only used by Google Speech-to-Text.
    
    Returns:
        str: The transcribed text from the audio file. Returns an error message
            string if the model doesn't support audio transcription, if the file
            cannot be read, or if the API call fails. Returns "No transcription
            available." if the audio couldn't be transcribed.
    
    Raises:
        None: This function catches all exceptions and returns error messages
            as strings instead of raising exceptions.
    
    Notes:
        - Validates that the model supports audio transcription using RECOMMENDED_MODELS
        - Provider-specific handling:
            - OpenAI (Whisper): Supports many languages automatically
            - Google: Requires explicit language_code parameter
        - File is read in binary mode and sent to the API
        - Google returns results with alternatives; uses the first alternative
        - Handles cases where no transcription is available
    
    Example:
        >>> # OpenAI Whisper transcription
        >>> client, model, provider = setup_llm_client("whisper-1")
        >>> text = transcribe_audio(
        ...     "recording.mp3", client, model, provider
        ... )
        >>> print(text)
        "Hello, this is a test recording."
        
        >>> # Google Speech-to-Text with Spanish audio
        >>> client, model, provider = setup_llm_client("google-cloud/speech-to-text/latest_short")
        >>> text = transcribe_audio(
        ...     "spanish_audio.wav", client, model, provider, language_code="es-ES"
        ... )
        >>> print(text)
        "Hola, esta es una grabación de prueba."
        
        >>> # Error handling
        >>> text = transcribe_audio("audio.mp3", client, "gpt-4o", "openai")
        >>> print(text)
        "Error: Model 'gpt-4o' does not support audio transcription."
    
    Dependencies:
        - google.cloud.speech: For Google Speech-to-Text (if using Google)
        - RECOMMENDED_MODELS: For audio transcription capability validation
    """
    if not client:
        return "API client not initialized."
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
    Cleans markdown code fences from LLM output.
    Supports various languages.
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
    Finds the project root by searching upwards for a known directory marker
    (like '.git' or 'artifacts'). This is more reliable than just using os.getcwd().
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
    """Saves content to a specified file path, creating directories if needed."""
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
    """Loads content from a specified file path."""
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
    Renders PlantUML code into a PNG image and displays it in Jupyter environments.
    
    This function takes PlantUML markup code and converts it into a visual diagram
    using the PlantUML web service. The generated image is saved to the specified
    path within the project and automatically displayed in Jupyter notebooks.
    
    Args:
        puml_code (str): The PlantUML markup code to render. Should be valid PlantUML
            syntax (e.g., "@startuml\\nclass Example\\n@enduml").
        output_path (str, optional): Relative path from project root where the PNG
            image will be saved. Defaults to "artifacts/diagram.png". Directory
            structure will be created automatically if it doesn't exist.
    
    Returns:
        None: This function doesn't return a value but produces side effects:
            - Saves PNG image to the specified file path
            - Prints status messages to console
            - Displays the image in Jupyter environments
    
    Raises:
        Exception: Catches and reports any errors during the rendering process,
            including network errors, file system errors, or PlantUML syntax errors.
    
    Notes:
        - Uses the public PlantUML web service (http://www.plantuml.com/plantuml/img/)
        - Handles different versions of the plantuml library automatically
        - Creates output directories as needed using os.makedirs
        - Falls back gracefully if image display fails in non-Jupyter environments
        - Supports both direct file writing and URL-based image fetching
    
    Example:
        >>> puml_code = '''
        ... @startuml
        ... class User {
        ...     +name: string
        ...     +email: string
        ...     +login()
        ... }
        ... @enduml
        ... '''
        >>> render_plantuml_diagram(puml_code, "diagrams/user_class.png")
        ✅ Diagram rendered and saved to: diagrams/user_class.png
    
    Dependencies:
        - plantuml: Python library for PlantUML integration
        - requests: For HTTP requests when fetching images from URLs
        - PIL (Pillow): Used internally by plantuml library
        - IPython.display: For displaying images in Jupyter notebooks (optional)
    """
    try:
        # FIX: Corrected the PlantUML URL
        pl = PlantUML(url='http://www.plantuml.com/plantuml/img/')
        project_root = _find_project_root()
        
        full_path = os.path.join(project_root, output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # plantuml library versions differ in their `processes` signature.
        # Some accept an `outfile` kwarg, others return the image data or a URL.
        result = None
        try:
            # Preferred: try calling with outfile (some versions support this)
            result = pl.processes(puml_code, outfile=full_path)
        except TypeError:
            # Fallback: call without outfile and handle returned data/result.
            result = pl.processes(puml_code)

        # If the library returned raw bytes, save them to the file.
        if isinstance(result, (bytes, bytearray)):
            with open(full_path, 'wb') as f:
                f.write(result)
        # If the library returned a URL string, fetch it and save the image bytes.
        elif isinstance(result, str) and result.startswith('http'):
            try:
                resp = requests.get(result)
                resp.raise_for_status()
                with open(full_path, 'wb') as f:
                    f.write(resp.content)
            except Exception:
                # If fetching the URL fails, still continue to let callers know result.
                pass

        # At this point, the plantuml lib may have already written the file
        # or we wrote it above. Check for file existence before displaying.
        if os.path.exists(full_path):
            print(f"✅ Diagram rendered and saved to: {output_path}")
            try:
                # IPython Image accepts filename= or url=. Use filename for local file.
                display(IPyImage(filename=full_path))
            except Exception:
                # Best-effort fallback to markdown link if display fails.
                display(Markdown(f"![diagram]({full_path})"))
        else:
            print(f"⚠️ Diagram rendering returned no file. Result: {result}")
    except Exception as e:
        print(f"❌ Error rendering PlantUML diagram: {e}")

def _encode_image_to_base64(image_path):
    """Encodes a local image file to a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image'):
        raise ValueError(f"Cannot determine image type for {image_path}")
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"