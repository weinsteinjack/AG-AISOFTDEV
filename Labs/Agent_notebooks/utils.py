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
from IPython.display import Image as IPyImage, display

# --- Dynamic Library Installation ---
try:
    from dotenv import load_dotenv
    from IPython.display import display, Markdown, Code, Image as IPyImage
    from plantuml import PlantUML
except ImportError:
    print("Core dependencies not found. Please install them by running:")
    print("pip install python-dotenv ipython plantuml anthropic")

# --- Model & Provider Configuration ---
RECOMMENDED_MODELS = {
    # OpenAI Models
    "gpt-4o":        {"provider": "openai", "vision": True, "overview": "Latest flagship model, fast and intelligent"},
    "gpt-4-turbo":   {"provider": "openai", "vision": True, "overview": "GPT-4 Turbo with vision capabilities"},
    "gpt-3.5-turbo": {"provider": "openai", "vision": False, "overview": "Fast and cost-effective for most tasks"},
    
    # Anthropic Models
    "claude-3-opus-20240229":   {"provider": "anthropic", "vision": True, "overview": "Most powerful Claude model"},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "vision": True, "overview": "Balanced Claude model"},
    "claude-3-haiku-20240307":  {"provider": "anthropic", "vision": True, "overview": "Fast and efficient Claude model"},
}

# --- Environment and API Client Setup ---

def load_environment():
    """Loads environment variables from a .env file in the project root."""
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


def setup_llm_client(model_name="gpt-3.5-turbo"):
    """Initializes and returns the API client for the specified model provider."""
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
    """Gets a text completion from the specified LLM."""
    if not client: return "API client not initialized."
    try:
        if api_provider == "openai":
            response = client.chat.completions.create(model=model_name, 
                                                      messages=[{"role": "user", "content": prompt}], 
                                                      temperature=temperature)
            return response.choices[0].message.content
        elif api_provider == "anthropic":
            response = client.messages.create(model=model_name,
                                              max_tokens=4096,
                                              temperature=temperature,
                                              messages=[{"role": "user", "content": prompt}]
                                              )
            return response.content[0].text
    except Exception as e:
        return f"An API error occurred: {e}"
