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
from utils.logging import get_logger

logger = get_logger()

# --- Dynamic Library Installation ---
try:
    from dotenv import load_dotenv
    from IPython.display import display, Markdown, Code, Image as IPyImage
    from plantuml import PlantUML
except ImportError:
    logger.warning(
        "Core dependencies not found. Please install them by running:"
    )
    logger.warning("pip install python-dotenv ipython plantuml anthropic")

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
        logger.warning(".env file not found. API keys may not be loaded.")


def setup_llm_client(model_name="gpt-3.5-turbo"):
    """
    Initializes and returns the API client for the specified model provider.
    
    This function configures the appropriate API client for the specified model,
    handling authentication and configuration for OpenAI and Anthropic providers.
    It automatically loads environment variables and validates API keys.
    
    Args:
        model_name (str, optional): The identifier of the model to use. Must be
            a key in the RECOMMENDED_MODELS dictionary. Defaults to "gpt-3.5-turbo".
            Examples: "gpt-4o", "claude-3-opus-20240229", "gpt-4-turbo"
    
    Returns:
        tuple: A 3-element tuple containing:
            - client: The initialized API client object (varies by provider)
                - OpenAI: OpenAI client instance
                - Anthropic: Anthropic client instance
            - model_name (str): The model name (echoed back)
            - api_provider (str): The provider name ("openai" or "anthropic")
            
            Returns (None, None, None) if initialization fails.
    
    Raises:
        None: This function handles all errors gracefully and prints error messages
            instead of raising exceptions.
    
    Notes:
        - Automatically calls load_environment() to load .env file
        - Validates that the model exists in RECOMMENDED_MODELS
        - Checks for required API keys in environment variables
        - Handles ImportError if provider libraries aren't installed
        - Prints success/error messages to console
        - Supports OpenAI and Anthropic providers only in this version
    
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
        - RECOMMENDED_MODELS: Global dictionary with model configurations
    """
    load_environment()
    if model_name not in RECOMMENDED_MODELS:
        logger.error(
            "Model '%s' is not in the list of recommended models.", model_name
        )
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
        logger.error(
            "The required library for '%s' is not installed.", api_provider
        )
        return None, None, None
    except ValueError as e:
        logger.error("%s", e)
        return None, None, None
    logger.info(
        "LLM Client configured",
        extra={"provider": api_provider, "model": model_name},
    )
    return client, model_name, api_provider

# --- Core Interaction Functions ---

def get_completion(prompt, client, model_name, api_provider, temperature=0.7):
    """
    Gets a text completion from the specified LLM.
    
    This function provides a unified interface for getting text completions from
    OpenAI and Anthropic models. It handles provider-specific API differences
    and error cases gracefully.
    
    Args:
        prompt (str): The text prompt to send to the model. This is the user's
            input or question that the model should respond to.
        client: The initialized API client object from setup_llm_client().
            Type varies by provider (OpenAI or Anthropic client).
        model_name (str): The identifier of the model to use for completion.
        api_provider (str): The provider name ("openai" or "anthropic").
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
        - OpenAI: Uses chat completions API
        - Anthropic: Uses messages API with max_tokens=4096
        - Returns descriptive error messages if API calls fail
        - Temperature parameter affects output randomness for both providers
    
    Example:
        >>> client, model, provider = setup_llm_client("gpt-4o")
        >>> response = get_completion(
        ...     "What is the capital of France?",
        ...     client, model, provider, temperature=0.5
        ... )
        >>> logger.info(response)
        "The capital of France is Paris."
        
        >>> # Handle API errors gracefully
        >>> response = get_completion("Hello", None, "gpt-4o", "openai")
        >>> logger.info(response)
        "API client not initialized."
    
    Dependencies:
        - Provider-specific client libraries (openai or anthropic)
    """
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
