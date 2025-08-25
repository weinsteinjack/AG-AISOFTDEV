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
    "gpt-4o":        {"provider": "openai", "vision": True, "image_generation": False},
    "gpt-4.1":       {"provider": "openai", "vision": True, "image_generation": False},
    "gpt-4.1-mini":  {"provider": "openai", "vision": True, "image_generation": False},
    "gpt-4.1-nano":  {"provider": "openai", "vision": True, "image_generation": False},
    "gpt-4.5":       {"provider": "openai", "vision": True, "image_generation": False},
    "o3":            {"provider": "openai", "vision": True, "image_generation": False},
    "o4-mini":       {"provider": "openai", "vision": True, "image_generation": False},
    "codex-mini":    {"provider": "openai", "vision": False, "image_generation": False},

    "claude-3-opus-20240229":   {"provider": "anthropic", "vision": True, "image_generation": False},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "vision": True, "image_generation": False},
    "claude-3-haiku-20240307":  {"provider": "anthropic", "vision": True, "image_generation": False},
    "gemini-2.5-pro":         {"provider": "gemini", "vision": True, "image_generation": False},
    "gemini-2.5-flash":       {"provider": "gemini", "vision": True, "image_generation": False},
    "gemini-2.5-flash-lite":  {"provider": "gemini", "vision": True, "image_generation": False},
    "gemini-veo-3":           {"provider": "gemini", "vision": True, "image_generation": False},
    "gemini-deep-think":      {"provider": "gemini", "vision": True, "image_generation": False},
    "meta-llama/Llama-3.3-70B-Instruct": {"provider": "huggingface", "vision": False, "image_generation": False},
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5": {"provider": "huggingface", "vision": False, "image_generation": False},
    "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3": {"provider": "huggingface", "vision": False, "image_generation": False},
    "mistralai/Mistral-7B-Instruct-v0.3": {"provider": "huggingface", "vision": False, "image_generation": False},
    "deepseek-ai/DeepSeek-VL2":         {"provider": "huggingface", "vision": True, "image_generation": False},
    "deepseek-ai/DeepSeek-VL2-Small":   {"provider": "huggingface", "vision": True, "image_generation": False},
    "deepseek-ai/DeepSeek-VL2-Tiny":    {"provider": "huggingface", "vision": True, "image_generation": False},
    "deepseek-ai/Janus-Pro-7B":         {"provider": "huggingface", "vision": True, "image_generation": False},
    "imagen-3.0-generate-002": {"provider": "google", "vision": False, "image_generation": True}, # Added for image generation
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


def setup_llm_client(model_name="gpt-4o"):
    """
    Configures and returns an LLM client based on the specified model name.
    Supports OpenAI, Anthropic, Hugging Face, and Google Gemini.
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
        elif api_provider == "gemini" or api_provider == "google": # Google for image generation
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") # Use GOOGLE_API_KEY for both Gemini text and Imagen
            if not api_key: raise ValueError("GOOGLE_API_KEY not found in .env file.")
            genai.configure(api_key=api_key)
            if config["image_generation"]:
                # For image generation, the client is not directly a GenerativeModel instance
                # but rather the genai module itself for the predict call.
                client = genai
            else:
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
    """Sends a text-only prompt to the LLM and returns the completion."""
    if not client: return "API client not initialized."
    try:
        if api_provider == "openai":
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temperature)
            return response.choices[0].message.content
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
        elif api_provider == "gemini":
            response = client.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"An API error occurred: {e}"

def get_vision_completion(prompt, image_url, client, model_name, api_provider):
    """Sends an image and a text prompt to a vision-capable LLM and returns the completion."""
    if not client: return "API client not initialized."
    if not RECOMMENDED_MODELS.get(model_name, {}).get("vision"):
        return f"Error: Model '{model_name}' does not support vision."
    try:
        # Fetch image from URL and convert to base64 for Gemini/HuggingFace if needed, or pass URL for OpenAI
        if api_provider == "openai":
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}], max_tokens=4096)
            return response.choices[0].message.content
        elif api_provider == "anthropic":
            response_img = requests.get(image_url)
            response_img.raise_for_status()
            img_data = base64.b64encode(response_img.content).decode('utf-8')
            mime_type = response_img.headers['Content-Type']
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
        elif api_provider == "gemini":
            # For Gemini, we need to convert the image URL to inlineData (base64)
            response_img = requests.get(image_url)
            response_img.raise_for_status()
            img_data = base64.b64encode(response_img.content).decode('utf-8')
            mime_type = response_img.headers['Content-Type']

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inlineData": {"mimeType": mime_type, "data": img_data}}
                        ]
                    }
                ]
            }
            # Use the raw genai client for direct API call
            # The client here is actually `genai` module from setup_llm_client
            api_key = "" # Canvas will provide this.
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Unexpected Gemini vision response structure: {result}"
        elif api_provider == "huggingface":
            response_img = requests.get(image_url)
            response_img.raise_for_status()
            img = Image.open(BytesIO(response_img.content))
            response = client.image_to_text(image=img, prompt=prompt)
            return response
    except Exception as e:
        return f"An API error occurred during vision completion: {e}"

def get_image_generation_completion(prompt, client, model_name, api_provider):
    """Generates an image from a text prompt using an image generation LLM."""
    if not client: return "API client not initialized."
    if not RECOMMENDED_MODELS.get(model_name, {}).get("image_generation"):
        return f"Error: Model '{model_name}' does not support image generation."

    # Display a loading indicator
    print("Generating image... This may take a moment.")
    display(Markdown("⏳ Generating image..."))
    start_time = time.time()

    try:
        # For imagen-3.0-generate-002, the client is the `genai` module itself.
        payload = {"instances": {"prompt": prompt}, "parameters": {"sampleCount": 1}}
        apiKey = "" # Canvas will automatically provide this.
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict?key={apiKey}"

        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()

        if result.get("predictions") and len(result["predictions"]) > 0 and result["predictions"][0].get("bytesBase64Encoded"):
            image_b64 = result["predictions"][0]["bytesBase64Encoded"]
            image_url = f"data:image/png;base64,{image_b64}"
            end_time = time.time()
            print(f"✅ Image generated in {end_time - start_time:.2f} seconds.")
            return image_url
        else:
            return f"Error: Unexpected image generation response structure: {result}"
    except Exception as e:
        return f"An API error occurred during image generation: {e}"


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
    """Renders PlantUML code and saves it as a PNG image."""
    try:
        # FIX: Corrected the PlantUML URL
        pl = PlantUML(url='http://www.plantuml.com/plantuml/img/')
        project_root = _find_project_root()
        
        full_path = os.path.join(project_root, output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        pl.processes(puml_code, outfile=full_path)
        print(f"✅ Diagram rendered and saved to: {output_path}")
        display(IPyImage(url=full_path))
    except Exception as e:
        print(f"❌ Error rendering PlantUML diagram: {e}")

