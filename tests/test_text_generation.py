import os
import sys
import pytest

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import setup_llm_client, get_completion, RECOMMENDED_MODELS


def get_text_generation_models():
    """Return only models explicitly marked as supporting text generation."""
    return sorted(
        name for name, cfg in RECOMMENDED_MODELS.items() if cfg.get("text_generation")
    )


PROMPT = "Hello! Tell me a short fun fact about programming."


@pytest.mark.integration
@pytest.mark.parametrize("model_name", get_text_generation_models())
def test_text_generation_models(model_name):
    client, configured_model_name, provider = setup_llm_client(model_name)
    if not client:
        pytest.skip(f"Client not configured for provider {provider} or missing API key/library")

    resp = get_completion(PROMPT, client, configured_model_name, provider)
    assert isinstance(resp, str)
    if resp.startswith("An API error occurred") or resp == "API client not initialized.":
        pytest.xfail(f"Provider returned error: {resp}")
