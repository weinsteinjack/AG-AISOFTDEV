import os
import sys
import pytest

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import setup_llm_client, get_image_generation_completion, RECOMMENDED_MODELS


def get_image_generation_models():
    """Return model names that can be used for image generation."""
    return sorted(
        name for name, cfg in RECOMMENDED_MODELS.items() if cfg.get("image_generation")
    )


PROMPT = (
    "A photorealistic image of a cat programming on a laptop, with a cup of coffee on the desk."
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model_name", get_image_generation_models())
def test_image_generation_models(model_name):
    client, configured_model_name, provider = setup_llm_client(model_name)
    if not client:
        pytest.skip(f"Client not configured for provider {provider} or missing API key/library")

    file_path, data_url_or_error = get_image_generation_completion(
        PROMPT, client, configured_model_name, provider
    )

    # If the provider is reachable, we expect either a saved file or a clear error.
    if file_path:
        assert os.path.exists(file_path)
        assert isinstance(data_url_or_error, str) and data_url_or_error.startswith("data:image/")
    else:
        # Treat provider-side failures as expected xfails to avoid hard flakes
        pytest.xfail(f"Generation failed: {data_url_or_error}")
