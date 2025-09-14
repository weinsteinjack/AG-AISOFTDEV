import os
import sys
import random
import pytest

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import setup_llm_client, get_vision_completion, RECOMMENDED_MODELS


def get_vision_models():
    """Return model names that can accept image inputs (vision == True)."""
    return sorted(name for name, cfg in RECOMMENDED_MODELS.items() if cfg.get("vision"))


def ensure_test_image(path="artifacts/screens/test_vision_input.png"):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (256, 256), color=(220, 240, 255))
            d = ImageDraw.Draw(img)
            d.ellipse([80, 80, 176, 176], outline=(0, 100, 200), width=4)
            d.text((90, 90), "Hi", fill=(0, 0, 0))
            img.save(path, format='PNG')
        except Exception:
            return None
    return path


def get_random_screen_image(screens_dir="artifacts/screens"):
    """Pick a random image from the screens folder; create one if empty."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    if not os.path.exists(screens_dir):
        os.makedirs(screens_dir, exist_ok=True)
    try:
        candidates = []
        for name in os.listdir(screens_dir):
            p = os.path.join(screens_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
                candidates.append(p)
        if candidates:
            return random.choice(candidates)
    except Exception:
        pass
    # Fallback: ensure a placeholder exists and return it
    return ensure_test_image(os.path.join(screens_dir, "test_vision_input.png"))


PROMPT = "Briefly describe the image content."


@pytest.mark.integration
@pytest.mark.parametrize("model_name", get_vision_models())
def test_vision_completion_models(model_name):
    image_path = get_random_screen_image()
    if not image_path:
        pytest.skip("Could not prepare a test image (Pillow missing?)")

    client, configured_model_name, provider = setup_llm_client(model_name)
    if not client:
        pytest.skip(f"Client not configured for provider {provider} or missing API key/library")

    resp = get_vision_completion(PROMPT, image_path, client, configured_model_name, provider)
    assert isinstance(resp, str)
    if resp.startswith("An API error occurred"):
        pytest.xfail(f"Provider returned error: {resp}")
