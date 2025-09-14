import os
import sys
import pytest

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    setup_llm_client,
    get_image_edit_completion,
    RECOMMENDED_MODELS,
)


def get_image_edit_models():
    """Return all model names flagged for image modification in RECOMMENDED_MODELS."""
    return sorted(
        name for name, cfg in RECOMMENDED_MODELS.items() if cfg.get("image_modification") is True
    )


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
            import random
            return random.choice(candidates)
    except Exception:
        pass
    # Fallback: create a simple placeholder if none exist
    try:
        from PIL import Image, ImageDraw
        placeholder = os.path.join(screens_dir, "edit_base_placeholder.png")
        img = Image.new('RGB', (256, 256), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], outline=(255, 0, 0), width=5)
        draw.text((60, 60), "Edit Me", fill=(0, 0, 0))
        img.save(placeholder, format='PNG')
        return placeholder
    except Exception:
        return None


def _model_edit_config(model_name: str):
    name = model_name.lower()
    # Default prompt and conservative params
    config = {
        "prompt": (
            "Add a small bright red hat on the subject's head. Keep the background unchanged and preserve original style."
        ),
        "params": {"guidance_scale": 7.0, "num_inference_steps": 28, "strength": 0.75, "seed": 42},
    }
    if "qwen-image-edit" in name:
        config["params"].update({"guidance_scale": 7.5, "num_inference_steps": 30, "strength": 0.8})
    if "flux.1-kontext" in name or "kontext" in name:
        # FLUX models can be sensitive; use a bit lower strength to keep composition
        config["params"].update({"guidance_scale": 5.0, "num_inference_steps": 26, "strength": 0.6})
    return config


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model_name", get_image_edit_models())
def test_image_edit_models(model_name):
    image_path = get_random_screen_image()
    if not image_path:
        pytest.skip("No base image available and placeholder could not be created (Pillow missing?)")

    client, configured_model_name, provider = setup_llm_client(model_name)
    if not client:
        pytest.skip(f"Client not configured for provider {provider} or missing API key/library")

    cfg = _model_edit_config(configured_model_name)
    file_path, data_url_or_error = get_image_edit_completion(
        cfg["prompt"], image_path, client, configured_model_name, provider, **cfg["params"]
    )

    if file_path:
        assert os.path.exists(file_path)
        assert isinstance(data_url_or_error, str) and data_url_or_error.startswith("data:image/")
    else:
        pytest.xfail(f"Image edit failed: {data_url_or_error}")
