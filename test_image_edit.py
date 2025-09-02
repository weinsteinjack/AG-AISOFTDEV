import os
import sys
import time
import json
import logging

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    setup_llm_client,
    get_image_edit_completion,
    get_vision_completion,
    RECOMMENDED_MODELS,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_image_edit_models():
    """Return all model names flagged for image modification in RECOMMENDED_MODELS."""
    return sorted(
        name
        for name, cfg in RECOMMENDED_MODELS.items()
        if cfg.get("image_modification") is True
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
            choice = random.choice(candidates)
            logging.info(f"Selected random base image for editing: {choice}")
            return choice
    except Exception as e:
        logging.warning(f"Failed listing screens dir '{screens_dir}': {e}")
    # Fallback: create a simple placeholder if none exist
    try:
        from PIL import Image, ImageDraw
        placeholder = os.path.join(screens_dir, "edit_base_placeholder.png")
        img = Image.new('RGB', (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], outline=(255, 0, 0), width=5)
        draw.text((60, 60), "Edit Me", fill=(0, 0, 0))
        img.save(placeholder, format='PNG')
        logging.info(f"Created placeholder base image at: {placeholder}")
        return placeholder
    except Exception as e:
        logging.error(f"Failed to create placeholder image: {e}")
        return None


def _model_edit_config(model_name: str):
    name = model_name.lower()
    # Default prompt and conservative params
    config = {
        "prompt": "Add a small bright red hat on the subject's head. Keep the background unchanged and preserve original style.",
        "params": {"guidance_scale": 7.0, "num_inference_steps": 28, "strength": 0.75, "seed": 42},
    }
    if "qwen-image-edit" in name:
        config["params"].update({"guidance_scale": 7.5, "num_inference_steps": 30, "strength": 0.8})
    if "flux.1-kontext" in name or "kontext" in name:
        # FLUX models can be sensitive; use a bit lower strength to keep composition
        config["params"].update({"guidance_scale": 5.0, "num_inference_steps": 26, "strength": 0.6})
    return config


def run_tests(output_path="artifacts/image_edit_report.json"):
    verify_keywords = ["hat"]
    image_path = get_random_screen_image()
    if not image_path:
        logging.error("Could not prepare a base image; aborting.")
        return

    # Pick a vision model for descriptions (best-effort)
    vision_model_candidates = ["gpt-4o", "gemini-2.5-flash", "claude-sonnet-4-20250514"]
    vision_client = vision_model = vision_provider = None
    for vm in vision_model_candidates:
        vc, vn, vp = setup_llm_client(vm)
        if vc:
            vision_client, vision_model, vision_provider = vc, vn, vp
            logging.info(f"Using vision model for descriptions: {vision_model} ({vision_provider})")
            break

    models = get_image_edit_models()
    logging.info(f"Found {len(models)} candidate image-editing models to test.")

    results = []

    for model_name in models:
        logging.info(f"\nTesting model: {model_name}")
        client, configured_model_name, provider = setup_llm_client(model_name)

        entry = {
            "model": model_name,
            "provider": provider,
            "client_configured": bool(client),
            "file_path": None,
            "image_url_present": False,
            "error": None,
            "duration_seconds": None,
            "before_description": None,
            "after_description": None,
            "verified_modification": None,
        }

        if not client:
            entry["error"] = f"Client not configured for provider {provider} or missing API key/library"
            logging.info(entry["error"])
            results.append(entry)
            continue

        start = time.time()
        try:
            # Describe image before edit (if vision client is available)
            if vision_client:
                try:
                    before_desc = get_vision_completion(
                        "Briefly describe the image.", image_path, vision_client, vision_model, vision_provider
                    )
                    entry["before_description"] = str(before_desc) if before_desc is not None else None
                except Exception as ve:
                    logging.info(f"Vision (before) error: {ve}")

            # Apply edit with model-specific tuning
            cfg = _model_edit_config(configured_model_name)
            file_path, resp = get_image_edit_completion(
                cfg["prompt"],
                image_path,
                client,
                configured_model_name,
                provider,
                **cfg["params"],
            )
            duration = time.time() - start
            entry["duration_seconds"] = round(duration, 3)

            if file_path and resp:
                entry["file_path"] = file_path
                entry["image_url_present"] = True
                summary = f"Image saved to {file_path}"
                # Describe image after edit and verify heuristic
                if vision_client:
                    try:
                        after_desc = get_vision_completion(
                            "Briefly describe the image.", file_path, vision_client, vision_model, vision_provider
                        )
                        entry["after_description"] = str(after_desc) if after_desc is not None else None
                        if entry["after_description"]:
                            ad = entry["after_description"].lower()
                            bd = (entry["before_description"] or "").lower()
                            found_after = any(k in ad for k in verify_keywords)
                            found_before = any(k in bd for k in verify_keywords)
                            entry["verified_modification"] = bool(found_after and (not found_before or found_after))
                    except Exception as ve:
                        logging.info(f"Vision (after) error: {ve}")
                logging.info(f"Success ({duration:.2f}s): {summary}")
            else:
                # resp contains the error message
                entry["error"] = resp
                logging.info(f"Error from model: {resp}")

        except Exception as e:
            duration = time.time() - start
            entry["error"] = str(e)
            entry["duration_seconds"] = round(duration, 3)
            logging.info(f"Exception while calling model: {e}")

        results.append(entry)
        # small delay between calls
        time.sleep(1)

    # Ensure artifacts folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": int(time.time()), "results": results}, f, indent=2)

    # Print a compact summary
    passed = sum(1 for r in results if r.get("file_path") and not r.get("error"))
    failed = sum(1 for r in results if r.get("error"))
    logging.info(f"\nTest complete. Passed: {passed}, Failed/Skipped: {failed}. Report: {output_path}")


if __name__ == "__main__":
    run_tests()
