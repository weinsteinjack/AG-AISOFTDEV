import os
import sys
import time
import json
import logging

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import setup_llm_client, get_image_edit_completion, RECOMMENDED_MODELS

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_image_edit_models():
    """Return model names that can be used for image editing."""
    models = []
    # Explicitly test the models that support image-to-image tasks
    edit_models = ["Qwen/Qwen-Image-Edit", "black-forest-labs/FLUX.1-Kontext-dev"]
    for name in edit_models:
        if name in RECOMMENDED_MODELS and RECOMMENDED_MODELS[name].get("image_generation"):
            models.append(name)
    return sorted(models)


def run_tests(output_path="artifacts/image_edit_report.json"):
    prompt = "Add a funny hat to the main subject."
    # Use a pre-existing image from the artifacts as the base for editing
    # Let's find one to use. We'll need to make sure one exists.
    # For now, assuming a file exists at this path.
    # A better implementation would be to generate one first or have a stable test asset.
    image_path = "artifacts/screens/image_1756471808087.png" 
    
    if not os.path.exists(image_path):
        logging.error(f"Base image for editing not found at: {image_path}")
        logging.error("Please generate an image first by running the image generation tests.")
        return

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
        }

        if not client:
            entry["error"] = f"Client not configured for provider {provider} or missing API key/library"
            logging.info(entry["error"])
            results.append(entry)
            continue

        start = time.time()
        try:
            file_path, resp = get_image_edit_completion(prompt, image_path, client, configured_model_name, provider)
            duration = time.time() - start
            entry["duration_seconds"] = round(duration, 3)

            if file_path and resp:
                entry["file_path"] = file_path
                entry["image_url_present"] = True
                summary = f"Image saved to {file_path}"
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
