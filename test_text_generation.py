import os
import sys
import time
import json
import logging

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import setup_llm_client, get_completion, RECOMMENDED_MODELS

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_text_generation_models():
    """Return model names that can be used for text generation.

    Rules:
      - Exclude audio-transcription-only models (audio_transcription == True)
      - Exclude image-generation-only models (image_generation == True and vision == False)
      - Otherwise include the model (text / multimodal / conversational image models)
    """
    models = []
    for name, cfg in RECOMMENDED_MODELS.items():
        if cfg.get("audio_transcription"):
            continue
        if cfg.get("image_generation") and not cfg.get("vision"):
            # Imagen / DALL·E style models that only generate images don't return text
            continue
        models.append(name)
    return sorted(models)


def run_tests(output_path="artifacts/text_generation_report.json"):
    prompt = "Hello! Tell me a short fun fact about programming."
    models = get_text_generation_models()
    logging.info(f"Found {len(models)} candidate text-generation models to test.")

    results = []

    for model_name in models:
        logging.info(f"\nTesting model: {model_name}")
        client, configured_model_name, provider = setup_llm_client(model_name)

        entry = {
            "model": model_name,
            "provider": provider,
            "client_configured": bool(client),
            "response": None,
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
            resp = get_completion(prompt, client, configured_model_name, provider)
            duration = time.time() - start
            entry["duration_seconds"] = round(duration, 3)

            # Normalize/serialize response so it's JSON-serializable
            def _serialize_response(r):
                if r is None:
                    return None
                # primitive types are fine
                if isinstance(r, (str, int, float, bool)):
                    return r
                # common LLM response shapes
                try:
                    # objects with .text
                    if hasattr(r, 'text') and isinstance(getattr(r, 'text'), (str,)):
                        return getattr(r, 'text')
                    # some OpenAI responses expose choices -> message -> content
                    if hasattr(r, 'choices'):
                        parts = []
                        for c in getattr(r, 'choices'):
                            # support nested message structure
                            msg = getattr(c, 'message', None)
                            if msg and hasattr(msg, 'content'):
                                parts.append(getattr(msg, 'content'))
                            elif hasattr(c, 'text'):
                                parts.append(getattr(c, 'text'))
                            else:
                                parts.append(str(c))
                        return '\n'.join([p for p in parts if p])
                    # dict-like
                    if isinstance(r, dict):
                        # try to pull common fields
                        for key in ('text', 'content', 'response', 'result'):
                            if key in r and isinstance(r[key], str):
                                return r[key]
                        return json.dumps(r)
                except Exception:
                    pass
                # Fallback to string representation
                try:
                    return str(r)
                except Exception:
                    return repr(r)

            entry["response"] = _serialize_response(resp)

            # Heuristic: if function returns an error string, capture it under error
            if isinstance(entry["response"], str) and (entry["response"].startswith("An API error occurred") or entry["response"] == "API client not initialized."):
                entry["error"] = entry["response"]
                logging.info(f"Error from model: {entry['response']}")
            else:
                summary = str(entry.get("response", "") or "").replace('\n', ' ')[:200]
                logging.info(f"Success ({duration:.2f}s): {summary}")

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
    passed = sum(1 for r in results if r.get("response") and not r.get("error"))
    failed = sum(1 for r in results if r.get("error"))
    logging.info(f"\nTest complete. Passed: {passed}, Failed/Skipped: {failed}. Report: {output_path}")


if __name__ == "__main__":
    run_tests()
