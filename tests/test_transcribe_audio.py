import os
import sys
import pytest

# Ensure repository root is on the path so we can import local modules reliably
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import setup_llm_client, transcribe_audio, RECOMMENDED_MODELS


def get_transcription_models():
    return sorted(
        name for name, cfg in RECOMMENDED_MODELS.items() if cfg.get("audio_transcription")
    )


def require_audio_path(path="artifacts/audio/sample_test.wav"):
    if not os.path.exists(path):
        pytest.skip(
            f"Audio sample not found at: {path}. Place a small WAV/MP3 there to run integration tests."
        )
    return path


@pytest.mark.integration
@pytest.mark.parametrize("model_name", get_transcription_models())
def test_audio_transcription_models(model_name):
    audio_path = require_audio_path()
    client, configured_model_name, provider = setup_llm_client(model_name)
    if not client:
        pytest.skip(f"Client not configured for provider {provider} or missing API key/library")

    text = transcribe_audio(audio_path, client, configured_model_name, provider)
    assert isinstance(text, str)
    if text.startswith("An API error occurred"):
        pytest.xfail(f"Provider returned error: {text}")
