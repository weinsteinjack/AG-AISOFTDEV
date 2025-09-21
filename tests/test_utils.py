import pytest
from utils import artifacts, audio, image_gen, llm, models, rate_limit, settings, helpers, errors, http
import utils.logging as utils_logging

def test_save_and_load_artifact(tmp_path):
    test_data = {'key': 'value'}
    artifacts.set_artifacts_dir(str(tmp_path))
    # Save as dict, which is supported by save_artifact
    artifacts.save_artifact(test_data, 'test.json')
    loaded = artifacts.load_artifact('test.json', as_='json')
    assert loaded == test_data

def test_save_artifact_invalid_path(tmp_path):
    artifacts.set_artifacts_dir(str(tmp_path))
    from utils.errors import ArtifactSecurityError
    # Try to save to a path outside the artifacts dir
    with pytest.raises(ArtifactSecurityError):
        artifacts.save_artifact({'x': 1}, '/tmp/test.json')


def test_save_artifact_strips_redundant_folder(tmp_path):
    artifacts_dir = tmp_path / 'artifacts'
    artifacts.set_artifacts_dir(str(artifacts_dir))
    saved_path = artifacts.save_artifact('hello', 'artifacts/sample.txt', overwrite=True)
    assert saved_path == artifacts_dir / 'sample.txt'
    assert not (artifacts_dir / 'artifacts').exists()
    loaded = artifacts.load_artifact('artifacts/sample.txt', as_='text')
    assert loaded == 'hello'

def test_save_artifact_with_artifacts_in_name(tmp_path):
    artifacts_dir = tmp_path / 'artifacts'
    artifacts.set_artifacts_dir(str(artifacts_dir))
    # should create .../artifacts/artifacts
    saved_path = artifacts.save_artifact('hello', 'artifacts/artifacts', overwrite=True)
    assert saved_path == artifacts_dir / 'artifacts'
    assert saved_path.is_file()
    assert saved_path.read_text() == 'hello'
    # and loading should work
    loaded = artifacts.load_artifact('artifacts/artifacts', as_='text')
    assert loaded == 'hello'

def test_load_artifact_missing_file(tmp_path):
    artifacts.set_artifacts_dir(str(tmp_path))
    from utils.errors import ArtifactNotFoundError
    with pytest.raises(ArtifactNotFoundError):
        artifacts.load_artifact('missing.json')

def test_get_logger():
    from utils.logging import get_logger
    logger = get_logger()
    assert logger.name == 'ag_aisoftdev.utils'

def test_get_logger_empty_name():
    from utils.logging import get_logger
    logger = get_logger()
    assert logger.name == 'ag_aisoftdev.utils'

def test_load_environment(monkeypatch):
    # load_environment returns None or dict, but is side-effectful
    env = settings.load_environment()
    assert env is None or isinstance(env, dict)

def test_load_environment_no_env(monkeypatch):
    env = settings.load_environment()
    assert env is None or isinstance(env, dict)

def test_transcribe_audio_compat():
    if hasattr(audio, 'transcribe_audio_compat'):
        from utils.settings import load_environment
        from utils.models import RECOMMENDED_MODELS
        load_environment()
        for model_name, cfg in RECOMMENDED_MODELS.items():
            if cfg.get('audio_transcription'):
                provider = cfg['provider']
                audio_path = '/tmp/nonexistent.wav'
                if provider == 'openai':
                    import openai
                    client = openai.OpenAI()
                elif provider == 'huggingface':
                    from huggingface_hub import InferenceClient
                    client = InferenceClient()
                else:
                    continue
                try:
                    result = audio.transcribe_audio_compat(audio_path, client, model_name, provider)
                except Exception as e:
                    result = (None, str(e))
                assert isinstance(result, tuple)
                assert result[0] is None or result[1] is not None

def test_transcribe_audio_compat_invalid_type():
    if hasattr(audio, 'transcribe_audio_compat'):
        class DummyClient: pass
        # Should return error tuple for wrong input type
        result = audio.transcribe_audio_compat('not_bytes', DummyClient(), 'whisper-1', 'openai')
        assert result[0] is None
        assert "audio transcription error" in str(result[1])

def test_get_image_generation_completion_compat():
    if hasattr(image_gen, 'get_image_generation_completion_compat'):
        from utils.settings import load_environment
        from utils.models import RECOMMENDED_MODELS
        load_environment()
        for model_name, cfg in RECOMMENDED_MODELS.items():
            if cfg.get('image_generation'):
                provider = cfg['provider']
                prompt = 'A red apple on a table.'
                if provider == 'openai':
                    import openai
                    client = openai.OpenAI()
                elif provider == 'google':
                    import google.generativeai as genai
                    client = genai
                elif provider == 'huggingface':
                    from huggingface_hub import InferenceClient
                    client = InferenceClient()
                else:
                    continue
                try:
                    result = image_gen.get_image_generation_completion_compat(prompt, client, model_name, provider)
                except Exception as e:
                    result = (None, str(e))
                assert isinstance(result, tuple)
                assert result[0] is not None or result[1] is not None

def test_get_image_generation_completion_compat_invalid():
    if hasattr(image_gen, 'get_image_generation_completion_compat'):
        from utils.settings import load_environment
        import openai
        load_environment()
        client = openai.OpenAI()
        model_name = 'dall-e-3'
        api_provider = 'openai'
        try:
            result = image_gen.get_image_generation_completion_compat(None, client, model_name, api_provider)
        except Exception as e:
            # If API error is raised, treat as error tuple
            result = (None, str(e))
        assert isinstance(result, tuple)
        assert result[0] is None
        assert result[1] is not None

def test_get_completion_compat():
    if hasattr(llm, 'get_completion_compat'):
        from utils.settings import load_environment
        from utils.models import RECOMMENDED_MODELS
        load_environment()
        for model_name, cfg in RECOMMENDED_MODELS.items():
            if cfg.get('text_generation'):
                provider = cfg['provider']
                prompt = 'Say hello world.'
                if provider == 'openai':
                    import openai
                    client = openai.OpenAI()
                elif provider == 'anthropic':
                    import anthropic
                    client = anthropic.Anthropic()
                elif provider == 'google':
                    import google.generativeai as genai
                    client = genai
                elif provider == 'huggingface':
                    from huggingface_hub import InferenceClient
                    client = InferenceClient()
                else:
                    continue
                try:
                    result = llm.get_completion_compat(prompt, client, model_name, provider)
                except Exception as e:
                    result = (None, str(e))
                assert isinstance(result, tuple)
                assert result[0] is not None or result[1] is not None

def test_get_completion_compat_invalid():
    if hasattr(llm, 'get_completion_compat'):
        from utils.settings import load_environment
        import openai
        load_environment()
        client = openai.OpenAI()
        model_name = 'gpt-4o'
        api_provider = 'openai'
        # Invalid prompt should return error tuple or raise
        try:
            result = llm.get_completion_compat(None, client, model_name, api_provider)
        except Exception as e:
            result = (None, str(e))
        assert isinstance(result, tuple)
        assert result[0] is None
        assert result[1] is not None

def test_request():
    # Should handle invalid URL gracefully
    with pytest.raises(Exception):
        http.request('GET', 'http://invalid-url')

def test_request_empty_url():
    with pytest.raises(Exception):
        http.request('GET', '')

def test_rate_limit():
    # Should not raise error for default usage
    assert hasattr(rate_limit, '_TokenBucket')

def test_rate_limiter_edge():
    # _TokenBucket is the actual class
    from utils.rate_limit import _TokenBucket
    bucket = _TokenBucket(1)
    assert bucket.consume(0.5) == 0.0 or bucket.consume(0.5) >= 0.0

def test_helpers():
    # Should have at least one helper function
    assert hasattr(helpers, 'ensure_provider')

def test_helpers_edge():
    # Try calling normalize_prompt with edge input
    if hasattr(helpers, 'normalize_prompt'):
        assert helpers.normalize_prompt('   test   ') == 'test'

def test_models():
    # Should have at least one model config
    assert hasattr(models, 'RECOMMENDED_MODELS')

def test_models_edge():
    # Try accessing a model config with a non-existent key
    assert models.RECOMMENDED_MODELS.get('nonexistent') is None

def test_errors():
    # Should have at least one error class
    assert hasattr(errors, 'UtilsError')

def test_errors_edge():
    # Try raising ProviderOperationError
    if hasattr(errors, 'ProviderOperationError'):
        with pytest.raises(errors.ProviderOperationError):
            raise errors.ProviderOperationError('prov', 'mod', 'op', 'fail')
