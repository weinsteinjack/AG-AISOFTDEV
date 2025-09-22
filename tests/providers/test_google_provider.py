import base64
import os
import sys

import pytest
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "requests" not in sys.modules:
    requests_module = types.ModuleType("requests")

    class _DummyResponse:  # minimal stub used for type hints only
        pass

    class _DummySession:
        def mount(self, *_args, **_kwargs):
            return None

        def request(self, *_args, **_kwargs):
            return _DummyResponse()

    requests_module.Session = _DummySession
    requests_module.Response = _DummyResponse

    adapters_module = types.ModuleType("requests.adapters")

    class _DummyHTTPAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

    adapters_module.HTTPAdapter = _DummyHTTPAdapter
    requests_module.adapters = adapters_module

    sys.modules["requests"] = requests_module
    sys.modules["requests.adapters"] = adapters_module

if "urllib3" not in sys.modules:
    urllib3_module = types.ModuleType("urllib3")
    util_module = types.ModuleType("urllib3.util")
    retry_module = types.ModuleType("urllib3.util.retry")

    class _DummyRetry:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_backoff_time(self):
            return 0

    retry_module.Retry = _DummyRetry
    util_module.retry = retry_module
    urllib3_module.util = util_module

    sys.modules["urllib3"] = urllib3_module
    sys.modules["urllib3.util"] = util_module
    sys.modules["urllib3.util.retry"] = retry_module

from utils.errors import ProviderOperationError
from utils.providers import google


class _FakeImage:
    def __init__(self, *, image_bytes=None, bytes_base64=None, mime_type=None):
        self.image_bytes = image_bytes
        self.bytes_base64 = bytes_base64
        self.mime_type = mime_type


class _FakeGeneratedImage:
    def __init__(self, image):
        self.image = image


class _FakeResponse:
    def __init__(self, generated_images):
        self.generated_images = generated_images


class _FakeModels:
    def __init__(self, response):
        self._response = response
        self.last_prompt = None
        self.last_model = None

    def generate_images(self, *, model, prompt, **_):
        self.last_model = model
        self.last_prompt = prompt
        return self._response


class _FakeGenAiClient:
    def __init__(self, response):
        self.models = _FakeModels(response)


def test_google_image_generation_uses_models_api():
    image_bytes = b"test-bytes"
    response = _FakeResponse([
        _FakeGeneratedImage(_FakeImage(image_bytes=image_bytes, mime_type="image/jpeg"))
    ])
    client = _FakeGenAiClient(response)

    b64, mime = google.image_generation(
        client, "create something", "gemini-2.5-flash-image-preview"
    )

    assert base64.b64decode(b64.encode("utf-8")) == image_bytes
    assert mime == "image/jpeg"


def test_google_image_generation_inline_fallback():
    class _InlineData:
        def __init__(self, data, mime_type):
            self.data = data
            self.mime_type = mime_type

    class _Part:
        def __init__(self, inline_data):
            self.inline_data = inline_data

    class _ContentClient:
        def __init__(self, parts):
            self._parts = parts

        def generate_content(self, *_args, **_kwargs):
            class _Response:
                def __init__(self, parts):
                    self.parts = parts

            return _Response(self._parts)

    inline_bytes = b"inline-data"
    parts = [_Part(_InlineData(inline_bytes, "image/webp"))]

    b64, mime = google.image_generation(
        _ContentClient(parts), "render", "imagen-3.0-generate"
    )

    assert base64.b64decode(b64.encode("utf-8")) == inline_bytes
    assert mime == "image/webp"


def test_google_image_generation_non_image_model():
    class _NoopClient:
        pass

    with pytest.raises(ProviderOperationError):
        google.image_generation(_NoopClient(), "prompt", "gemini-2.5-pro")
