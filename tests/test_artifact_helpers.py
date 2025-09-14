import os
import sys
import pytest

# Ensure repository root on sys.path to import `utils`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import save_artifact, load_artifact, _find_project_root


def artifacts_root():
    return os.path.join(_find_project_root(), 'artifacts')


def test_save_and_load_under_artifacts_simple():
    # Save without the 'artifacts/' prefix; should go to artifacts/<path>
    rel_path = 'test_artifacts/simple.txt'
    content = 'hello world'
    save_artifact(content, rel_path)

    # Verify file exists under artifacts
    expected = os.path.join(artifacts_root(), rel_path)
    assert os.path.exists(expected)
    assert load_artifact(rel_path) == content


def test_save_and_load_with_artifacts_prefix():
    rel_path = 'artifacts/test_artifacts/prefixed.txt'
    content = 'prefixed content'
    save_artifact(content, rel_path)

    expected = os.path.join(_find_project_root(), rel_path)
    assert os.path.exists(expected)
    # Load works with either form
    assert load_artifact(rel_path) == content
    assert load_artifact('test_artifacts/prefixed.txt') == content


@pytest.mark.parametrize(
    'bad', [
        '../outside.txt',
        '../../secrets.txt',
        '/tmp/absolute_outside.txt',
    ]
)
def test_save_rejects_outside_artifacts(bad):
    with pytest.raises(ValueError):
        save_artifact('nope', bad)


@pytest.mark.parametrize(
    'bad', [
        '../outside.txt',
        '../../secrets.txt',
        '/tmp/absolute_outside.txt',
    ]
)
def test_load_rejects_outside_artifacts(bad):
    with pytest.raises(ValueError):
        load_artifact(bad)

