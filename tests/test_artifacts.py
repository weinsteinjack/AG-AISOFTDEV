from pathlib import Path
import shutil
import pytest

import utils.artifacts as art
from utils.artifacts import (
    save_artifact,
    load_artifact,
    set_artifacts_dir,
    ArtifactSecurityError,
    ArtifactNotFoundError,
)


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    art._ARTIFACTS_DIR = None
    monkeypatch.delenv("AGA_ARTIFACTS_DIR", raising=False)
    yield
    art._ARTIFACTS_DIR = None
    monkeypatch.delenv("AGA_ARTIFACTS_DIR", raising=False)
    if Path("artifacts").exists():
        shutil.rmtree("artifacts")


def test_default_artifacts_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_artifact("hello", "demo/hello.txt", overwrite=True)
    project_root = Path(__file__).resolve().parents[1]
    expected = project_root / "artifacts" / "demo/hello.txt"
    assert path == expected
    assert load_artifact("demo/hello.txt", as_="text") == "hello"


def test_set_artifacts_dir_override(tmp_path):
    set_artifacts_dir(tmp_path)
    path = save_artifact("hi", "file.txt")
    assert path == tmp_path / "file.txt"
    assert load_artifact("file.txt", as_="text") == "hi"


def test_env_var_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AGA_ARTIFACTS_DIR", str(tmp_path))
    art._ARTIFACTS_DIR = None
    dir_path = art.get_artifacts_dir()
    assert dir_path == tmp_path


def test_per_call_override(tmp_path):
    base = tmp_path / "data_store"
    path = save_artifact(b"raw", "blob.bin", base_dir=base, subdir="exp1")
    assert path == base.resolve() / "exp1" / "blob.bin"


def test_path_traversal_blocked(tmp_path):
    set_artifacts_dir(tmp_path)
    with pytest.raises(ArtifactSecurityError):
        save_artifact("bad", "../../etc/passwd")


def test_atomic_writes(tmp_path):
    set_artifacts_dir(tmp_path)

    class Boom:
        def save(self, path):
            with open(path, "w") as f:
                f.write("partial")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        save_artifact(Boom(), "boom.txt")

    target = tmp_path / "boom.txt"
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_load_missing_artifact(tmp_path):
    set_artifacts_dir(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        load_artifact("missing.txt")
