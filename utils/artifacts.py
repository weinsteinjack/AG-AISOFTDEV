import io
import json
import os
from pathlib import Path
from typing import Optional, Union, Literal, Any
from .errors import ArtifactError, ArtifactNotFoundError, ArtifactSecurityError

# Global, overridable at runtime
_ARTIFACTS_DIR: Optional[Path] = None
_PROJECT_MARKERS = frozenset({"pyproject.toml", ".git", "requirements.txt", "setup.cfg", "README.md"})

def detect_project_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in _PROJECT_MARKERS):
            return p
    # Fallback: directory containing this module or CWD
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if any((p / m).exists() for m in _PROJECT_MARKERS):
            return p
    return Path.cwd()

def set_artifacts_dir(path: Union[str, Path]) -> Path:
    """Set a custom artifacts directory; created if missing."""
    global _ARTIFACTS_DIR
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    _ARTIFACTS_DIR = p
    return p

def get_artifacts_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    if _ARTIFACTS_DIR is not None:
        return _ARTIFACTS_DIR
    env_dir = os.getenv("AGA_ARTIFACTS_DIR")
    if env_dir:
        return set_artifacts_dir(env_dir)
    root = detect_project_root()
    return set_artifacts_dir(root / "artifacts")

def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

def resolve_artifact_path(
    filename: Union[str, Path],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    subdir: Optional[Union[str, Path]] = None,
    must_exist: bool = False
) -> Path:
    base = get_artifacts_dir(base_dir)
    target = Path(filename)
    if target.is_absolute():
        resolved = target.resolve()
        if not _is_within(resolved, base):
            raise ArtifactSecurityError(f"Absolute path '{resolved}' is outside artifacts dir '{base}'.")
        final = resolved
    else:
        # allow optional subdir
        final = (base / (Path(subdir) if subdir else Path()) / target).resolve()
        if not _is_within(final, base):
            raise ArtifactSecurityError(f"Resolved path '{final}' escapes artifacts dir '{base}'.")
    if must_exist and not final.exists():
        raise ArtifactNotFoundError(f"Artifact not found: {final}")
    return final

# Public API (backward compatible names)
def save_artifact(
    content: Union[str, bytes, dict, io.BytesIO],
    filename: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    subdir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    encoding: str = "utf-8",
) -> Path:
    path = resolve_artifact_path(filename, base_dir=base_dir, subdir=subdir, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactError(f"Artifact already exists: {path}. Pass overwrite=True to replace.")

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if isinstance(content, bytes):
            tmp.write_bytes(content)
        elif isinstance(content, io.BytesIO):
            tmp.write_bytes(content.getvalue())
        elif isinstance(content, dict):
            tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding=encoding)
        elif isinstance(content, str):
            tmp.write_text(content, encoding=encoding)
        else:
            # Attempt best-effort for objects with .save or .read
            if hasattr(content, "save") and callable(getattr(content, "save")):
                # e.g. PIL Image
                content.save(tmp)
            elif hasattr(content, "read") and callable(getattr(content, "read")):
                # file-like
                tmp.write_bytes(content.read())
            else:
                raise ArtifactError(f"Unsupported content type: {type(content)!r}")
        os.replace(tmp, path)  # atomic
        return path
    except Exception:
        # clean up temp on error
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise

def load_artifact(
    filename: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    subdir: Optional[Union[str, Path]] = None,
    as_: Optional[Literal["bytes", "text", "json", "auto"]] = "auto",
    encoding: str = "utf-8",
) -> Union[bytes, str, dict, Any]:
    path = resolve_artifact_path(filename, base_dir=base_dir, subdir=subdir, must_exist=True)
    if as_ == "bytes":
        return path.read_bytes()
    if as_ == "text":
        return path.read_text(encoding=encoding)
    if as_ == "json":
        return json.loads(path.read_text(encoding=encoding))
    # auto by extension
    ext = path.suffix.lower()
    if ext in {".json"}:
        return json.loads(path.read_text(encoding=encoding))
    if ext in {".txt", ".md", ".csv", ".tsv", ".log"}:
        return path.read_text(encoding=encoding)
    return path.read_bytes()
