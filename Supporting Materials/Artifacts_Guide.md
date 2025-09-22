# Artifacts Guide

The labs save nearly every generated asset—PRDs, prompts, code snippets, evaluation reports—so that students can resume work each day. The `utils.artifacts` module provides a safe, provider-agnostic way to manage those files. This guide explains how the helpers behave, how directory overrides are resolved, and how to integrate them into notebooks or standalone scripts.

---

## Why Use `utils.artifacts`?

* **Consistent storage:** Every lab, notebook, or automation script interacts with artifacts the same way. Students can open a peer’s artifact directory and immediately understand the layout.
* **Security built in:** Path traversal protections prevent accidents such as overwriting files outside the project.
* **Portability:** Artifacts can live anywhere (local folder, mounted drive, temporary path) while the rest of the code remains unchanged.
* **Interoperability:** Tests (`tests/test_artifact_helpers.py`) rely on the same API, ensuring that whatever works in class will pass automated checks.

---

## Directory Resolution Rules

The helper determines the active artifacts directory using the following priority order:

1. **`base_dir` argument** – One-off override passed directly into a helper call.
2. **`set_artifacts_dir()`** – Session-wide override (e.g., in a notebook cell).
3. **`AGA_ARTIFACTS_DIR` environment variable** – Useful for CI or Docker containers.
4. **Project root default** – Automatically detected by walking upward from the current working directory until a marker file (`pyproject.toml`, `.git`, `requirements.txt`, `setup.cfg`, or `README.md`) is found, then creating `<project_root>/artifacts` on demand.

Because the default directory is created lazily, you will not see an `artifacts/` folder in a fresh clone until you save something.

> **Security note:** Regardless of the chosen base directory, all resolved paths must remain inside that folder. If a filename attempts to escape (e.g., `../../etc/passwd`), an `ArtifactSecurityError` is raised before any file I/O occurs.

---

## Quick Start

```python
from utils.artifacts import save_artifact, load_artifact

# Persist structured data
prd = {"feature": "AI onboarding", "status": "draft"}
path = save_artifact(prd, "day1/prd.json")
print(f"Saved to {path}")

# Retrieve it later (auto-detects JSON by extension)
loaded = load_artifact("day1/prd.json")
assert loaded["status"] == "draft"
```

### Common Helpers at a Glance

| Function | Purpose |
| --- | --- |
| `set_artifacts_dir(path)` | Permanently override the active directory for the current Python process. Creates the folder if missing. |
| `get_artifacts_dir(base_dir=None)` | Resolve the effective directory given an optional `base_dir`. |
| `resolve_artifact_path(filename, *, base_dir=None, subdir=None, must_exist=False)` | Convert a relative filename into an absolute `Path` inside the sandbox, optionally ensuring it already exists. |
| `save_artifact(content, filename, *, base_dir=None, subdir=None, overwrite=False, encoding="utf-8")` | Persist strings, bytes, dictionaries, `BytesIO`, or PIL images with automatic JSON/text encoding. |
| `load_artifact(filename, *, base_dir=None, subdir=None, as_="auto", encoding="utf-8")` | Read content back as bytes, text, JSON, or auto-detected based on extension. |

---

## Usage Patterns

### 1. Notebook Sessions

```python
from utils.artifacts import set_artifacts_dir, save_artifact
from datetime import date

set_artifacts_dir(f"~/ai_course/artifacts/{date.today():%Y%m%d}")
summary = "Key persona insights from stakeholder interview."
save_artifact(summary, "day1/persona_notes.md")
```

This pattern keeps each cohort’s output separate while allowing notebooks to call `save_artifact` without further changes.

### 2. Temporary Directories in Tests

All shipped tests create a temporary directory and call `set_artifacts_dir()` so they never touch real student data:

```python
def test_save_and_load_artifact(tmp_path):
    from utils import artifacts
    artifacts.set_artifacts_dir(tmp_path)
    artifacts.save_artifact({"x": 1}, "sample.json")
    assert artifacts.load_artifact("sample.json", as_="json") == {"x": 1}
```

When writing your own tests, follow the same approach to avoid polluting the main artifact folder.

### 3. Subdirectories & Prefixes

The helpers automatically strip redundant prefixes so you can call `save_artifact("hi", "artifacts/notes.txt")` without creating nested `artifacts/artifacts` folders. If you still want a nested layout, use the `subdir` argument explicitly:

```python
save_artifact("schema draft", "db/schema.sql", subdir="day2")
# -> stores file at <artifacts_dir>/day2/db/schema.sql
```

### 4. Binary Content

Pass bytes or file-like objects to `save_artifact` for models, images, or audio clips:

```python
with open("diagram.png", "rb") as fh:
    save_artifact(fh, "day3/system_diagram.png", overwrite=True)
```

When you later call `load_artifact(..., as_="bytes")`, you will receive raw bytes suitable for writing back to disk or embedding in reports.

---

## Error Handling

* `ArtifactSecurityError` – Raised when a resolved path leaves the sandbox. Double-check filenames provided by users or LLMs.
* `ArtifactNotFoundError` – Raised by `load_artifact` when `must_exist=True` or when attempting to load a missing file.
* `ArtifactError` – Raised when overwriting a file without `overwrite=True` or when the content type is unsupported.

Wrap calls in `try`/`except` blocks when handling untrusted filenames or when saving optional assets:

```python
from utils.artifacts import save_artifact, ArtifactError

try:
    save_artifact(dataframe.to_csv(index=False), "reports/summary.csv")
except ArtifactError as exc:
    print(f"Could not persist report: {exc}")
```

---

## Advanced Tips

1. **Environment-based overrides:** Configure `AGA_ARTIFACTS_DIR` in CI/CD or Docker so the same notebooks write to `/data/artifacts` in production while defaulting to `./artifacts` locally.
2. **Project discovery:** `detect_project_root()` recognises the repository root even when a notebook launches from a nested folder (for example, inside `Labs/Day_05_...`). You rarely need to call it yourself, but it guarantees that relative paths remain stable.
3. **Atomic writes:** Files are written to a temporary sibling path and then atomically replaced. This protects against corrupted artifacts if a notebook crashes mid-write.
4. **Extension-aware loading:** `load_artifact(..., as_="auto")` treats `.json`, `.md`, `.txt`, `.csv`, `.tsv`, `.log`, `.sql`, and `.py` as text; everything else returns bytes. Specify `as_` explicitly if you want different behaviour.
5. **Collaboration workflows:** Encourage students to save intermediate prompts, evaluation outputs, and agent transcripts as Markdown. Sharing the artifacts directory with classmates accelerates peer reviews and capstone retrospectives.

---

## Troubleshooting Checklist

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ArtifactSecurityError: Resolved path ... escapes artifacts dir` | Filename contains `../` or an absolute path. | Clean the filename or use `subdir` to control the destination. |
| `ArtifactNotFoundError` when loading | File was never saved or you are pointing at the wrong base directory. | Call `get_artifacts_dir()` to confirm the active path, then re-run the cell that saves the file. |
| File saved but contents empty | Provided object lacked a `.read()` method and was not recognised as bytes/string/dict. | Convert the object to one of the supported types before calling `save_artifact`. |
| Artifacts scattered across machines | Different developers using different folders. | Agree on an `AGA_ARTIFACTS_DIR` convention or call `set_artifacts_dir()` at the start of each notebook. |

---

Using `utils.artifacts` consistently keeps classroom exercises reproducible, simplifies grading, and prepares students for production-ready AI workflows.
