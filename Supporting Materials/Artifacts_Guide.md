# Artifacts Guide

Utilities in `utils.artifacts` manage generated files with strict directory
rules.

## Resolution Flow

```mermaid
flowchart TD
    A[start] --> B{base_dir provided?}
    B -- yes --> C[use base_dir]
    B -- no --> D{set_artifacts_dir called?}
    D -- yes --> E[use override]
    D -- no --> F{AGA_ARTIFACTS_DIR env?}
    F -- yes --> G[use env dir]
    F -- no --> H[project_root/artifacts]
```

## Security

Paths must remain within the resolved artifacts directory.

```mermaid
flowchart LR
    P[requested path] --> Q{inside artifacts dir?}
    Q -- no --> X[raise ArtifactSecurityError]
    Q -- yes --> Y[allow]
```

## Basic Usage

```python
from utils.artifacts import save_artifact, load_artifact

save_artifact("hello", "greeting.txt")
print(load_artifact("greeting.txt", as_="text"))
```
