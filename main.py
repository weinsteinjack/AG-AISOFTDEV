"""Proxy module to expose the FastAPI ASGI app at the repository root.

This lets `uvicorn main:app` work when the real application lives in
`app/main.py` (module `app.main`).
"""

from app.main import app  # re-export FastAPI instance for uvicorn

__all__ = ["app"]

if __name__ == "__main__":
    # Allow running with: python main.py
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
