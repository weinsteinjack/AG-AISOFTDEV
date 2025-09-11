import os
from .logging import get_logger

logger = get_logger()

# Optional dependencies: dotenv and IPython display utilities
try:
    from dotenv import load_dotenv
    from IPython.display import display, Markdown, Image as IPyImage
    from plantuml import PlantUML
except ImportError:  # pragma: no cover - graceful fallback when deps missing
    logger.warning(
        "Optional core dependencies not found. Some features will be degraded.")
    logger.warning(
        "To enable full functionality run: pip install python-dotenv ipython plantuml")

    def load_dotenv(*args, **kwargs):
        logger.warning("python-dotenv not installed; .env will not be loaded.")

    def display(*args, **kwargs):
        return None

    def Markdown(text):
        return text

    class IPyImage:  # minimal placeholder used in notebooks
        def __init__(self, *args, **kwargs):
            pass

    class PlantUML:  # pragma: no cover - diagnostic only
        def __init__(self, url=None):
            logger.warning("plantuml not installed; rendering disabled.")

        def processes(self, *args, **kwargs):
            logger.warning("PlantUML rendering skipped (plantuml not installed).")


def load_environment():
    """Load environment variables from the nearest .env file.

    The search walks up from the current working directory until a directory
    containing either a ``.env`` file or a ``.git`` folder is found.  This
    mirrors the behaviour that existed in the original ``utils.py``.
    """
    path = os.getcwd()
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, '.env')) or os.path.exists(os.path.join(path, '.git')):
            project_root = path
            break
        path = os.path.dirname(path)
    else:
        project_root = os.getcwd()

    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logger.warning(".env file not found. API keys may not be loaded.")

__all__ = [
    'load_environment', 'load_dotenv', 'display', 'Markdown', 'IPyImage', 'PlantUML'
]
