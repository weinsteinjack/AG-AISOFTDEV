"""Utilities package for AI-driven software engineering course.

This module re-exports the public API that historically lived in a single
``utils.py`` file.  The implementation is now split across a number of
submodules to make it easier to maintain and to add new providers.
"""
from .settings import load_environment, load_dotenv, display, Markdown, IPyImage, PlantUML
from .models import RECOMMENDED_MODELS, recommended_models_table
from .llm import (
    setup_llm_client,
    get_completion, get_completion_compat,
    get_vision_completion, get_vision_completion_compat,
    clean_llm_output,
    prompt_enhancer, prompt_enhancer_compat,
)
from .image_gen import (
    get_image_generation_completion, get_image_generation_completion_compat,
    get_image_edit_completion, get_image_edit_completion_compat,
)
from .audio import transcribe_audio, transcribe_audio_compat
from .artifacts import *  # noqa: F401,F403 re-export for backwards compatibility
from .errors import *  # noqa: F401,F403
from .logging import *  # noqa: F401,F403

__all__ = [
    'load_environment', 'load_dotenv', 'display', 'Markdown', 'IPyImage', 'PlantUML',
    'RECOMMENDED_MODELS', 'recommended_models_table',
    'setup_llm_client', 'get_completion', 'get_completion_compat',
    'get_vision_completion', 'get_vision_completion_compat',
    'get_image_generation_completion', 'get_image_generation_completion_compat',
    'get_image_edit_completion', 'get_image_edit_completion_compat',
    'transcribe_audio', 'transcribe_audio_compat',
    'clean_llm_output', 'prompt_enhancer', 'prompt_enhancer_compat',
]
