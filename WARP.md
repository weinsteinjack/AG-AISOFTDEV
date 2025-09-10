# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is the AI-Driven Software Engineering Program repository - a comprehensive 10-day intensive course teaching modern software development with AI assistance. The repository contains course materials, labs, solutions, and a unified AI provider interface for interacting with multiple LLM providers.

## Essential Commands

### Environment Setup
```bash
# Initial setup (run once)
python -m venv venv

# Activate environment (run every session)
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_clean_llm_output.py
pytest tests/test_main_simple.py
pytest tests/test_recommended_models_table.py

# Run tests with coverage
pytest --cov=utils tests/

# Run database setup test
python test_db_setup.py

# Run integration tests (that call external APIs)
pytest -m integration

# Run slow tests (image generation/editing)
pytest -m slow
```

### Development Workflow
```bash
# Start Jupyter for interactive development
jupyter notebook

# Run a specific notebook
jupyter nbconvert --to notebook --execute "Labs/Day_01_Planning_and_Requirements/D1_Lab1_AI_Powered_Requirements_User_Stories.ipynb"

# Start FastAPI development server (if app directory exists)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Operations
```bash
# Build container
docker build -t ai-software-engineering .

# Run containerized application
docker run -p 8000:8000 --env-file .env ai-software-engineering
```

## Architecture Overview

### Core Components

**`utils.py` - Unified AI Interface**
The heart of this repository is the `utils.py` module, which provides a standardized interface for multiple AI providers (OpenAI, Anthropic, Hugging Face, Google Gemini). It includes:

- `RECOMMENDED_MODELS`: Comprehensive model configuration database with capabilities metadata
- `setup_llm_client()`: Provider-agnostic client initialization
- `get_completion()`: Text-only completions
- `get_vision_completion()`: Multimodal image + text processing
- `get_image_generation_completion()`: Text-to-image generation
- `get_image_edit_completion()`: Image editing capabilities
- `transcribe_audio()`: Speech-to-text processing
- `prompt_enhancer()`: Meta-prompt optimization system

### Educational Structure

**Labs Directory (`Labs/`)**
Organized by days (Day_01 through Days_9_and_10) with:
- Each day contains specific learning objectives and hands-on exercises
- Jupyter notebooks for interactive development
- README files with day-specific guidance
- Special projects (SP) for advanced practice

**Supporting Materials**
- Environment setup guides
- Docker deployment instructions  
- API key generation documentation
- React component viewing guide

**Testing Framework**
- `conftest.py`: Test fixtures with in-memory SQLite database setup
- Test markers for integration tests (`integration`) and slow tests (`slow`)
- Database model validation tests

### Key Patterns

**Environment Management**
- Uses `python-dotenv` for secure API key management
- `.env` file pattern for local configuration
- Project root detection via markers (`.git`, `artifacts`, `README.md`)

**Artifact Management**
- All generated content saved to `artifacts/` directory
- Path resolution with security controls to prevent directory traversal
- Automatic directory creation for nested structures

**Error Handling**
- Graceful degradation when optional dependencies are missing
- Provider-specific error handling with informative messages
- Fallback mechanisms for different API endpoint variations

## Development Guidelines

### Working with AI Providers

When adding new AI provider integrations, follow the established pattern in `utils.py`:

1. Add model configuration to `RECOMMENDED_MODELS` with capability flags
2. Implement provider-specific client setup in `setup_llm_client()`
3. Add provider handling to relevant completion functions
4. Update model filtering in `recommended_models_table()`

### Testing Considerations

- Use `pytest.mark.integration` for tests that call external APIs
- Use `pytest.mark.slow` for tests involving image generation (10-30 seconds)
- Test database models use in-memory SQLite with proper fixture isolation
- Mock external API calls in unit tests to avoid rate limits

### Jupyter Notebook Development

- Notebooks expect activated virtual environment with installed dependencies
- Use `utils.py` functions for consistent AI provider interactions
- Save artifacts using `save_artifact()` for proper path management
- Display generated content using provided display helpers

### Docker Considerations

- Multi-stage build optimizes for production deployment
- FastAPI application expects to run at `/app` with `PYTHONPATH=/app`
- Environment variables passed via `--env-file .env`
- Exposes port 8000 for web services

## Course-Specific Context

This repository teaches AI-assisted development across the full SDLC:
- **Week 1**: AI-powered planning, design, development, testing, and basic agents
- **Week 2**: Advanced RAG systems, multi-agent workflows, vision, and capstone projects

The `utils.py` module serves as the foundation for all AI interactions throughout the course, providing students with a consistent interface regardless of which provider or model they choose to use.

## API Key Requirements

Essential for full functionality:
- `OPENAI_API_KEY`: Required for most course exercises
- `HUGGINGFACE_API_KEY`: Optional for open-source model experiments  
- `GOOGLE_API_KEY`: Optional for Gemini model usage
- `ANTHROPIC_API_KEY`: Optional for Claude model usage

Store these in a `.env` file in the repository root (never commit this file).
