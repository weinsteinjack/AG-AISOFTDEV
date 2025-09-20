# GEMINI.md

This file provides guidance to Gemini (the AI assistant) when working with code in this repository.

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
pytest tests/test_utils.py

# Run tests with coverage
pytest --cov=utils tests/

# Run integration tests (that call external APIs)
pytest -m integration
```

### Development Workflow
```bash
# Start Jupyter for interactive development
jupyter notebook

# Run a specific notebook
jupyter nbconvert --to notebook --execute "Labs/Day_01_Planning_and_Requirements/D1_Lab1_AI_Powered_Requirements_User_Stories.ipynb"
```

## Architecture Overview

### Core Components

**`utils/` - Unified AI Interface**
The heart of this repository is the `utils` module, which provides a standardized interface for multiple AI providers (OpenAI, Anthropic, Hugging Face, Google Gemini). It includes:

- `llm.py`: Contains the core functions for interacting with LLMs, such as `get_completion`, `get_vision_completion`, etc.
- `providers/`: Contains the provider-specific implementations for interacting with each AI provider.
- `helpers.py`: Contains helper functions used throughout the `utils` module.
- `models.py`: Contains the data models used in the `utils` module.

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
- `tests/`: Contains all the tests for the `utils` module.
- `pytest` is the testing framework used.

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

When adding new AI provider integrations, follow the established pattern in the `utils/` module:

1. Create a new provider file in `utils/providers/`.
2. Implement the provider-specific logic in the new file, inheriting from the base provider.
3. Update the `llm.py` file to include the new provider.

### Testing Considerations

- Use `pytest.mark.integration` for tests that call external APIs.
- Mock external API calls in unit tests to avoid rate limits.

### Jupyter Notebook Development

- Notebooks expect activated virtual environment with installed dependencies.
- Use functions from the `utils` module for consistent AI provider interactions.
- Save artifacts using `utils.artifacts.save_artifact()` for proper path management.
- Display generated content using provided display helpers.

## API Key Requirements

Essential for full functionality:
- `OPENAI_API_KEY`: Required for most course exercises
- `HUGGINGFACE_API_KEY`: Optional for open-source model experiments
- `GOOGLE_API_KEY`: Optional for Gemini model usage
- `ANTHROPIC_API_KEY`: Optional for Claude model usage

Store these in a `.env` file in the repository root (never commit this file).
