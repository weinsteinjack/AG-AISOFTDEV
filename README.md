# AI-Driven Software Engineering Program

A comprehensive 10-day intensive program designed to teach modern software development with AI assistance. This repository contains all course materials, labs, solutions, and supporting tools for building AI-enhanced applications.

## 🎯 Program Overview

This program focuses on integrating AI tools throughout the entire Software Development Lifecycle (SDLC), from requirements gathering to deployment. Students learn to leverage Large Language Models (LLMs) and AI coding assistants to accelerate development while maintaining code quality and best practices.

### Key Learning Outcomes

- Master AI-assisted software development workflows
- Build complete applications using AI pair programming
- Implement RAG (Retrieval-Augmented Generation) systems
- Create intelligent agents and multi-agent workflows
- Apply AI for testing, documentation, and code quality
- Deploy AI-enhanced applications to production

## 📋 Prerequisites

- **Experience**: 1-3 years of software development (Python preferred)
- **Skills**: Basic familiarity with Git, REST APIs, and SDLC
- **Hardware**: Computer with minimum 8GB RAM and stable internet
- **Accounts Required**:
  - [GitHub Account](https://github.com/)
  - [OpenAI API Key](https://platform.openai.com/account/api-keys)
  - (Optional) [Hugging Face API Key](https://huggingface.co/settings/tokens)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Digital-Ethos-Academy/AG-AISOFTDEV.git
cd AG-AISOFTDEV

# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here  # Optional
```

### 3. Verify Installation

```bash
# Run tests to verify setup
pytest tests/

# Start Jupyter for interactive labs
jupyter notebook
```

### 4. Asynchronous LLM Calls (Optional)

The `utils` package includes async counterparts for I/O helpers. You can run many
model calls concurrently using `asyncio.gather`:

```python
import asyncio
from utils import async_setup_llm_client, async_get_completion

async def main():
    client, model, provider = await async_setup_llm_client()
    prompts = [f"Hello {i}!" for i in range(5)]
    tasks = [async_get_completion(p, client, model, provider) for p in prompts]
    return await asyncio.gather(*tasks)

results = asyncio.run(main())
```

## 📚 Course Structure

### Week 1: AI-Assisted Software Development

**Focus**: Building complete web applications using AI throughout the SDLC

#### Day 1: AI-Powered Planning & Requirements

- **Content**: GenAI landscape, AI tools overview
- **Labs**: AI-powered requirements gathering, user story generation
- **Artifacts**: Product Requirements Document (PRD)

#### Day 2: AI-Driven Design & Architecture

- **Content**: AI for data modeling, architectural decisions
- **Labs**: System design, database schema generation, ADRs
- **Artifacts**: Database schema, Architectural Decision Records

#### Day 3: AI-Assisted Development & Documentation

- **Content**: AI pair programming, code generation best practices
- **Labs**: FastAPI backend development, code refactoring
- **Artifacts**: Complete backend application with documentation

#### Day 4: Testing & Quality Assurance

- **Content**: AI-powered testing strategies, quality metrics
- **Labs**: Test generation, code review automation
- **Artifacts**: Comprehensive test suite

#### Day 5: Advanced Agents & RAG

- **Content**: Agent frameworks, RAG fundamentals
- **Labs**: Building intelligent agents, document processing
- **Artifacts**: RAG-powered applications

### Week 2: Advanced AI Engineering

#### Day 6: Building RAG Systems

- **Content**: Vector databases, embedding strategies
- **Labs**: Enterprise RAG implementation
- **Artifacts**: Production-ready RAG system

#### Day 7: Advanced Agent Workflows

- **Content**: Multi-agent systems, workflow orchestration
- **Labs**: Complex agent coordination
- **Artifacts**: Multi-agent applications

#### Day 8: Vision & Evaluation

- **Content**: Computer vision, model evaluation
- **Labs**: Vision-powered applications, performance metrics
- **Artifacts**: Vision-enabled systems

#### Days 9-10: Capstone Project

- **Content**: End-to-end project development
- **Labs**: Individual capstone projects
- **Artifacts**: Complete production application

## 🛠 Key Technologies & Tools

### AI/ML Frameworks

- **Agent Frameworks**: AutoGen, CrewAI, LangChain, LangGraph
- **Model Providers**: OpenAI, Anthropic, Hugging Face, Google Gemini
- **RAG & Vector Stores**: FAISS, Sentence Transformers
- **Vision & Audio**: PIL, PyTorch, Transformers

### Development Stack

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Frontend**: React, Streamlit
- **Database**: SQLite (development), PostgreSQL (production)
- **Testing**: Pytest
- **Documentation**: Jupyter, Markdown

### DevOps & Deployment

- **Containerization**: Docker
- **Environment Management**: Python venv, python-dotenv
- **Version Control**: Git, GitHub
- **CI/CD**: GitHub Actions (configured in labs)

## 📁 Repository Structure

```text
├── Labs/                          # Day-by-day lab exercises
│   ├── Day_01_Planning_and_Requirements/
│   ├── Day_02_Design_and_Architecture/
│   ├── Day_03_Development_and_Coding/
│   ├── Day_04_Testing_and_Quality_Assurance/
│   ├── Day_05_Advanced_Agents_and_RAG/
│   ├── Day_06_Building_RAG_Systems/
│   ├── Day_07_Advanced_Agent_Workflows/
│   ├── Day_08_Vision_and_Evaluation/
│   └── Days_9_and_10_Capstone/
├── Solutions/                     # Complete lab solutions
├── Supporting Materials/          # Setup guides and documentation
├── app/                          # Sample applications
├── artifacts/                    # Generated project artifacts
├── templates/                    # PRD and ADR templates
├── tests/                        # Test suite
├── slides/                       # Course presentation materials
├── utils.py                      # Unified AI provider interface
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
└── Daily agenda.ipynb           # Course schedule and agenda
```

## 🔧 Core Utilities

### `utils.py` - Unified AI Interface

The `utils.py` module provides a standardized interface for multiple AI providers:

```python
from utils import text_generation, image_generation, vision_completion

# Text generation with multiple providers
response = text_generation(
    prompt="Generate a Python function to calculate fibonacci",
    model="gpt-4",
    provider="openai"
)

# Image generation
image_url = image_generation(
    prompt="A modern web application interface",
    provider="openai"
)

# Vision analysis
analysis = vision_completion(
    image_path="./screenshot.png",
    prompt="Describe the UI elements in this image"
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_text_generation.py
pytest tests/test_image_generation.py
pytest tests/test_vision_completion.py

# Run with coverage
pytest --cov=utils tests/
```

## 🐳 Docker Deployment

```bash
# Build the container
docker build -t ai-software-engineering .

# Run the application
docker run -p 8000:8000 --env-file .env ai-software-engineering
```

## 📖 Documentation

- **[Environment Setup Guide](Supporting%20Materials/Environment_Setup_Guide.md)**: Detailed setup instructions
- **[Docker Guide](Supporting%20Materials/Docker_Guide.md)**: Container deployment guide
- **[API Key Generation Guide](Supporting%20Materials/🔑%20API%20Key%20Generation%20Guide%20for%20Labs.pdf)**: Step-by-step API setup
- **[React Components Guide](Supporting%20Materials/How_to_View_Your_React_Components_Locally.md)**: Frontend development setup
- **[Productionizing Utils](Supporting%20Materials/Productionizing_Utils.md)**: Timeouts, retries, rate limiting, and logging
- **[Artifacts Guide](Supporting%20Materials/Artifacts_Guide.md)**: Artifact directory overrides and security

## 🕰️ Deprecations

Legacy helpers that returned ``(result, error_str)`` are deprecated.  Update
existing code to call the standard functions and catch
``ProviderOperationError`` instead.  Compatibility wrappers remain temporarily
for transition but will be removed in a future release.

## 🤝 Contributing

This is an educational repository. For improvements or bug fixes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🆘 Support

For technical support or questions:

- **Course Materials**: Check the `Supporting Materials/` directory
- **Lab Issues**: Review the `Solutions/` directory for reference implementations
- **Setup Problems**: Follow the detailed `Environment_Setup_Guide.md`
- **API Issues**: Consult the `API Key Generation Guide`

## 🌟 Acknowledgments

- **Digital Ethos Academy**: Course design and curriculum development
- **OpenAI**: GPT models and API access
- **Hugging Face**: Open-source model ecosystem
- **LangChain Community**: Agent framework development
- **FastAPI Team**: Modern Python web framework

---

**Ready to transform your software development with AI?** Start with the [Environment Setup Guide](Supporting%20Materials/Environment_Setup_Guide.md) and dive into Day 1 labs!
