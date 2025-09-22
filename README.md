# AI-Driven Software Engineering Program

Welcome to the Digital Ethos Academy AI-Driven Software Engineering Program. This 10-day intensive curriculum shows software engineers how to weave large language models (LLMs) into every phase of the software development lifecycle (SDLC). The repository is the single source of truth for:

* daily lab notebooks and reference solutions,
* a production-ready utilities package for working with multiple AI providers, and
* the supporting guides you will lean on while planning, building, testing, and shipping AI-enabled software.

Whether you are self-paced or teaching in a classroom, this README explains how the pieces fit together and how to get started quickly.

---

## 📦 Repository Layout

| Path | Description |
| --- | --- |
| `Labs/` | Student-facing Jupyter notebooks organised by day (`Day_01_...` through `Days_9_and_10_Capstone`). Each notebook scaffolds a hands-on lab aligned with the daily learning objectives. |
| `Solutions/` | Fully worked notebooks that mirror the lab structure. These are the reference implementations instructors can demo or students can review after attempting a lab. |
| `Supporting Materials/` | Long-form documentation that supplements the labs: setup instructions, Docker walkthroughs, artifact management, React viewing tips, and more. |
| `templates/` | Markdown templates for Product Requirements Documents (PRDs), Architectural Decision Records (ADRs), evaluation rubrics, and other reusable assets. |
| `slides/` | Presentation decks that support each day’s lectures. |
| `utils/` | The Python package that powers AI interactions across the curriculum. It wraps provider SDKs (OpenAI, Anthropic, Google, Hugging Face, etc.), enforces artifact safety rules, and exposes helpers used throughout the labs. |
| `tests/` | Fast unit tests for the utilities package (all safe to run offline). Integration and slow tests are annotated with pytest markers. |
| `async_tests/` | Dedicated asyncio test coverage that demonstrates parallel LLM calls and provider patching patterns. |
| `pyproject.toml`, `requirements.txt`, `pytest.ini` | Tooling configuration for formatters, dependency management, and test execution. |
| `Daily agenda.ipynb` | Instructor agenda covering the minute-by-minute plan for the two-week experience. |

> **Tip:** The repository intentionally starts without an `app/` or `artifacts/` directory. Those assets are produced inside the labs; you can always peek at the corresponding notebook in `Solutions/` if you want a completed reference.

---

## 🎯 Program Outcomes

By the end of the program learners will be able to:

1. Translate ambiguous stakeholder needs into AI-assisted product requirements and architectural plans.
2. Co-develop FastAPI backends, Streamlit dashboards, and React components with LLM coding assistants while keeping humans in the loop.
3. Implement Retrieval-Augmented Generation (RAG) systems, evaluation harnesses, and multi-agent workflows using the shared utilities.
4. Automate quality workflows with AI: synthetic tests, docstring generation, PR reviews, and CI/CD scaffolding.
5. Package and deploy AI-enabled applications with Docker while documenting operational decisions.

Daily goals, labs, and artifacts live in each `Labs/Day_*` README; skim those summaries before class to stay aligned with the teaching plan.

---

## ⚙️ Getting Started

### 1. Prerequisites

* **Experience:** 1–3 years of software development (Python familiarity recommended).
* **Accounts:** GitHub plus any model provider accounts you intend to use (OpenAI is strongly recommended; Hugging Face, Google, and Anthropic keys unlock optional exercises).
* **Software:** Python 3.11, Git, and a modern IDE (VS Code, PyCharm, etc.).
* **Hardware:** 8 GB RAM minimum, solid internet connection, and permission to install Python packages.

### 2. Clone the Repository

```bash
git clone https://github.com/Digital-Ethos-Academy/AG-AISOFTDEV.git
cd AG-AISOFTDEV
```

If you received a `.zip` during an instructor-led cohort, extract it and open the folder in your IDE instead of cloning.

### 3. Create and Activate a Virtual Environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

All notebooks and utilities use the shared dependency list in `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure API Keys

Create a `.env` file at the project root (alongside this README) and add the secrets you plan to use. Only the OpenAI key is strictly required for the core labs.

```env
# .env
OPENAI_API_KEY="sk-..."          # required for most labs
ANTHROPIC_API_KEY="..."          # optional
GOOGLE_API_KEY="..."             # optional for Gemini-based labs
HUGGINGFACE_API_KEY="hf_..."     # optional for open-source experiments
```

The utilities package automatically loads this file via `python-dotenv` when you call `utils.load_environment()` or any helper that depends on provider credentials.

### 6. Validate Your Environment

Run the quick checks below after installing dependencies. They confirm that Python can import the utilities package and that pytest is configured correctly.

```bash
pytest tests
pytest async_tests
```

> **Integration tests:** Files such as `tests/test_text_generation.py` contain `@pytest.mark.integration` markers and require real API keys plus network access. Skip them by default with `pytest -m "not integration"` unless you explicitly want to hit provider endpoints.

### 7. Open the Lab Notebooks

Start Jupyter Notebook or JupyterLab from the repository root:

```bash
jupyter notebook
```

Each lab notebook imports from the `utils` package. If you run a notebook from a subdirectory, set the working directory to the project root or add `sys.path` entries so imports resolve correctly.

---

## 🧰 Working with the `utils` Package

The `utils` package consolidates course-wide helpers. Key modules include:

* `utils.llm` – Synchronous and asynchronous text/vision completions plus prompt enhancement. Central entry points: `setup_llm_client()`, `get_completion()`, `async_get_completion()`, and `prompt_enhancer()`.
* `utils.image_gen` – Text-to-image and image-edit APIs with async counterparts for OpenAI, Google, and Hugging Face.
* `utils.audio` – Speech-to-text wrappers (and compatibility layers for legacy tuple-returning helpers).
* `utils.artifacts` – Safe file persistence with directory sandboxing. Used heavily in labs to store generated assets.
* `utils.settings` – Environment loading, Jupyter display shortcuts (`Markdown`, `IPyImage`, `PlantUML`), and global configuration helpers.
* `utils.rate_limit` & `utils.http` – Provider-aware throttling and shared HTTP session utilities for production scenarios.
* `utils.logging` – Structured loggers that standardise output across notebooks and scripts.

A minimal text completion flow looks like this:

```python
from utils import setup_llm_client, get_completion

client, model_name, provider = setup_llm_client(preferred_provider="openai")
response = get_completion(
    "Summarise the key risks from our onboarding PRD.",
    client=client,
    model_name=model_name,
    api_provider=provider,
    temperature=0.2,
)
print(response)
```

To parallelise prompts, pair `async_setup_llm_client()` with `async_get_completion()` inside an `asyncio.gather` call. The `async_tests/test_async_llm.py` file demonstrates this pattern in a test harness.

---

## 🧪 Testing and Quality Gates

* **Unit tests:** `pytest tests` exercises artifact helpers, logging, HTTP wrappers, and synchronous compatibility shims without leaving your machine.
* **Async coverage:** `pytest async_tests` verifies concurrency helpers.
* **Integration tests:** Opt-in suites marked with `@pytest.mark.integration` call live provider APIs (text, vision, audio, and image generation). Run them only after supplying valid keys: `pytest -m integration`.
* **Slow tests:** Image workflows that may take 10–30 seconds each use `@pytest.mark.slow`. Combine markers to run them selectively, e.g. `pytest -m "integration and slow"`.

Before shipping new utilities or course assets, run the fast suite locally and ensure any provider-specific behaviour is guarded by feature flags or mocks so students without credentials are not blocked.

---

## 📚 Supporting Documentation

The `Supporting Materials/` directory contains detailed guides that extend this README:

* **Environment Setup Guide** – Step-by-step environment bootstrap with troubleshooting tips for common installation issues.
* **Artifacts Guide** – Deep dive on `utils.artifacts`, directory overrides, and security guarantees.
* **Docker Guide** – Conceptual and hands-on introduction to containerising the FastAPI projects you assemble in the labs.
* **Deployment Guide (Onboarding Tool)** – Blueprint for stitching the Day 1–Day 7 artifacts into a full-stack onboarding assistant, including container choices and wiring diagrams.
* **React Components Viewing Guide** – Zero-build workflow for rendering JSX snippets generated during the front-end labs.
* **Productionizing Utils** – Environment variables, rate limiting, logging, and retry configuration for taking the helper library beyond notebooks.

Each document has been refreshed to match the utilities and lab artefacts in this repository. Start there whenever you need deeper context or run into an edge case during class.

---

## 🤝 Contributing & Classroom Usage

This repository is primarily used in a facilitated training environment. When proposing improvements:

1. Fork the repository or create a feature branch (if you have write access).
2. Install dependencies and run `pytest tests` plus any affected async or integration suites.
3. Submit a pull request describing the change and how it supports the curriculum.

For classroom delivery, instructors typically:

* Distribute this repository and walk through the Environment Setup Guide on Day 0.
* Use the `Daily agenda.ipynb` notebook and slide decks to anchor lectures.
* Pair live coding with the lab notebooks, leaning on the `Solutions/` directory for demos.
* Encourage students to save generated assets via `utils.artifacts.save_artifact()` so their work is reproducible between sessions.

---

## 📄 License & Support

* **License:** These materials are licensed for classroom use by Digital Ethos Academy cohorts. If you need clarification on redistribution rights, contact your program coordinator before sharing content outside your organisation.
* **Support:**
  * Environment or dependency issues – follow the troubleshooting section in the Environment Setup Guide.
  * Lab blockers – compare your progress with the matching notebook in `Solutions/`.
  * Utilities questions – review docstrings in `utils/` and the Productionizing Utils guide.

Ready to dive in? Launch the Day 1 notebooks from `Labs/Day_01_Planning_and_Requirements/` and begin crafting AI-assisted product requirements.
