# AI-Driven Software Engineering Program: Environment Setup Guide

This guide walks you through preparing a reliable workspace for the Digital Ethos Academy AI-Driven Software Engineering Program. Follow each section in order before attempting the Day 1 labs. The entire process takes roughly 20–30 minutes on a typical laptop.

---

## 1. Confirm Prerequisites

| Requirement | Details |
| --- | --- |
| **Operating System** | Windows 10/11, macOS 12+, or a modern Linux distribution. |
| **Python** | Version 3.11 (check with `python --version`). Install from [python.org](https://www.python.org/downloads/) if needed. |
| **Git** | Required for cloning and version control. Install from [git-scm.com](https://git-scm.com/downloads). |
| **IDE / Editor** | VS Code (recommended), PyCharm, or similar. Enable Python extensions (linting, notebooks, etc.). |
| **Hardware** | Minimum 8 GB RAM, 10 GB free disk space, stable internet connection. |
| **Accounts** | GitHub, OpenAI API key (core labs), plus optional Anthropic, Google Gemini, or Hugging Face keys for advanced exercises. |

> If you are learning in a classroom, ask your facilitator to confirm which providers your organization allows.

---

## 2. Obtain the Course Repository

### Option A – Git Clone (preferred)

```bash
git clone https://github.com/Digital-Ethos-Academy/AG-AISOFTDEV.git
cd AG-AISOFTDEV
```

### Option B – Instructor Zip File

1. Download the `.zip` distributed for your cohort.
2. Extract it to a convenient location (e.g., `~/Projects/AG-AISOFTDEV`).
3. Open the folder in your IDE or `cd` into it from the terminal.

> Regardless of the option you choose, the remainder of this guide assumes your terminal is pointed at the repository root (where `README.md` lives).

---

## 3. Create an Isolated Python Environment

Keeping project dependencies separate avoids conflicts with other work.

```bash
python -m venv venv
```

Activate the environment before installing packages:

```bash
# macOS / Linux
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)` indicating the environment is active. Repeat the activation command whenever you open a new terminal.

---

## 4. Install Dependencies

Upgrade `pip`, then install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file includes Jupyter, pytest, FastAPI, LangChain, CrewAI, and other frameworks used throughout the labs. Installation may take several minutes the first time.

---

## 5. Configure API Keys with `.env`

The utilities package loads API keys from a `.env` file using `python-dotenv`. Create the file manually in the project root if it does not exist:

```env
# .env - store secrets locally only
OPENAI_API_KEY="sk-..."          # Required for core labs
ANTHROPIC_API_KEY="..."          # Optional, unlocks Claude-based exercises
GOOGLE_API_KEY="..."             # Optional, used by Gemini demos
HUGGINGFACE_API_KEY="hf_..."     # Optional, for open-source model exploration
```

**Best practices:**

* Never commit `.env` to version control.
* Use separate keys for development vs. production if your organization requires audit logging.
* If you cannot obtain a provider key, you can still run planning and design labs; code execution labs will require keys to access LLM functionality.

---

## 6. Validate the Installation

Run the fast test suites to verify that imports, pytest configuration, and asyncio helpers all work on your machine.

```bash
pytest tests
pytest async_tests
```

*If you see `ModuleNotFoundError`, double-check that your virtual environment is activated.*

### Optional: Integration Tests

Some tests make real API calls to OpenAI, Google, or Hugging Face. They are marked with `@pytest.mark.integration` and `@pytest.mark.slow`. Run them only after you configure credentials and are comfortable with the associated usage costs:

```bash
pytest -m integration
pytest -m "integration and slow"
```

---

## 7. Launch Jupyter Notebooks

Start Jupyter from the repository root so notebooks can import the `utils` package without extra path manipulation:

```bash
jupyter notebook
```

Open `Labs/Day_01_Planning_and_Requirements/D1_Lab1_AI_Powered_Requirements_User_Stories.ipynb` to begin the first exercise. The `Solutions/` directory hosts completed notebooks for reference—save them for after you attempt each lab yourself.

---

## 8. Troubleshooting

| Issue | Cause | Resolution |
| --- | --- | --- |
| `pip install` fails with build errors | Missing system packages (e.g., Rust, C++ build tools) on Windows/Linux. | Install the recommended build tools for your OS, then retry. Many wheels ship as binaries so retries often succeed. |
| `ModuleNotFoundError: utils` inside a notebook | Notebook launched from a nested directory without project root on `sys.path`. | Start Jupyter from the repository root **or** add `import sys; sys.path.append("..")` in the first cell. |
| `openai.AuthenticationError` during labs | API key missing or incorrect. | Confirm the key in `.env`, restart the kernel, and run `from utils import load_environment; load_environment()` to reload variables. |
| `pytest` cannot find tests | Command executed outside the repository root. | `cd` into the project directory before running tests. |
| Slow package installation | Large ML libraries (PyTorch, Transformers) downloading for the first time. | Allow the download to finish. Future installs will reuse cached wheels. |

If you become stuck, capture the full error message and share it with your instructor or peers in the course Slack channel.

---

## 9. Next Steps

1. Review the main [README](../README.md) for a program overview and repository map.
2. Read the **Artifacts Guide** to understand how generated assets are stored.
3. Keep the **Docker Guide** handy—you will revisit it during the testing and deployment modules.
4. For front-end labs, bookmark the **React Components Viewing Guide** so you can preview JSX components without scaffolding a full build system.

With your environment ready, you can confidently jump into Day 1 and start applying AI to real software engineering workflows.
