# 🚀 Deployment Guide: New Hire Onboarding Tool

This guide explains how to assemble and deploy the capstone project that threads through the first seven days of the program: the AI-assisted New Hire Onboarding Tool. Instead of introducing brand-new code, we will reference the notebooks you already completed and show how to stitch their outputs into a production-ready application.

> **Scope:** The repository does **not** ship a pre-built `app/` or `frontend/` directory. You will create those folders by exporting code from your lab notebooks (or by reusing the matching notebooks under `Solutions/`).

---

## 1. Architecture Recap

```
 ┌──────────────┐      ┌───────────────┐      ┌─────────────────┐
 │ React Client │◀────▶│ FastAPI Layer │◀────▶│ SQLite Knowledge │
 │ (Day 8 Labs) │      │ (Day 3 Labs)  │      │  & Workflow Data │
 └──────────────┘      └───────────────┘      └─────────────────┘
           ▲                    │                         ▲
           │                    ▼                         │
           └──────────── RAG + Agents (Days 5–7) ─────────┘
```

* **Backend:** `Solutions/Day_03_Development_and_Coding/D3_Lab1_AI_Driven_Backend_Development_SOLUTION.ipynb` contains the FastAPI scaffolding (models, CRUD routes, chat endpoint). Export relevant cells into `app/main.py`.
* **Database & RAG assets:** Day 2 and Day 5 labs produce schema diagrams, seed data, and RAG-ready Markdown files. Save them via `utils.artifacts.save_artifact()` so they can be copied into `artifacts/` during deployment.
* **Agent workflows:** Day 6–7 labs extend the backend with LangGraph/LangChain orchestrations. Integrate those modules under `app/agents.py` or a similar file.
* **Frontend:** Day 8 labs generate React components (e.g., onboarding dashboard, evaluation panels). Consolidate them under `frontend/src/components/`.

---

## 2. Export Checklist

Use this checklist to ensure you gather every artifact before building the container. The “Source Notebook” column points to the canonical solution for reference.

| Asset | Target Location | Source Notebook |
| --- | --- | --- |
| FastAPI app (`main.py`, routers, database helpers) | `app/` | Day 3 Lab 1 & Lab 2 |
| SQLAlchemy models & schema | `app/models.py` | Day 2 Lab 1 (design) + Day 3 Lab 1 (implementation) |
| Seed data (`onboarding.db`) | `artifacts/onboarding.db` | Day 2 Lab 2 |
| Knowledge base docs (`*.md`) | `artifacts/docs/` | Day 1 Lab 2 (PRD) + Day 5 Lab 1 (RAG prep) |
| Evaluation prompts / grading rubrics | `artifacts/evaluation/` | Day 4 Lab 1 |
| Agent orchestration (`agents.py`, `workflows.py`) | `app/agents/` | Day 5 Lab 2, Day 6, Day 7 |
| React screens (`*.jsx`) | `frontend/src/components/` | Day 8 Lab 1 & Lab 2 |
| Environment configuration | `.env` | Environment Setup Guide |

Export code by either:

1. Copying from your personal lab notebook into `.py`/`.jsx` files, or
2. Using **File → Download as → Python (.py)** in Jupyter, then cleaning the output (remove magic commands, ensure proper imports).

Remember to keep business logic (FastAPI, agents) in the backend and UI logic in the frontend. Shared configuration such as API base URLs should live in environment variables.

---

## 3. Wiring the Backend and Frontend

### Backend: Runbook

1. Create `app/main.py` and `app/dependencies.py` (or similar) in the repository root.
2. Add a `get_settings()` helper that loads API keys using `utils.load_environment()` so both the web app and the agent modules share credentials.
3. Mount routers for onboarding workflows, document retrieval, and the `/chat` endpoint that wraps the LangGraph agent from Day 7.
4. Point SQLAlchemy to the SQLite file that lives under `artifacts/onboarding.db` (import `create_engine` from `sqlalchemy`):

   ```python
   DATABASE_URL = "sqlite:///./artifacts/onboarding.db"
   engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
   ```

5. When deploying, bundle the `artifacts/` directory alongside the app so migrations and RAG resources load automatically.

### Frontend: Runbook

1. Create a Vite or Create React App project **or** follow the zero-build approach in the [React Components Viewing Guide](How_to_View_Your_React_Components_Locally.md).
2. Ensure the base URL for API calls points to the FastAPI server (e.g., `http://localhost:8000`). Use environment variables such as `VITE_API_URL` or `REACT_APP_API_URL`.
3. Map form fields to Pydantic models exactly—the Day 3 notebooks provide the canonical schema.
4. Validate flows manually: create a new hire, update status, trigger the agent-based onboarding checklist, and review the generated knowledge base links.

---

## 4. Containerization Strategy

Once the backend and frontend are wired together locally, package everything into a Docker image. Below is a multi-stage Dockerfile that mirrors what you generate in the Day 5 CI/CD lab. Adjust paths if your folder structure differs.

```dockerfile
# Stage 1 – Build the React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /workspace/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2 – Assemble the FastAPI backend
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY --from=frontend-builder /workspace/frontend/dist ./app/static
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Place this file at the project root next to `requirements.txt`. Create a `.dockerignore` containing `.env`, `**/__pycache__/`, and any local notebooks you do not want packaged.

### Build & Run

```bash
docker build -t onboarding-tool .
docker run -p 8000:8000 --env-file .env onboarding-tool
```

Visit `http://localhost:8000` to load the bundled React build. Interactive API docs remain available at `http://localhost:8000/docs`.

---

## 5. Deployment Options

| Pattern | When to Use | Notes |
| --- | --- | --- |
| **Single container (FastAPI serves React)** | Classroom demos, hackathon projects, initial pilot deployments. | Simplest setup. React build output is copied into `app/static` and served by FastAPI. |
| **Docker Compose (separate services)** | Teams that want independent scaling or a dedicated Node.js server. | Define `frontend` and `backend` services, optionally add an `nginx` reverse proxy for TLS/HTTP2. |
| **Cloud services (Render, Azure Web Apps, AWS App Runner)** | When you need managed infrastructure. | Use the same Docker image produced above. Configure environment variables and persistent storage for artifacts. |

For each option, ensure environment variables (API keys, vector store URLs, optional analytics keys) are injected securely. Avoid baking secrets into the container image.

---

## 6. Production Checklist

* ✅ **Logging:** Use `utils.logging.get_logger()` inside the FastAPI app so logs remain structured and consistent.
* ✅ **Rate limiting:** Configure `UTILS_RATE_LIMIT_QPS_*` environment variables if you call provider APIs from the backend. See the Productionizing Utils guide for details.
* ✅ **Health checks:** Add a `/health` endpoint that verifies database connectivity and optionally pings provider APIs with a lightweight request.
* ✅ **Error handling:** Wrap agent calls in `try`/`except` blocks and return helpful HTTP status codes to the frontend.
* ✅ **Security:** Enable CORS for the frontend domain only, validate user input, and avoid exposing API keys to the client.
* ✅ **CI/CD:** Extend the Day 5 GitHub Actions workflow to run `pytest` (fast suite) and `docker build` on every push.

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `COPY failed: file not found` during `docker build` | Running the build from the wrong directory or missing exported files. | Ensure `app/`, `frontend/`, and `artifacts/` exist at the repository root before building. |
| `sqlite3.OperationalError: no such table` | SQLite file not copied or migrations not applied. | Re-run the Day 2 schema notebook, export the database to `artifacts/onboarding.db`, and rebuild the image. |
| Frontend shows network errors (`CORS` or 404) | API base URL mismatch or missing CORS configuration. | Update environment variables, confirm FastAPI routes, and enable CORS middleware for the frontend origin. |
| Agent endpoint times out | Missing provider key or rate limit hit. | Verify `.env`, check logs for provider errors, and configure rate limiting/backoff in `utils.llm`. |
| React build missing styling | Generated component relies on Tailwind or CSS not bundled. | Import required CSS files into the React entry point before running `npm run build`. |

---

## 8. Putting It All Together

1. Finish the core labs and export code/artifacts into source files.
2. Test the FastAPI backend locally (`uvicorn app.main:app --reload`).
3. Wire up the React frontend and confirm all flows against the live backend.
4. Containerize using the multi-stage Dockerfile and run the image locally.
5. Deploy to your chosen environment and monitor logs for the first set of user journeys.

When each layer passes acceptance tests, you have successfully transformed the daily lab outputs into a cohesive onboarding assistant ready for stakeholders to trial.
