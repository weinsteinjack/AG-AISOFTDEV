# Docker Guide for the AI-Driven Software Engineering Program

Docker is the bridge between the code you generate in the labs and a reliable deployment environment. This guide introduces Docker concepts, explains how they relate to our FastAPI + React onboarding tool, and shows you how to containerize the project using artifacts produced throughout the course.

---

## 1. Why Docker Matters Here

By Day 3 you will have a working FastAPI backend, and by Day 8 you will pair it with a React interface. Without Docker, every teammate or CI server must manually install the right Python, Node.js, and system dependencies. Containers solve this by packaging:

* Your application code (`app/`, `frontend/`),
* The utilities package and Python dependencies (`requirements.txt`), and
* Runtime configuration (environment variables, artifact files).

Running `docker run onboarding-tool` should feel the same whether you are on your laptop, a teammate’s Windows machine, or a cloud service like Azure Container Apps.

---

## 2. Key Concepts Refresher

| Term | Analogy | In this Course |
| --- | --- | --- |
| **Dockerfile** | Recipe | Instructions for building an image that contains FastAPI, the artifacts directory, and the compiled React bundle. |
| **Image** | Baked cake | Output of `docker build`. Immutable snapshot ready for distribution. |
| **Container** | Slice of cake | Running instance of an image started with `docker run`. |
| **Registry** | Bakery display case | Optional storage for sharing images (Docker Hub, GitHub Container Registry). |

Unlike full virtual machines, containers share the host OS kernel. They start quickly and use less disk space—ideal for iteration during class.

---

## 3. Preparing Your Project Structure

Before writing a Dockerfile, confirm that the repository contains the following folders. They are created by exporting code from the lab notebooks (see the Deployment Guide for export tips).

```
AG-AISOFTDEV/
├── app/
│   ├── main.py
│   ├── agents/
│   └── models.py
├── artifacts/
│   ├── onboarding.db
│   └── docs/
├── frontend/
│   ├── package.json
│   └── src/components/
├── requirements.txt
└── Dockerfile  # you will add this
```

If any folder is missing, revisit the relevant day’s notebook or copy from the `Solutions/` directory.

---

## 4. Authoring the Dockerfile

Use a multi-stage build so Node dependencies and build tooling stay out of the final Python image:

```dockerfile
# Stage 1 – Frontend build
FROM node:18-alpine AS frontend-builder
WORKDIR /workspace/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2 – Backend runtime
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

Place the file at the repository root. Create `.dockerignore` with entries such as:

```
.env
**/__pycache__/
**/*.pyc
node_modules
frontend/dist
Labs/
Solutions/
```

This keeps secrets and development artifacts out of the image.

---

## 5. Building and Running Locally

```bash
docker build -t onboarding-tool .
docker run -p 8000:8000 --env-file .env onboarding-tool
```

* `-t onboarding-tool` names the image.
* `--env-file .env` injects your API keys into the container. The FastAPI app loads them through `utils.load_environment()`.

After the container starts, open `http://localhost:8000` to view the React UI served by FastAPI. API docs remain available at `http://localhost:8000/docs`.

---

## 6. Integrating with CI/CD

In Day 5 you use an LLM to draft a GitHub Actions workflow. Update that workflow so it builds the Docker image and runs tests:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: false  # set to true and configure registry secrets when ready to publish
          tags: onboarding-tool:ci
```

This mirrors what you should do manually before class demos: run tests, then build the container.

---

## 7. Deploying the Image

Pick the option that matches your environment:

1. **Single VM / Laptop:** Run `docker run` directly. Great for workshops or user testing sessions.
2. **Docker Compose:** Add a `docker-compose.yml` with `backend`, `frontend` (if running separately), and optional `db` services. Useful if you want a live reload experience during development.
3. **Managed Container Service:** Push the image to a registry and deploy using Render, Azure Web Apps, AWS App Runner, or Google Cloud Run. Make sure to mount persistent storage if the onboarding database should retain state between deployments.

Regardless of platform, confirm the following after deployment:

* Environment variables are present (`OPENAI_API_KEY`, etc.).
* Logs show successful startup (`Application startup complete` from Uvicorn).
* `/health` or `/docs` endpoints respond as expected.

---

## 8. Troubleshooting

| Issue | Diagnosis | Resolution |
| --- | --- | --- |
| `ModuleNotFoundError: utils` inside the container | `utils` package not copied or `PYTHONPATH` incorrect. | Ensure `COPY app/ ./app/` is present and that your modules use absolute imports (`from utils import ...`). |
| React assets 404 | Frontend build not copied into the container. | Confirm the multi-stage copy step is correct and `npm run build` succeeds locally. |
| Container cannot reach provider APIs | Missing API keys or outbound network restrictions. | Verify `.env` is passed to `docker run` and check firewall policies. |
| Image size too large | Node modules or build caches included. | Use multi-stage builds (as above) and prune dangling images with `docker image prune`. |
| Database reset on every deploy | SQLite file not mounted or persisted. | Copy the database from `artifacts/` during build or mount a volume with `-v /host/path:/app/artifacts`. |

---

## 9. Beyond the Classroom

The same Docker practices apply when you extend the onboarding tool after the course:

* Swap SQLite for PostgreSQL by pointing SQLAlchemy at an external database and linking the container to it.
* Enable HTTPS by placing Nginx or Traefik in front of the FastAPI container.
* Add observability by shipping logs to an ELK stack or Datadog; the structured logs produced by `utils.logging.get_logger()` make this straightforward.

Containers make these evolutions incremental—you can iterate safely without breaking the base application students build during the program.

Keep this guide handy during Days 4–7 when DevOps, testing, and deployment topics take centre stage.
