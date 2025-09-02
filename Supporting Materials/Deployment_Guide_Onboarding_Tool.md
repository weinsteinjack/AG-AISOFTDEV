# 🚀 Deployment Guide: New Hire Onboarding Tool

### 🎯 Goal

Guide students through the **final assembly and deployment** of the full-stack New Hire Onboarding Tool.
This consolidates all course artifacts into a single, runnable application packaged inside a **Docker container**.

> **Who this is for:**
> Students who have completed the labs and want a **step-by-step walkthrough** to stitch the frontend to the backend and deploy a full application.

---

## 📑 Table of Contents

1. [Application Architecture: A Deeper Look](#-1-application-architecture-a-deeper-look)
2. [Connecting Frontend Components to Backend Endpoints](#-2-connecting-frontend-components-to-backend-endpoints)
3. [Inventory of Required Artifacts](#-3-inventory-of-required-artifacts)
4. [Assembling the Application](#-4-assembling-the-application)
5. [Containerization: The Enhanced Dockerfile](#-5-containerization-the-enhanced-dockerfile)
6. [Deployment and Execution](#-6-deployment-and-execution)
7. [Troubleshooting Guide](#-7-troubleshooting-guide-for-beginners)
8. [Stitching Overview (Intuition First)](#-8-stitching-overview-intuition-first)
9. [Choose Your Deployment Pattern](#-9-choose-your-deployment-pattern)

   * 9A. [Pattern A: Single Image (FastAPI serves React)](#9a-pattern-a-single-image-fastapi-serves-react)
   * 9B. [Pattern B: Two Services with Nginx Proxy (Docker Compose)](#9b-pattern-b-two-services-with-nginx-proxy-docker-compose)
10. [Frontend Wiring: Buttons → Endpoints → UI](#-10-frontend-wiring-buttons--endpoints--ui)
11. [CRUD Forms: Map Fields to Models](#-11-crud-forms-map-fields-to-models)
12. [Production Polishing Checklist](#-12-production-polishing-checklist)
13. [Quick Decision Flow](#-13-quick-decision-flow)

---

## 🔹 1. Application Architecture: A Deeper Look

```
 ┌────────────┐      ┌─────────────┐       ┌───────────────┐
 │  Frontend  │◀────▶│   Backend   │◀────▶ │   Database     │
 │   (React)  │      │  (FastAPI)  │       │   (SQLite)     │
 └────────────┘      └─────────────┘       └───────────────┘
        ▲                     │
        │                     ▼
        └────────── RAG Agent & LangGraph ────────────┘
```

* **Backend (FastAPI):** Business logic, CRUD, DB ops, AI chat agent (`/chat`).
* **Frontend (React):** User experience & API calls.
* **Database (SQLite):** Persistent data (`onboarding.db`).
* **Deployment (Docker):** Portable, reproducible, single command run.

---

## 🔹 2. Connecting Frontend Components to Backend Endpoints

### 🔗 The API Data Contract

**Backend Contract (FastAPI + Pydantic):**

```python
from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    email: str
    role: str
```

**Frontend Implementation (React):**

```jsx
import { useState } from "react";

const [formData, setFormData] = useState({
  name: "",
  email: "",
  role: "New Hire"
});
```

✅ Keys must **match exactly** between frontend state and backend Pydantic model.

---

### 🧑‍💻 Example: Full CRUD in React

See **UserManagement.jsx** (Create, Read, Update, Delete) pattern in [Section 11](#-11-crud-forms-map-fields-to-models).
Includes form submission, editing, deletion, refresh, and error handling.

---

## 🔹 3. Inventory of Required Artifacts

| **Artifact**              | **Purpose**                                     | **Created In** | **AI-Generated?** |
| ------------------------- | ----------------------------------------------- | -------------- | ----------------- |
| `app/main.py`             | FastAPI backend (endpoints, DB logic, RAG chat) | Day 3, Day 6   | ✅                 |
| `requirements.txt`        | Python dependencies                             | Day 4          | ✅                 |
| `artifacts/onboarding.db` | SQLite database with schema + seed data         | Day 2          | ✅                 |
| `artifacts/day1_prd.md`   | Knowledge base for RAG                          | Day 1          | ✅                 |
| `frontend/*.jsx`          | React UI components                             | Day 8          | ✅                 |
| `Dockerfile`              | Container instructions                          | Day 4          | ✅                 |
| `.env`                    | Secrets (API keys, configs)                     | Setup          | ❌                 |

---

## 🔹 4. Assembling the Application

### 📂 Project Structure

```
onboarding-tool/
├── app/                 # Backend
│   └── main.py
├── artifacts/           # DB + RAG docs
│   ├── onboarding.db
│   └── day1_prd.md
├── frontend/            # React project
│   └── src/components/
├── tests/               # Unit tests
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .env
```

### ⚠️ Recommended `.dockerignore`

```
.git
**/__pycache__/
**/*.pyc
node_modules
frontend/node_modules
frontend/dist
.env
```

---

## 🔹 5. Containerization: The Enhanced Dockerfile

```dockerfile
# --- Stage 1: Build Frontend ---
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Build Backend + Final Image ---
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY --from=frontend-builder /app/frontend/dist ./app/static
COPY artifacts/ ./artifacts/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]
```

---

## 🔹 6. Deployment and Execution

### ⚡ Quickstart

```bash
# Build image
docker build -t onboarding-tool .

# Run container
docker run -p 8000:8000 --env-file .env onboarding-tool
```

### 🌐 Access

* **Frontend UI:** [http://localhost:8000](http://localhost:8000)
* **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔹 7. Troubleshooting Guide for Beginners

| ❌ **Issue**                   | 🛠️ **Cause**                    | ✅ **Fix**                                                         |
| ----------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| `COPY failed: file not found` | Built from wrong folder          | Run `docker build` from repo **root** with the `Dockerfile`.      |
| `Connection Refused`          | App crashed inside container     | `docker logs <container_id>` and fix the Python error.            |
| API calls fail (404)          | Route mismatch or wrong base URL | Verify `/api/*` routes exist and frontend calls match.            |
| `no such table: users`        | DB not copied or wrong path      | Verify `artifacts/` COPY & `sqlite:///./artifacts/onboarding.db`. |
| API Key error                 | `.env` not passed to container   | Use `--env-file .env` on `docker run`.                            |

---

## 🔹 8. Stitching Overview (Intuition First)

* **What “stitching” means:** Wiring your **React** UI events → **HTTP calls** → **FastAPI** endpoints, then rendering returned data.
* **Where “base URL” comes from:** Environment variable in React (e.g., `VITE_API_URL` or `REACT_APP_API_URL`).
* **Two painless ways to avoid CORS:**

  1. Serve React **from FastAPI** (same origin).
  2. Serve React with **Nginx** and **proxy** `/api/*` to FastAPI (still same origin from browser’s POV).

> **Best practice:** Prefix backend routes with `/api` and use **relative calls** in React in production.

---

## 🔹 9. Choose Your Deployment Pattern

### 9A. Pattern A: Single Image (FastAPI serves React)

**Fastest & simplest**. Everything runs on port 8000, no CORS.

**FastAPI SPA fallback** (ensure API routes are defined **before** the catch-all):

```python
# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# API routers mounted at /api/... (define these FIRST)
# app.include_router(users_router, prefix="/api/users")
# app.include_router(tasks_router, prefix="/api/tasks")
# ...

# Serve Vite assets
STATIC_DIR = "app/static"
app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

# SPA fallback for client-side routing
@app.get("/{full_path:path}")
async def spa(full_path: str):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
```

**React calls in production:**

```jsx
// Use relative path in production (same origin)
await fetch(`/api/users`);
```

---

### 9B. Pattern B: Two Services with Nginx Proxy (Docker Compose)

**More modular**. Nginx serves the SPA and proxies `/api/*` → FastAPI.

**Backend Dockerfile**:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY artifacts/ ./artifacts/
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

**Frontend Dockerfile**:

```dockerfile
FROM node:18 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx","-g","daemon off;"]
```

**Nginx config** (`frontend/nginx.conf`):

```nginx
server {
  listen 80;

  location / {
    root /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
  }

  location /api/ {
    proxy_pass http://backend:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
  }
}
```

**docker-compose.yml**:

```yaml
version: "3.8"
services:
  backend:
    build: .
    container_name: backend
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=sqlite:///./artifacts/onboarding.db

  frontend:
    build: ./frontend
    container_name: frontend
    ports: ["80:80"]
    depends_on: [backend]
```

**React env for prod** (`frontend/.env.production`):

```dotenv
# Vite
VITE_API_URL=/api
# CRA
REACT_APP_API_URL=/api
```

**Bring it up:**

```bash
docker compose up --build
# Visit http://localhost
```

> **No CORS** needed here: browser only talks to Nginx origin; Nginx talks to FastAPI.

---

## 🔹 10. Frontend Wiring: Buttons → Endpoints → UI

**Pattern:** Each button gets an `onClick` → calls a function → performs `fetch` → updates component state → UI re-renders.

```jsx
// src/components/ApiButtons.jsx
import React, { useState } from "react";

const API = import.meta?.env?.VITE_API_URL || process.env.REACT_APP_API_URL || "";

export default function ApiButtons() {
  const [data,setData] = useState(null);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState(null);

  const call = async (path) => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}${path}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  };

  return (
    <div>
      <h2>API Controls</h2>
      <button onClick={() => call("/api/users")}>Fetch Users</button>
      <button onClick={() => call("/api/tasks")}>Fetch Tasks</button>
      <button onClick={() => call("/api/project/info")}>Fetch Project</button>
      <hr />
      {loading && <p>Loading…</p>}
      {error && <p style={{color:"red"}}>{error}</p>}
      {data && <pre><code>{JSON.stringify(data, null, 2)}</code></pre>}
    </div>
  );
}
```

---

## 🔹 11. CRUD Forms: Map Fields to Models

* **Contract-first:** Pydantic keys ⇄ form `name` attributes.
* **State-driven UI:** `onChange` updates state, submit triggers POST/PUT, refresh list.

```jsx
// src/components/UserManagement.jsx
import React, { useEffect, useState } from "react";
const API = import.meta?.env?.VITE_API_URL || process.env.REACT_APP_API_URL || "";

const empty = { id:null, name:"", email:"", role:"New Hire" };

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState(empty);
  const [editing, setEditing] = useState(false);

  const load = async () => setUsers(await (await fetch(`${API}/api/users`)).json());
  useEffect(() => { load(); }, []);

  const onChange = e => setForm({ ...form, [e.target.name]: e.target.value });

  const onSubmit = async (e) => {
    e.preventDefault();
    const url = editing ? `${API}/api/users/${form.id}` : `${API}/api/users/`;
    const method = editing ? "PUT" : "POST";
    await fetch(url, { method, headers:{ "Content-Type":"application/json" }, body: JSON.stringify(form) });
    setEditing(false);
    setForm(empty);
    load();
  };

  const onEdit = u => { setEditing(true); setForm(u); };
  const onDelete = async id => { await fetch(`${API}/api/users/${id}`, { method:"DELETE" }); load(); };

  return (
    <section>
      <h2>User Management</h2>
      <form onSubmit={onSubmit}>
        <input name="name"  value={form.name}  onChange={onChange} placeholder="Name"  required />
        <input name="email" value={form.email} onChange={onChange} placeholder="Email" required />
        <input name="role"  value={form.role}  onChange={onChange} placeholder="Role"  required />
        <button type="submit">{editing ? "Update User" : "Create User"}</button>
      </form>

      <h3>Users</h3>
      <ul>
        {users.map(u => (
          <li key={u.id}>
            {u.name} ({u.email}) - {u.role}
            <button onClick={() => onEdit(u)}>Edit</button>
            <button onClick={() => onDelete(u.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </section>
  );
}
```

**Flow:**
Create → POST `/api/users/` → refresh list
Read → GET `/api/users` → render list
Update → PUT `/api/users/{id}` → refresh
Delete → DELETE `/api/users/{id}` → refresh

---

## 🔹 12. Production Polishing Checklist

* **API Prefix:** All backend endpoints under `/api/*`.
* **SPA Fallback:** Catch-all route serves `index.html`.
* **Env Config:**

  * Dev: `VITE_API_URL=http://localhost:8000`
  * Prod (Single Image or Proxy): `VITE_API_URL=/api` (use relative calls)
* **Health Check:** Add `/api/healthz` for load balancers / uptime.
* **Error UX:** Centralized `fetch` wrapper (timeouts, JSON errors, toasts).
* **DB URL (SQLite):** `sqlite:///./artifacts/onboarding.db` works inside container.
* **Nginx cache:** Long-lived cache for assets, SPA fallback for routes.
* **Logs:** Uvicorn access logs to stdout (Docker-friendly).
* **Security Headers:** Consider CSP, X-Frame-Options at Nginx in production.
* **Auth (if applicable):** Prefer httpOnly cookies; avoid localStorage tokens.

---

## 🔹 13. Quick Decision Flow

* Want **fewest moving parts** and **no CORS**? → **Pattern A (Single Image)**
* Want **edge proxy** + **clean separation** + **scale later**? → **Pattern B (Nginx + FastAPI via Compose)**

Both approaches are fully aligned with course constraints (local hosting, Dockerized, small footprint, no GPU).