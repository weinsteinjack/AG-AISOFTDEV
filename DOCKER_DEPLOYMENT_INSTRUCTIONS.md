# Docker Deployment Instructions - FastAPI Onboarding Application

## ✅ Completed Setup

All CI/CD artifacts have been successfully generated and are ready for deployment:

### 1. **requirements.txt** ✅
- **Location**: Project root (`requirements.txt`)
- **Status**: Generated and validated
- **Contains**: All necessary dependencies with proper version pinning
  - FastAPI, uvicorn, SQLAlchemy
  - Pydantic, pydantic-settings, email-validator
  - pytest, httpx for testing

### 2. **Dockerfile** ✅
- **Location**: Project root (`Dockerfile`)
- **Status**: Production-ready multi-stage build
- **Features**:
  - Multi-stage build (Builder + Runtime)
  - Security: Non-root user (appuser)
  - Optimization: Virtual environment isolation
  - Python 3.11-slim base image
  - Proper file permissions for SQLite database

### 3. **.dockerignore** ✅
- **Location**: Project root (`.dockerignore`)
- **Status**: Created with comprehensive exclusions
- **Excludes**: Development files, tests, notebooks, IDE configs

### 4. **GitHub Actions CI Workflow** ✅
- **Location**: `artifacts/.github/workflows/ci.yml`
- **Status**: Production-ready with best practices
- **Features**:
  - Automated testing on push/PR to main
  - Pip dependency caching
  - Coverage reporting with pytest-cov
  - Concurrency control

---

## 📋 Next Steps: Installing Docker

To build and run your Docker container, you need to install Docker Desktop:

### Windows Installation:

1. **Download Docker Desktop**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download the Windows installer

2. **Install Docker Desktop**
   - Run the installer
   - Follow the installation wizard
   - **Important**: Enable WSL 2 integration if prompted

3. **Restart Your Computer**
   - Docker requires a restart to complete installation

4. **Verify Installation**
   ```powershell
   docker --version
   docker run hello-world
   ```

### Alternative: Use GitHub Actions for Docker Builds

If you don't want to install Docker locally, you can use GitHub Actions to build and test your Docker image automatically. The CI workflow can be extended to include Docker build steps.

---

## 🚀 Building the Docker Image (After Installing Docker)

Once Docker is installed, run these commands from the project root:

### Build the Image
```powershell
docker build -t onboarding-tool .
```

Expected output:
- Layer-by-layer build process
- Final image tagged as `onboarding-tool:latest`
- Build time: ~2-5 minutes (first build)

### Verify the Build
```powershell
docker images | Select-String "onboarding-tool"
```

You should see your image listed with size ~150-200 MB (thanks to multi-stage build optimization).

---

## 🏃 Running the Container

### Basic Run Command
```powershell
docker run -p 8000:8000 onboarding-tool
```

### With Environment Variables (Recommended)
```powershell
docker run -p 8000:8000 --env-file .env onboarding-tool
```

**Note**: Create a `.env` file if you need to pass API keys or environment variables.

### Verify the Application
Open your browser and navigate to:
- **API Documentation**: http://localhost:8000/docs
- **Application**: http://localhost:8000

---

## 🔍 Testing the Deployment

### 1. Check Container Status
```powershell
docker ps
```

### 2. View Container Logs
```powershell
docker logs <container-id>
```

### 3. Test the API
```powershell
# Using PowerShell
Invoke-WebRequest -Uri http://localhost:8000/docs
```

### 4. Stop the Container
```powershell
docker stop <container-id>
```

---

## 📊 Docker Image Details

Your generated Dockerfile creates an optimized image with:

### Stage 1: Builder
- Installs all Python dependencies
- Creates virtual environment in `/opt/venv`
- Uses pip with `--no-cache-dir` for smaller layers

### Stage 2: Runtime
- Minimal Python 3.11-slim base
- Copies only the virtual environment and app code
- Creates non-root user for security
- Sets proper permissions for SQLite database
- Exposes port 8000
- Runs uvicorn server

### Security Features
✅ Non-root user (appuser)
✅ Minimal base image (no build tools)
✅ No cache files or temporary data
✅ Proper file permissions

### Size Optimization
✅ Multi-stage build excludes build tools
✅ Virtual environment isolation
✅ .dockerignore excludes unnecessary files
✅ Expected final size: ~150-200 MB

---

## 🔧 Troubleshooting

### Issue: "Cannot connect to Docker daemon"
**Solution**: Start Docker Desktop and wait for it to fully initialize

### Issue: "Port 8000 already in use"
**Solution**: 
```powershell
# Find process using port 8000
netstat -ano | findstr :8000
# Kill the process or use a different port
docker run -p 8080:8000 onboarding-tool
```

### Issue: "Application can't write to database"
**Solution**: The Dockerfile already includes proper permissions with `chown -R appuser:appuser /app`

### Issue: "Module not found errors"
**Solution**: Rebuild the image to ensure all dependencies are installed:
```powershell
docker build --no-cache -t onboarding-tool .
```

---

## 🌐 Deployment Options

### Option 1: Local Development
- Use `docker run` as shown above
- Good for testing and development

### Option 2: Cloud Deployment

#### Azure Container Apps
```powershell
az containerapp up --name onboarding-app --resource-group myRG --image onboarding-tool:latest
```

#### AWS App Runner
- Push image to ECR
- Create App Runner service from ECR image

#### Google Cloud Run
```powershell
gcloud run deploy onboarding-app --image onboarding-tool --platform managed
```

### Option 3: Docker Compose (Multi-Service)
Create a `docker-compose.yml` if you need additional services (database, cache, etc.)

---

## 📝 Project Structure Summary

```
AG-AISOFTDEV-1/
├── Dockerfile                    # ✅ Multi-stage production build
├── requirements.txt              # ✅ Python dependencies
├── .dockerignore                 # ✅ Build exclusions
├── artifacts/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   └── onboarding.db        # SQLite database
│   ├── tests/                   # Test suite
│   └── .github/
│       └── workflows/
│           └── ci.yml           # ✅ GitHub Actions CI pipeline
└── [other project files]
```

---

## 🎯 Summary

**What You've Accomplished:**
✅ Generated production-ready requirements.txt
✅ Created optimized multi-stage Dockerfile
✅ Set up .dockerignore for clean builds
✅ Configured GitHub Actions CI workflow
✅ All files follow industry best practices

**What's Next:**
1. Install Docker Desktop (5-10 minutes)
2. Build the Docker image (2-5 minutes)
3. Run and test the container (1 minute)
4. Deploy to your preferred platform (optional)

**Total Setup Time**: ~15-20 minutes after Docker installation

---

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Docker Guide** (in this project): `Supporting Materials/Docker_Guide.md`
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **GitHub Actions Documentation**: https://docs.github.com/en/actions

---

**Generated on**: October 31, 2025
**Project**: AI-Driven Software Engineering - Employee Onboarding System
**Lab**: Day 4 - Lab 2: Generating a CI/CD Pipeline

