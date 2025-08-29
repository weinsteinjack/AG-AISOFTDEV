# Welcome to the AI-Driven Software Engineering Program!

As you progress through this 10-day immersive course, you will be using Generative AI to accelerate every phase of the Software Development Lifecycle (SDLC). You will be building a real-world FastAPI application, and to ensure it can be deployed reliably and efficiently, we will be using a crucial technology called **Docker**.

This guide will introduce you to Docker, explain how it works, and detail how it fits into this course and modern software engineering practices.

-----

## A Detailed Guide to Docker

### 1. The Problem: "But it works on my machine!"

Imagine you’ve spent days coding a Python application. It relies on specific versions of libraries (like FastAPI and SQLAlchemy) and requires Python 3.11 (as specified in the course prerequisites). It works perfectly on your laptop.

You then try to run it on a production server or share it with a teammate. Suddenly, it breaks.

Why?

Perhaps the server is running Python 3.9. Maybe some required system libraries are missing, or the configuration is slightly different. This problem—**environment inconsistency**—has plagued software development for decades. It leads to the classic excuse: "But it works on my machine!"

### 2. What is Docker?

**Docker is a platform that solves this problem by allowing you to package your application and all of its dependencies (code, libraries, system tools, runtime—everything it needs to run) into a single, standardized unit called a container.**

#### The Shipping Container Analogy

Think about the global shipping industry. Before standardized shipping containers, moving goods (barrels, boxes, machinery) was chaotic and inefficient because every item was a different shape and size.

The invention of the standard shipping container revolutionized logistics. It doesn't matter what is inside the container; the container itself is a standard unit that can be moved seamlessly between trucks, trains, and ships.

Docker applies this concept to software:

- **The Goods Inside:** Your application code (the Python FastAPI app you'll build in Week 1).
- **The Dependencies:** The specific version of Python, the required libraries, and the operating system configuration.
- **The Container:** The standardized Docker container that holds everything together.
- **The Transport (Ships/Trucks):** Different computing environments (your laptop, a testing server, the cloud).

Because the container is standardized, the application will run the exact same way, regardless of where it is deployed.

### 3. How Docker Works: Containers vs. Virtual Machines

You might wonder how this is different from a Virtual Machine (VM), like VMware or VirtualBox.

- **Virtual Machines (VMs)** virtualize the *hardware*. Each VM runs a full, separate operating system (OS) on top of the host machine. This makes VMs heavy (gigabytes in size) and slow to boot.
- **Containers** virtualize the *operating system*. They sit on top of the **Docker Engine** (the software that manages containers), which runs on the host OS. All containers share the host machine's OS kernel.

This makes containers:

- **Lightweight:** They don't need a full OS, so they use much less space and memory.
- **Fast:** They can start up almost instantly.
- **Portable:** They ensure consistency across different environments.

Here is a visual representation of the difference:

![VMs vs Containers](https://media.licdn.com/dms/image/v2/D4D12AQGqjSqEGElq1w/article-inline_image-shrink_400_744/article-inline_image-shrink_400_744/0/1688608296456?e=2147483647&v=beta&t=FgkL4P6Yvn0U_UosEH4KylVBmDgRz6r3FXWaDTWap78)

### 4. Core Docker Artifacts

To work with Docker, you need to understand the relationship between three key artifacts. A helpful analogy is baking a cake:

#### A. The `Dockerfile` (The Recipe)

The `Dockerfile` is a simple text file that contains a set of instructions on how to build a Docker image. It is the recipe for your application's environment.

It defines things like:

- **`FROM`**: The starting point (e.g., "Start with a standard Python 3.11 environment").
- **`WORKDIR`**: Where to put the files inside the container.
- **`COPY`**: Copying your application code into the container.
- **`RUN`**: Commands to install dependencies (e.g., `pip install -r requirements.txt`).
- **`CMD`**: The command to execute when the container starts.

Here is a simplified example relevant to the FastAPI application you will build:

```dockerfile
# Start with a base Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency list
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# The command to run when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### B. The Docker Image (The Baked Cake)

The Docker Image is the result of executing the instructions in the Dockerfile (using the `docker build` command). It is a static, read-only, executable package.

It's the baked cake—complete and ready, but not yet being consumed. Images are portable and can be stored in a registry (like Docker Hub) and pulled down onto any machine running Docker.

#### C. The Docker Container (The Slice)

The Container is a running instance of an Image (started using the `docker run` command).

This is the "slice" of cake being served. It's where your application actually executes, isolated from the host system and other containers. You can run multiple containers from the same image.

### 5. Docker in the Software Development Lifecycle (SDLC)

Docker streamlines the entire SDLC by ensuring consistency from the developer's laptop all the way to production.

1. **Development:**
   - Ensures every developer uses the exact same environment, eliminating setup inconsistencies and speeding up onboarding.

2. **Testing & QA (Continuous Integration - CI):**
   - Automated tests can be run inside isolated containers. This guarantees that tests are consistent and reproducible because the environment is identical every time.

3. **Deployment (Continuous Deployment - CD):**
   - This is where Docker is essential. The process of moving code to production becomes the process of moving an image.
   - If tests pass in the CI phase, that *exact same image* is deployed to production. There are no surprises because the environment is identical to what was tested.

### 6. Docker in This Course

In this AI-Driven Software Engineering program, Docker is the bridge between the code you write and the infrastructure it runs on.

Here is the workflow you will experience in Week 1:

- **Days 1-3:** You focus on planning, designing, and coding your Python FastAPI application.
- **Day 4:** You focus on testing and QA, generating automated tests.
- **Day 5 (Deployment & Maintenance):** This is where Docker takes center stage.

#### The Key Lab: Day 5 - CI/CD Pipeline Lab

In the lab focused on CI/CD (labeled `D5_Lab1_CICD_Pipeline.ipynb` in the daily agenda), you will leverage Generative AI (LLMs) as your DevOps co-pilot to automate the creation of your deployment artifacts. This is a perfect example of the course theme: using AI to accelerate the SDLC.

1. **Generating `requirements.txt`:**
   - You will prompt an LLM to analyze your FastAPI source code (developed on Day 3) and automatically generate this list of dependencies.

2. **Generating the `Dockerfile`:**
   - You will prompt an LLM to create the `Dockerfile` needed to containerize your application. The lab specifically focuses on generating an optimized, "multi-stage" Dockerfile for better security and efficiency.

3. **Generating the CI/CD Workflow (`ci.yml`):**
   - You will prompt an LLM to generate a GitHub Actions workflow file. This file instructs the CI/CD system to automatically use your `Dockerfile` to build the Docker image and run your Day 4 tests inside the container.

By the end of Week 1, you won't just have code; you will have a containerized application managed within a professional CI/CD pipeline.

-----

## Summary

- **Docker solves** the "it works on my machine" problem by ensuring environment consistency.
- A **`Dockerfile`** is the recipe used to build an **Image** (the package).
- A **Container** is the standardized, lightweight running instance of an Image.
- In this course, you will use AI on **Day 5** to generate the `Dockerfile` and set up your CI/CD pipeline to automate the containerization process.
