# Day 4: Testing & Quality Assurance

## Overview

  * **Synopsis:** An application without tests is an application that is broken by design. Today, we focus on building an automated safety net for the API we created on Day 3. You will learn how to automate quality assurance by generating a comprehensive test suite and then creating a complete Continuous Integration (CI) pipeline to run those tests automatically on every code change.
  * **Core Question of the Day:** How can GenAI automate testing and help us find bugs sooner?
  * **Key Artifacts Produced:**
      * `tests/conftest.py`: A configuration file containing a professional `pytest` fixture for isolated database testing.
      * `tests/test_main_with_fixture.py`: A complete test suite for our FastAPI application.
      * `requirements.txt`: A list of all Python dependencies for our project.
      * `Dockerfile`: A file that defines how to build a containerized version of our application.
      * `.github/workflows/ci.yml`: A GitHub Actions workflow to automate our entire testing process.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Generate `pytest` tests for both "happy path" and edge case scenarios.
  * Create advanced `pytest` fixtures to ensure tests run against a clean, isolated database.
  * Generate a `Dockerfile` to containerize the application for consistent deployments.
  * Generate a complete GitHub Actions workflow to automate the testing process on every code change.

## Agenda & Labs Covered

  * **Lab 1: Automated Testing & Quality Assurance** - Generate a complete `pytest` suite for our API, including happy paths, edge cases, and an isolated database fixture.
  * **Lab 2: Generating a CI/CD Pipeline** - Create all the configuration-as-code files needed for an automated CI pipeline.
  * **Self-Paced Practice: Test-Driven Debugging** - Use a test-first approach to identify and fix a bug in a sample piece of code.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps for a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the Jupyter and Python extensions.
      * **Docker Desktop:** Please ensure it is installed and **running** on your machine before starting Lab 2.
  * **Environment Setup:**
      * Ensure your Python virtual environment is activated.
        ```bash
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Your `.env` file should be correctly configured with your API keys.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: AI as a QA Partner:** In Lab 1, you'll use AI to brainstorm test cases. AI excels at thinking of "what if" scenarios (e.g., duplicate emails, non-existent IDs) that a human might forget, leading to more robust test coverage and finding bugs earlier.
  * **Concept 2: Test Isolation with Fixtures:** Creating an isolated, in-memory database for each test run is a professional best practice. It guarantees that your tests are repeatable, fast, and have no side effects on your development database. The AI-generated `conftest.py` automates this complex setup.
  * **Concept 3: Configuration as Code (CI/CD):** In Lab 2, you are not clicking buttons in a UI to build your pipeline; you are generating text files (`Dockerfile`, `ci.yml`). This is "Configuration as Code," a core DevOps principle that makes your build and test process version-controllable, auditable, and easily repeatable across any machine.
  * **"Why it Matters":** The skills you learn today allow you to build an automated safety net for your code. This gives you and your team the confidence to develop and deploy new features quickly, knowing that the automated pipeline will guard against regressions.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: Automated Testing & Quality Assurance

1.  **Open the Notebook:** `labs/Day_04_Testing_and_Quality_Assurance/D4_Lab1_Automated_Testing_Quality_Assurance.ipynb`
2.  **Goal:** To generate a complete and professional `pytest` suite for the API built on Day 3.
3.  **Execution:**
      * Run the notebook cells in order to generate happy path tests, edge case tests, and the advanced database fixture.
      * The notebook will save the generated test files into the `tests/` directory.
4.  **Tool-Specific Guidance:**
      * **VS Code:**
          * **For Newcomers:** VS Code has a fantastic built-in test runner.
            1.  Open the "Testing" tab from the sidebar (it looks like a beaker).
            2.  Click "Configure Python Tests" and select `pytest`.
            3.  Choose the `tests` directory.
            4.  VS Code will now discover all your generated tests. You can run them individually or all at once by clicking the "Play" buttons that appear in the test explorer and directly in your `test_*.py` files.
      * **Git:**
          * Your test suite is a critical part of your codebase. Commit it.
            ```bash
            git add tests/
            git commit -m "feat: Add pytest suite for user API"
            ```

### Lab 2: Generating a CI/CD Pipeline

1.  **Open the Notebook:** `labs/Day_04_Testing_and_Quality_Assurance/D4_Lab2_Generating_a_CI_CD_Pipeline.ipynb`
2.  **Goal:** To generate the `requirements.txt`, `Dockerfile`, and `ci.yml` files needed to automate testing.
3.  **Execution:**
      * Complete the `TODO` blocks to write prompts that will generate each of the configuration files.
      * The notebook will save these files to the correct locations in your project.
4.  **Tool-Specific Guidance:**
      * **Docker:**
          * **For Newcomers:** Docker allows you to package your application and all its dependencies into a "container." This container can run identically on any machine, which solves the classic "it works on my machine" problem.
          * After the `Dockerfile` is generated, you can build and run it locally. Open a terminal in VS Code and run:
            ```bash
            # Build the container image and tag it as 'onboarding-app'
            docker build -t onboarding-app .

            # Run the container, mapping port 8000 on your machine to port 8000 in the container
            docker run -p 8000:8000 onboarding-app
            ```
      * **Postman:**
          * Once your Docker container is running (from the command above), you can use Postman to test it, just like you did on Day 3. The URL (`http://127.0.0.1:8000/users/`) will be the same. This proves your application is running correctly inside the container.
      * **Git:**
          * The `ci.yml` file only works when it's in a GitHub repository. After generating the files, commit them and push them to your repo to see the magic happen.
            ```bash
            git add requirements.txt Dockerfile .github/workflows/ci.yml
            git commit -m "build: Add CI pipeline configuration"
            git push origin main
            ```
          * After pushing, go to your repository on GitHub and click the "Actions" tab to see your automated workflow run.

## 🔍 Troubleshooting & Common Issues

  * **Issue:** `pytest` command or VS Code test discovery fails.
      * **Solution:** Ensure your virtual environment is active. The `pytest` library should also be listed in your `requirements.txt` and installed via `pip install -r requirements.txt`.
  * **Issue:** `docker build` command fails.
      * **Solution:** First, make sure Docker Desktop is open and running. If it is, the error is likely a syntax mistake in the `Dockerfile` or a missing `requirements.txt` file. Check the terminal output for specific error messages.
  * **Issue:** My GitHub Action is failing.
      * **Solution:** Go to the "Actions" tab in your GitHub repository. Click on the failing workflow run to see a detailed log of each step. The error message there will tell you exactly which command failed (e.g., a test failed, or a dependency couldn't be installed).

## ✅ Key Takeaways

  * You have learned to use AI as a partner to generate robust test suites and brainstorm edge cases.
  * You have containerized an application using a `Dockerfile`, a fundamental skill for modern cloud-native development.
  * You have built a complete, automated CI pipeline using GitHub Actions, ensuring that your code is automatically tested for quality with every change.
  * With our application now built, documented, tested, and automated, we have completed the core SDLC for a traditional application. On Day 5, we will pivot and begin to make our application *intelligent* by building our first AI agents.