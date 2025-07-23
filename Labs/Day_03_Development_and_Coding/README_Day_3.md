# Day 3: Development & Coding

## Overview

  * **Synopsis:** Today, we move from design to implementation. You will take the architectural blueprint and database created on Day 2 and bring them to life by generating the complete backend API for our Onboarding Tool. This is the core "building" phase where you will collaborate with an AI co-pilot to write, integrate, and improve functional code.
  * **Core Question of the Day:** How do we effectively collaborate with AI to write better code, faster?
  * **Key Artifacts Produced:**
      * `app/main.py`: A fully functional, database-connected FastAPI application.
      * `README.md`: A professional, auto-generated README file for the entire project, complete with API usage examples.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Generate a complete FastAPI application, including Pydantic and SQLAlchemy models, from a SQL schema.
  * Integrate AI-generated components to connect an API to a live SQLite database.
  * Use an LLM to refactor code to improve readability and adhere to software engineering principles.
  * Generate professional project documentation, including docstrings and a comprehensive README.md file.

## Agenda & Labs Covered

  * **Lab 1: AI-Driven Backend Development** - Generate all the necessary code for our API and integrate it with the live `onboarding.db` database from Day 2.
  * **Lab 2: Refactoring & Documentation** - Improve the quality of a code snippet through AI-assisted refactoring and generate high-quality project documentation.
  * **Self-Paced Practice: Building a URL Shortener API** - Apply the day's API generation skills to a new, classic software problem.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps for a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the Jupyter extension.
      * Postman (or another API client) for testing your new endpoints.
  * **Environment Setup:**
      * Ensure your Python virtual environment is activated.
        ```bash
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Your `.env` file should be correctly configured with your API keys from Day 1.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: AI as a Boilerplate Engine:** A significant portion of backend development involves writing repetitive boilerplate code (e.g., data models, basic CRUD endpoints). Lab 1 demonstrates how AI excels at this, generating the entire skeleton of our application in minutes. This frees you, the developer, to focus on the unique business logic that delivers real value.
  * **Concept 2: The Developer as Integrator:** Challenge 3 in Lab 1 is a manual integration step. This is a critical real-world workflow. The AI generates the *parts* (SQLAlchemy models, Pydantic schemas, endpoint functions), but the skilled developer's job is to *assemble* them correctly. This "human-in-the-loop" approach combines the speed of AI generation with the precision of expert oversight.
  * **Concept 3: Continuous Code Improvement:** Lab 2 shows that AI's role doesn't end after the first draft. Using AI to refactor code and generate documentation makes it easy to continuously improve the quality and maintainability of your codebase throughout its lifecycle.
  * **"Why it Matters":** The skills you learn today define the modern rhythm of AI-assisted development: **Generate -\> Integrate -\> Refactor**. Mastering this cycle will make you a faster, more efficient, and more effective software engineer.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: AI-Driven Backend Development

1.  **Open the Notebook:** `labs/Day_03_Development_and_Coding/D3_Lab1_AI_Driven_Backend_Development.ipynb`
2.  **Goal:** To generate and assemble a complete, database-connected FastAPI application in the `app/main.py` file.
3.  **Execution:**
      * Run the notebook cells in order to generate the initial in-memory API code and the necessary database integration code.
      * The final challenge is a hands-on integration task. You will combine the AI-generated code snippets into the final `app/main.py` file.
4.  **Tool-Specific Guidance:**
      * **VS Code:**
          * **For Newcomers:** Use the File Explorer in VS Code to create a new file: `app/main.py`. Having the Jupyter notebook open in one editor tab and `main.py` in another (split-screen view) is a highly effective way to work.
      * **Postman:**
          * **For Newcomers:** Postman is an application that helps you test APIs. Once your `app/main.py` is complete, you need to run it. Open a terminal in VS Code and run: `uvicorn app.main:app --reload`.
          * To test your `POST /users/` endpoint:
            1.  Open Postman and create a new request.
            2.  Change the method from `GET` to `POST`.
            3.  Enter the URL: `http://127.0.0.1:8000/users/`.
            4.  Click the "Body" tab, select "raw", and choose "JSON" from the dropdown.
            5.  Paste a user object into the text area, like: `{"email": "new.user@example.com", "name": "New User", "role": "New Hire"}`.
            6.  Click "Send". You should see the newly created user returned in the response panel below.
      * **Git:**
          * This is a major feature. Once your API is working, commit it.
            ```bash
            git add app/main.py
            git commit -m "feat: Implement database-connected user API endpoints"
            ```

### Lab 2: Refactoring & Documentation

1.  **Open the Notebook:** `labs/Day_03_Development_and_Coding/D3_Lab2_Refactoring_Documentation.ipynb`
2.  **Goal:** To use an LLM to improve a piece of poorly written code and then to generate a professional `README.md` for our project.
3.  **Execution:**
      * Run the cells and complete the `TODO` blocks to write prompts for refactoring code, adding docstrings, and finally, generating the project's main README file.
4.  **Tool-Specific Guidance:**
      * **VS Code:** Use the built-in markdown preview to view your newly generated `README.md`. This lets you see the formatted output instantly without leaving your editor.
      * **Git:** A project's README is its front door. Commit the new documentation.
        ```bash
        git add README.md
        git commit -m "docs: Generate comprehensive project README"
        ```

## 🔍 Troubleshooting & Common Issues

  * **Issue:** The `uvicorn app.main:app --reload` command fails.
      * **Solution:** There are two common causes: 1) Your terminal is not in the project's root directory (the same level as the `app` folder). 2) Your virtual environment is not activated. Run `source .venv/bin/activate` and ensure you are in the correct directory.
  * **Issue:** API requests in Postman fail with a "500 Internal Server Error".
      * **Solution:** This usually indicates a problem with the database connection or logic. Check the terminal where `uvicorn` is running for a detailed error message. A common cause is the `onboarding.db` file from Day 2 not being present in the `artifacts` directory.

## ✅ Key Takeaways

  * You have learned to generate functional backend code, including data models and API endpoints, from a technical specification (the SQL schema).
  * You have practiced the essential developer skill of integrating AI-generated components into a working application.
  * You have seen how AI can be used not just to write initial code, but also to improve its quality through refactoring and documentation.
  * With a functional and documented API now complete, we are ready for the next critical step in the SDLC: building a robust test suite to ensure its quality. That will be our focus on Day 4.
