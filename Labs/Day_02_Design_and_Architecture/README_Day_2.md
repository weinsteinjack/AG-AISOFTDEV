# Day 2: Design & Architecture

## Overview

  * **Synopsis:** Today we transition from *what* we're building to *how* we'll build it. You will act as the lead architect, transforming the requirements and user stories from Day 1 into a concrete technical blueprint. This involves designing the data structure, making key technology decisions, and documenting those decisions for the future.
  * **Core Question of the Day:** How can GenAI serve as an architectural co-pilot?
  * **Key Artifacts Produced:**
      * `artifacts/schema.sql`: A complete SQL script to create our database tables.
      * `artifacts/seed_data.sql`: A SQL script with realistic sample data for testing.
      * `artifacts/onboarding.db`: A live, local SQLite database file built from the above scripts.
      * `artifacts/adr_001_database_choice.md`: A formal Architectural Decision Record (ADR) documenting our choice of database technology.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Use a Product Requirements Document (PRD) to generate a normalized SQL database schema.
  * Generate realistic, context-aware seed data for a database.
  * Write a Python script to programmatically create and seed a live SQLite database from SQL files.
  * Leverage an LLM as a research assistant to compare technical options and generate a formal Architectural Decision Record (ADR).

## Agenda & Labs Covered

  * **Lab 1: AI-Generated System Design & Database Seeding** - Generate a SQL schema and seed data from the PRD, then create a live database file.
  * **Lab 2: Documenting Key Decisions with ADRs** - Research a key technology choice and generate a formal ADR to document the decision.
  * **Self-Paced Practice: Designing a Real-Time Chat Application** - Apply the day's concepts to generate architectural diagrams for a new problem domain.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps for a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the Jupyter extension installed.
      * (Optional but Recommended) A database client that can connect to SQLite, such as the "SQLite" extension for VS Code, or a standalone tool like DB Browser for SQLite. This will help you visualize the database you create.
  * **Environment Setup:**
      * Ensure your Python virtual environment is activated.
        ```bash
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Your `.env` file should be correctly configured with your API keys from Day 1.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: From Requirements to Schema:** The ability to generate a structured database schema from an unstructured PRD is a massive accelerator. This automates a critical design step that is often manual and time-consuming, allowing you to create a data foundation for your application in minutes instead of hours.
  * **Concept 2: Documenting the "Why":** Great engineers don't just build things; they document *why* they built them a certain way. An Architectural Decision Record (ADR) is a simple but powerful tool for this. Using an AI to research options and then synthesize that research into a formal ADR makes this best practice fast and easy to follow.
  * **"Why it Matters":** A well-defined architecture and data model are the bedrock of a stable application. The skills you learn today ensure that the foundation of your project is solid, well-documented, and built quickly, setting you up for success in the development phase.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: AI-Generated System Design & Database Seeding

1.  **Open the Notebook:** `labs/Day_02_Design_and_Architecture/D2_Lab1_AI_Generated_System_Design_Database_Seeding.ipynb`
2.  **Goal:** To use the PRD from Day 1 to produce a live, seeded `onboarding.db` database file.
3.  **Execution:**
      * Run the cells in order. The first cells will load the PRD from Day 1.
      * Complete the **`TODO`** blocks to write prompts that generate the SQL schema and seed data.
      * The final challenge involves completing a Python function that executes your generated SQL files to create the live database.
4.  **Tool-Specific Guidance:**
      * **VS Code:**
          * **For Newcomers:** After running the final cell, a new file named `onboarding.db` will appear in your `artifacts` directory. If you have the "SQLite" extension for VS Code installed, you can click on this file, and VS Code will open a new tab allowing you to directly view the tables (`users`, `onboarding_tasks`) and the data inside them. This is a great way to visually confirm your work.
      * **Git:**
          * Commit all the artifacts you've created in this lab. The `.sql` files are just as important as application code because they define your data structure.
            ```bash
            git add artifacts/schema.sql artifacts/seed_data.sql artifacts/onboarding.db
            git commit -m "feat: Generate database schema, seed data, and db file"
            ```

### Lab 2: Documenting Key Decisions with ADRs

1.  **Open the Notebook:** `labs/Day_02_Design_and_Architecture/D2_Lab2_Documenting_Key_Decisions_with_ADRs.ipynb`
2.  **Goal:** To research a technical choice and produce a formal `adr_001_database_choice.md` file to document it.
3.  **Execution:**
      * Complete the **`TODO`** blocks to write prompts.
      * The first prompt generates a reusable template for ADRs.
      * The second prompt uses the LLM as a research assistant to compare database technologies.
      * The final prompt synthesizes the research into the template, creating the final ADR artifact.
4.  **Tool-Specific Guidance:**
      * **VS Code:** After running the final cell, open `artifacts/adr_001_database_choice.md`. Use VS Code's built-in markdown preview to see how the formatted document will look to other developers. This is a great example of "documentation as code."
      * **Git:** An ADR is a critical project document. Commit it to your repository so the whole team knows what was decided and why.
        ```bash
        git add artifacts/adr_001_database_choice.md templates/adr_template.md
        git commit -m "docs: Add ADR for database technology choice"
        ```

## 🔍 Troubleshooting & Common Issues

  * **Issue:** The generated SQL code has an error when the Python script tries to run it.
      * **Solution:** LLMs can sometimes make minor syntax errors in SQL. Read the error message in the notebook output carefully. You can either manually fix the `.sql` file or, better yet, go back to your prompt and add more specific instructions. For example: "Ensure the schema is compatible with SQLite."
  * **Issue:** The `onboarding.db` file is created but is empty.
      * **Solution:** This usually means the `conn.commit()` line was missed in the `create_database` function, or the `seed_data.sql` file was not executed correctly. Double-check your Python code in the final challenge of Lab 1.

## ✅ Key Takeaways

  * You have learned to translate high-level product requirements directly into a technical data model (a SQL schema).
  * You have practiced the powerful **Research -\> Synthesize -\> Format** workflow, using an LLM first to gather information and then to structure it into a formal document.
  * The tangible artifacts you created today—especially the `onboarding.db` file—are not just exercises. This database is the live foundation we will build our backend API on top of in Day 3.
