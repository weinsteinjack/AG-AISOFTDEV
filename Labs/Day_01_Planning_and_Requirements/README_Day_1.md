# Day 1: Planning & Requirements

## Overview

  * **Synopsis:** Today marks the beginning of our journey into AI-Driven Software Engineering. We will focus on the most critical phase of any project: planning and requirements. You will learn how to transform a simple, vague idea into a set of structured, validated, and machine-readable artifacts that will form the foundation for our entire application.
  * **Core Question of the Day:** How can GenAI help us start projects with more clarity, speed, and alignment?
  * **Key Artifacts Produced:**
      * `artifacts/day1_user_stories.json`: A validated JSON file containing Agile user stories and acceptance criteria.
      * `artifacts/day1_prd.md`: A formal Product Requirements Document generated from our user stories.
      * `app/validation_models/prd_model.py`: A Pydantic model for programmatically validating our documentation.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Decompose a vague problem statement into structured features, user personas, and Agile user stories using an LLM.
  * Synthesize detailed user stories into a formal Product Requirements Document (PRD).
  * Generate a Pydantic model to programmatically define and validate documentation structure.

## Agenda & Labs Covered

  * **Lab 1: AI-Powered Requirements & User Stories** - Brainstorm features and personas, then generate structured user stories as a JSON artifact.
  * **Lab 2: Generating a Product Requirements Document (PRD)** - Synthesize the user stories into a formal PRD and create a validation model for it.
  * **Self-Paced Practice: Agile Planning for a Library App** - Apply the day's concepts to a new problem domain to reinforce your skills.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps to ensure a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the following extensions installed:
          * Jupyter (for running `.ipynb` notebooks)
          * GitHub Copilot
          * Gemini
  * **Environment Setup:**
      * Before you begin, ensure you have created and activated your Python virtual environment from the project's root directory.
        ```bash
        # Create the virtual environment (only needs to be done once)
        python3 -m venv .venv

        # Activate the virtual environment (do this every time you start a new terminal session)
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Make a copy of the `.env.example` file and rename it to `.env`. Open the new `.env` file and populate it with your secret API keys for OpenAI and other services. The `utils.py` script will automatically load these keys.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: From Unstructured Ideas to Structured Data:** The most critical skill today is turning a simple sentence (`problem_statement`) into a validated JSON file. This is the cornerstone of AI-assisted engineering because machine-readable data enables automation. Once requirements are in a structured format like JSON, they can be fed into other tools to generate project plans, code, and tests.
  * **Concept 2: Documentation as Code:** In Lab 2, you don't just write a document; you generate a Pydantic model (`prd_model.py`) to validate that document's structure. This is a powerful professional practice. It means your documentation's format can be version-controlled with Git and automatically checked in a CI/CD pipeline, ensuring consistency and quality at scale.
  * **"Why it Matters":** Getting the requirements right is the single most important factor in a project's success. Using AI to structure, clarify, and formalize requirements at the very beginning prevents costly mistakes and rework later in the development cycle.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: AI-Powered Requirements & User Stories

1.  **Open the Notebook:** `labs/Day_01_Planning_and_Requirements/D1_Lab1_AI_Powered_Requirements_User_Stories.ipynb`
2.  **Goal:** To transform a one-sentence problem statement into a validated `day1_user_stories.json` file.
3.  **Execution:**
      * Run the cells in order from top to bottom.
      * Pay close attention to the **`TODO`** blocks. These are the sections where you will write your own prompts to the LLM.
      * Review the output of each cell. Notice how the quality and structure improve as your prompts become more specific.
4.  **Tool-Specific Guidance:**
      * **VS Code:**
          * **For Newcomers:** To run a cell in the Jupyter notebook, click on the cell and press `Shift+Enter` or click the "Play" button in the toolbar.
          * **Copilot/Gemini:** In Challenge 2, if your LLM doesn't produce perfect JSON, use the Chat panel in VS Code. Paste your prompt and ask the AI, "How can I improve this prompt to guarantee the output is only a valid JSON array?" This is a great way to learn prompt refinement.
      * **Git:**
          * **For Newcomers:** Git is a version control system that tracks changes to your files. A "commit" is like a snapshot of your project at a specific point in time.
          * After completing this lab, save your work by committing the artifact you created. Open a terminal in VS Code (`Terminal > New Terminal`) and run these commands:
            ```bash
            git add artifacts/day1_user_stories.json
            git commit -m "feat: Generate initial user stories from problem statement"
            ```

### Lab 2: Generating a Product Requirements Document (PRD)

1.  **Open the Notebook:** `labs/Day_01_Planning_and_Requirements/D1_Lab2_Generating_a_Product_Requirements_Document_PRD.ipynb`
2.  **Goal:** To use the `user_stories.json` artifact from Lab 1 to generate a complete, professional PRD and a Pydantic validation model.
3.  **Execution:**
      * This lab builds directly on the output from Lab 1. Ensure `artifacts/day1_user_stories.json` exists before you begin.
      * Complete the **`TODO`** blocks to write the prompts that will generate the PRD and the Pydantic model.
4.  **Tool-Specific Guidance:**
      * **VS Code:** After running the final cell, use the VS Code file explorer on the left to navigate to the `artifacts` directory and open `day1_prd.md`. VS Code has a built-in markdown preview that will render the file beautifully. Also, inspect the generated Python file in `app/validation_models/prd_model.py`.
      * **Git:** Commit the new artifacts you've created. This ensures a complete record of the day's work.
        ```bash
        git add artifacts/day1_prd.md app/validation_models/prd_model.py
        git commit -m "feat: Generate PRD and Pydantic validation model"
        ```

## 🔍 Troubleshooting & Common Issues

  * **Issue:** `ModuleNotFoundError: No module named 'utils'`
      * **Solution:** This usually means the project root directory isn't in your Python path. The setup cell in each notebook is designed to fix this. Make sure you've run the first code cell, and that your working directory is correct.
  * **Issue:** The LLM's JSON output in Lab 1 is invalid or contains extra text.
      * **Solution:** This is a classic challenge. The `clean_llm_output` function in `utils.py` is designed to strip common markdown fences. If it still fails, the best solution is to improve your prompt. Be extremely specific: "Your response must begin with `[` and end with `]`. Do not include any explanatory text before or after the JSON."

## ✅ Key Takeaways

  * You have learned to use an LLM as a partner in the requirements-gathering process.
  * You have mastered the critical skill of transforming unstructured text into structured, machine-readable data (JSON).
  * You have seen how using templates and providing context allows an LLM to generate high-quality, consistent documentation.
  * The artifacts you created today—the user stories and PRD—are not just exercises; they will be the direct input for Day 2, where we will begin to design our system's architecture.