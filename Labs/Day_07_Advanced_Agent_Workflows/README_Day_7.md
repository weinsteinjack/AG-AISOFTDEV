# Day 7: Advanced Agent Workflows

## Overview

  * **Synopsis:** Today, we elevate our skills from building functional agents to engineering production-ready agentic systems. We will focus on two critical aspects: making an agent's understanding of our requests more reliable using the Model Context Protocol (MCP), and making the agent's interaction with end-users more intuitive and effective using the Ask-Advise-Act (A2A) framework.
  * **Core Question of the Day:** How do we build agents that are not just functional, but also reliable, predictable, and user-friendly?
  * **Key Artifacts Produced:**
      * `sql_query_generator.py`: A complete, standalone Streamlit application that uses the A2A framework to guide users through generating complex SQL queries.
      * Refactored Python code snippets demonstrating the power of MCP and few-shot prompting.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Structure complex, multi-part prompts reliably using the Model Context Protocol (MCP).
  * Build a context-aware refactoring agent that uses MCP and LangChain adapters.
  * Implement the Ask-Advise-Act (A2A) framework to create guided, conversational user experiences.
  * Build a complete Streamlit application that uses the A2A pattern to help users generate complex SQL.

## Agenda & Labs Covered

  * **Lab 1: Advanced Agent Workflows with MCP** - Learn to structure complex prompts with MCP and build a more predictable code refactoring agent.
  * **Lab 2: Building Agent Frontends with the A2A Framework** - Create a guided, conversational UI for a SQL generation agent using Streamlit and the A2A pattern.
  * **Self-Paced Practice: Advanced Code Refactoring with MCP** - Apply the MCP agent to a new, more complex refactoring challenge using few-shot examples.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps for a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the Jupyter and Python extensions.
  * **Environment Setup:**
      * Ensure your Python virtual environment is activated. The labs for Day 7 will install new libraries (`model-context-protocol`, `langchain-mcp-adapters`, `streamlit`).
        ```bash
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Your `.env` file must be correctly configured with your `OPENAI_API_KEY`.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: The Predictability Problem & MCP:** As you give an agent more context (code to edit, specific instructions, examples to follow), a simple text prompt becomes "brittle" and unreliable. The **Model Context Protocol (MCP)** solves this by turning your prompt into a structured request, much like a well-defined API call. This clear separation of `<request>`, `<context>`, and `<instructions>` makes the agent's behavior more predictable and robust.
  * **Concept 2: Guided Interaction with A2A:** Users often don't know how to ask the "perfect question" to get what they want from an AI. The **Ask-Advise-Act framework** is a user experience (UX) pattern for agents that solves this. It creates a guided conversation:
    1.  **Ask:** The agent asks clarifying questions to understand the user's true intent.
    2.  **Advise:** The agent proposes a plan of action for the user to approve.
    3.  **Act:** The agent executes the approved plan to produce the final result.
        This builds user trust and leads to far better outcomes.
  * **"Why it Matters":** These patterns are what separate a cool prototype from a production-ready AI tool. MCP makes your agent reliable from a developer's perspective, while A2A makes it intuitive and effective from an end-user's perspective.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: Advanced Agent Workflows with MCP

1.  **Open the Notebook:** `labs/Day_07_Advanced_Agent_Workflows/D7_Lab1_Advanced_Agent_Workflows_with_MCP.ipynb`
2.  **Goal:** To understand and apply the Model Context Protocol to build a more reliable code refactoring agent.
3.  **Execution:**
      * Run the notebook cells in order.
      * Complete the **`TODO`** blocks to progress from manually formatting an MCP prompt to using the SDK and finally to integrating it seamlessly with LangChain.
4.  **Tool-Specific Guidance:**
      * **VS Code:** In Challenge 2, pay close attention to the output of the `.render()` method. This allows you to see how the Python SDK objects are translated into the clean, XML-style MCP format that the LLM receives. This visual confirmation is key to understanding the value of the protocol.
      * **Git:** The logic for your MCP-powered agent is a valuable asset. Commit the completed notebook.
        ```bash
        git add labs/Day_07_Advanced_Agent_Workflows/D7_Lab1_Advanced_Agent_Workflows_with_MCP.ipynb
        git commit -m "feat: Develop MCP-powered refactoring agent"
        ```

### Lab 2: Building Agent Frontends with the A2A Framework

1.  **Open the Notebook:** `labs/Day_07_Advanced_Agent_Workflows/D7_Lab2_Building_Agent_Frontends_with_the_A2A_Framework.ipynb`
2.  **Goal:** To create a standalone Streamlit application that implements the A2A pattern.
3.  **Execution:**
      * The main task is to complete the `ask_prompt`, `advise_prompt`, and `act_prompt` variables within the provided Streamlit application code.
      * Once complete, you will save the code as a `.py` file and run it from your terminal.
4.  **Tool-Specific Guidance:**
      * **VS Code:**
          * **For Newcomers:** This lab requires you to run a Python script outside the notebook.
            1.  In the VS Code File Explorer, create a new file named `sql_query_generator.py` in the project root.
            2.  Copy the entire completed code block from the final challenge of the notebook into this new file.
            3.  Open a new terminal in VS Code (`Terminal > New Terminal`).
            4.  Make sure your virtual environment is active.
            5.  Run the application with the command: `streamlit run sql_query_generator.py`
            6.  This will automatically open a new tab in your web browser displaying your live application.
      * **Git:** Your new Streamlit application is a significant piece of work. Commit it to your repository.
        ```bash
        git add sql_query_generator.py
        git commit -m "feat: Create A2A-powered SQL generator with Streamlit"
        ```

## 🔍 Troubleshooting & Common Issues

  * **Issue:** `ModuleNotFoundError: No module named 'streamlit'` or `model_context_protocol`.
      * **Solution:** The lab notebooks include an auto-installation script. If this fails, you can install the packages manually. Make sure your virtual environment is activated and run: `pip install streamlit model-context-protocol langchain-mcp-adapters`.
  * **Issue:** The Streamlit app shows an error or doesn't update correctly after a button click.
      * **Solution:** Streamlit reruns the entire script on each interaction. The most common source of errors is incorrect management of `st.session_state`. Ensure you are setting and reading the `step` variable correctly. Also, check the terminal where you ran `streamlit run` for detailed error tracebacks.

## ✅ Key Takeaways

  * You have learned to make agent prompts more robust and predictable by structuring them with the Model Context Protocol (MCP).
  * You have mastered the Ask-Advise-Act (A2A) framework, a powerful UX pattern for building guided, trustworthy AI applications.
  * You have built a complete, interactive web application using Streamlit to serve as a user-friendly frontend for an AI agent.
  * On Day 8, we will expand our agent's capabilities into a new dimension: **vision**. You will build agents that can see and understand images, and you will learn how to rigorously evaluate and secure your AI systems.