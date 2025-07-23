# Day 5: Advanced Agents & RAG

## Overview

  * **Synopsis:** Welcome to the second week\! Today marks a pivotal shift in the course. Having built, tested, and automated a traditional application, we now focus on making that application *intelligent*. You will learn the fundamental patterns for building AI agents—systems that can reason, plan, and use tools to accomplish complex goals.
  * **Core Question of the Day:** How do we move beyond simple prompts to build AI systems that can reason and act autonomously?
  * **Key Artifacts Produced:**
      * `app/day4_sp_fixed_shopping_cart.py`: A Python script containing a bug fix derived from a test-driven workflow.
      * `docker-compose.yml`: A sample Docker Compose file generated for an agent to interact with.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Build a tool-using agent with LangChain that can access external APIs (e.g., web search).
  * Create custom tools from Python functions to extend an agent's capabilities.
  * Implement the "Plan-and-Execute" pattern to solve problems in a structured, multi-step process.
  * Construct a multi-agent system where specialized agents collaborate to complete a task using AutoGen.

## Agenda & Labs Covered

  * **Lab 1: Tool-Using Agents** - Build your first agents using LangChain, giving them both pre-built and custom tools to solve problems.
  * **Lab 2: Plan-and-Execute & Multi-Agent Systems** - Explore advanced agent architectures, including two-step planners and collaborative, conversational agent teams.
  * **Self-Paced Practice: The AI Travel Agent** - Apply the tool-using agent pattern to a practical, multi-step research problem.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps for a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the Jupyter and Python extensions.
  * **Environment Setup:**
      * Ensure your Python virtual environment is activated. The labs for Day 5 will install new libraries (`langchain`, `pyautogen`) as needed.
        ```bash
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Your `.env` file must be correctly configured. Today's labs will use your `OPENAI_API_KEY` and `TAVILY_API_KEY`.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: Agents as Tool Users:** An LLM's knowledge is frozen in time. A true AI agent overcomes this by using tools—like a web search API or a custom Python function—to access live data and interact with external systems. This is the fundamental difference between a simple chatbot and a capable agent.
  * **Concept 2: Division of Labor for AI:** Complex problems are best solved by breaking them down. Multi-agent systems apply this principle to AI. Instead of one giant, generalist agent, you create a team of specialists (e.g., a Planner, a Coder, a Reviewer). This division of labor leads to higher-quality, more reliable results.
  * **Concept 3: The ReAct Framework (Reason-Act):** Agents work in a loop: they **Reason** about the problem, **Act** by choosing a tool, and **Observe** the tool's result. This `Thought -> Action -> Observation` cycle, which you can see when `verbose=True`, is the core "thinking" process that allows an agent to solve multi-step problems.
  * **"Why it Matters":** The skills you learn today are the building blocks for creating sophisticated AI assistants. Understanding how to give agents tools and orchestrate their collaboration is essential for building AI-native applications that can automate complex workflows.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: Tool-Using Agents

1.  **Open the Notebook:** `labs/Day_05_Advanced_Agents_and_RAG/D5_Lab1_Tool_Using_Agents.ipynb`
2.  **Goal:** To build a LangChain agent that can reason about and choose between multiple tools (web search, calculator) to answer a user's question.
3.  **Execution:**
      * Run the notebook cells in order. The first cell will install the required `langchain` and `tavily` libraries.
      * Complete the **`TODO`** blocks to define a custom tool, construct the agent, and invoke it with different types of questions.
4.  **Tool-Specific Guidance:**
      * **VS Code:**
          * **For Newcomers:** The "verbose=True" output from the `AgentExecutor` can be long. VS Code's Jupyter viewer allows you to scroll through these large outputs easily. Pay close attention to the agent's "Thought" process to understand its reasoning.
      * **Git:**
          * Commit the practice script you created. While it's a practice lab, it represents a new capability.
            ```bash
            git add app/day4_sp_fixed_shopping_cart.py
            git commit -m "feat: Add test-driven bug fix for shopping cart"
            ```

### Lab 2: Plan-and-Execute & Multi-Agent Systems

1.  **Open the Notebook:** `labs/Day_05_Advanced_Agents_and_RAG/D5_Lab2_Plan_and_Execute_Multi_Agent_Systems.ipynb`
2.  **Goal:** To create a collaborative team of specialized AI agents that can generate and review code through a conversational workflow.
3.  **Execution:**
      * Run the cells and complete the **`TODO`** blocks to first build a simple two-step "Plan-and-Execute" agent, and then a more complex four-agent team using the AutoGen framework.
4.  **Tool-Specific Guidance:**
      * **VS Code:** The AutoGen framework will create a `coding` directory in your project root to save and execute the code generated by the agent team. You can use the VS Code File Explorer to inspect the Python files created during the agent conversation.
      * **Git:** You can commit the `docker-compose.yml` file created in the self-paced lab.
        ```bash
        git add docker-compose.yml
        git commit -m "build: Add docker-compose file for agent practice"
        ```

## 🔍 Troubleshooting & Common Issues

  * **Issue:** `Tavily` or other LangChain tools give an authentication error.
      * **Solution:** This is almost always an API key issue. Make sure your `.env` file is in the project's root directory, is named correctly (not `.env.example`), and contains the correct, unexpired keys for services like `TAVILY_API_KEY`.
  * **Issue:** The AutoGen chat seems to run forever or gets stuck in a loop.
      * **Solution:** This can happen if the agents' instructions are unclear or the termination condition is never met. You can interrupt the Jupyter kernel to stop it. Try adjusting the agents' system messages to be more specific or tweaking the `is_termination_msg` function to be less strict. The `max_consecutive_auto_reply` parameter is also a safety valve to prevent infinite loops.

## ✅ Key Takeaways

  * You have learned how to give agents "superpowers" by connecting them to external tools, allowing them to access information and perform actions beyond their built-in capabilities.
  * You have implemented two advanced agent architectures: a sequential **Plan-and-Execute** workflow and a collaborative **Multi-Agent System**.
  * This is the foundation for building intelligent applications. On Day 6, we will combine these agentic concepts with a knowledge base to build a powerful Retrieval-Augmented Generation (RAG) system.
