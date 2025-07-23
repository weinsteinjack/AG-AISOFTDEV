# Day 6: Building RAG Systems & Conversational Agents

## Overview

  * **Synopsis:** Today, you will build one of the most powerful and common patterns for enterprise AI: a Retrieval-Augmented Generation (RAG) system. You will create a "knowledge base" from your project's documents and then build a team of AI agents that can reason about this private data to answer questions. Finally, you will learn how to expose this powerful system as a stateful, conversational API.
  * **Core Question of the Day:** How do we give AI agents access to our private data and enable them to have natural, multi-turn conversations?
  * **Key Artifacts Produced:**
      * `labs/Day_06_Building_RAG_Systems/D6_Lab1_Building_RAG_Systems.ipynb`: A notebook containing a complete, multi-agent RAG system built with LangGraph.
      * `chat_ui.py`: A Streamlit web application for interacting with your conversational agent.
      * An updated `app/main.py` with a new, stateful `/chat` endpoint.

## Learning Objectives

By the end of today's labs, you will be able to:

  * Build a complete RAG pipeline: Load, Split, Embed, and Store documents in a vector database.
  * Use LangGraph to orchestrate a team of specialized AI agents (Retriever, Grader, Router) to answer questions about a knowledge base.
  * Integrate a LangGraph agent into a FastAPI backend to create a chat API.
  * Implement conversational memory to create a stateful, multi-turn chat experience.

## Agenda & Labs Covered

  * **Lab 1: Building RAG Systems** - Construct a knowledge base and build a multi-agent RAG system from scratch using LangGraph.
  * **Lab 2: Creating a Conversational Multi-Agent System** - Integrate your RAG agent into the FastAPI application and build a Streamlit UI to chat with it.
  * **Self-Paced Practice: Building a Docker Compose Agent** - Create a practical developer assistant that can read and answer questions about local `docker-compose.yml` files.

## 🛠️ Prerequisites & Setup

This section provides the essential setup steps for a smooth lab experience.

  * **Software Requirements:**
      * VS Code with the Jupyter and Python extensions.
      * Postman for testing the new API endpoint.
  * **Environment Setup:**
      * Ensure your Python virtual environment is activated. The labs for Day 6 will install new libraries (`langgraph`, `faiss-cpu`, `pypdf`, `streamlit`) as needed.
        ```bash
        # On macOS/Linux:
        source .venv/bin/activate
        ```
  * **API Keys & Configuration:**
      * Your `.env` file must be correctly configured with your `OPENAI_API_KEY`.

## 💡 Core Concepts & Value Proposition

  * **Concept 1: Retrieval-Augmented Generation (RAG):** This is the core concept of the day. RAG allows an LLM to answer questions about information it was never trained on. By retrieving relevant documents from a private knowledge base and providing them as context in the prompt, you ground the model in facts, dramatically reducing hallucinations and enabling it to become an expert on your specific domain.
  * **Concept 2: LangGraph for Agent Orchestration:** While frameworks like AutoGen are great for conversational collaboration, LangGraph excels at creating directed, stateful workflows. You define your agent team as a graph of nodes and edges, giving you precise control over the flow of logic. This is ideal for building reliable, production-grade RAG systems.
  * **Concept 3: Stateful Conversations:** A simple Q\&A bot is stateless. A true conversational agent is stateful—it remembers the history of the conversation. In Lab 2, you'll learn the key mechanism for this: passing a `session_id`. This ID allows the backend to retrieve the correct conversation history, giving the agent the context it needs to understand follow-up questions.
  * **"Why it Matters":** Building a RAG-powered conversational agent is one of the most valuable skills in the current AI landscape. The ability to create a chatbot that can accurately answer questions about your company's internal documents, products, or codebase is a transformative business capability.

## 🚀 Step-by-Step Lab Instructions

### Lab 1: Building RAG Systems

1.  **Open the Notebook:** `labs/Day_06_Building_RAG_Systems/D6_Lab1_Building_RAG_Systems.ipynb`
2.  **Goal:** To build a sophisticated RAG agent that uses a team of specialists (Router, Researchers, Synthesizer) to answer questions about our project's documentation.
3.  **Execution:**
      * Run the notebook cells in order. The first cells will build the vector database from our project's artifacts.
      * Complete the **`TODO`** blocks to build the LangGraph workflow, progressing from a simple two-node graph to a complex multi-agent system with conditional routing.
4.  **Tool-Specific Guidance:**
      * **VS Code:** LangGraph outputs can be very verbose, showing the inputs and outputs of each node. Use the collapsible sections in the VS Code Jupyter viewer to navigate this output and trace the agent's reasoning process.
      * **Git:**
          * Commit the notebook that contains your powerful new RAG agent.
            ```bash
            git add labs/Day_06_Building_RAG_Systems/D6_Lab1_Building_RAG_Systems.ipynb
            git commit -m "feat: Develop multi-agent RAG system with LangGraph"
            ```

### Lab 2: Creating a Conversational Multi-Agent System

1.  **Open the Notebook:** `labs/Day_06_Building_RAG_Systems/D6_Lab2_Creating_a_Conversational_Multi_Agent_System.ipynb`
2.  **Goal:** To integrate the RAG agent into our FastAPI backend and create a web UI to chat with it.
3.  **Execution:**
      * This lab primarily involves editing your `.py` files.
      * Follow the instructions to add the new `/stateful_chat` endpoint to `app/main.py`.
      * Create a new `chat_ui.py` file and add the Streamlit code to build the user interface.
4.  **Tool-Specific Guidance:**
      * **VS Code:** A great workflow is to have three terminals open in VS Code: one to run the FastAPI server (`uvicorn app.main:app --reload`), one to run the Streamlit UI (`streamlit run chat_ui.py`), and one for Git commands.
      * **Postman:**
          * **For Newcomers:** Before building the UI, you can test your new stateful endpoint directly with Postman.
            1.  Send a `POST` request to `http://127.0.0.1:8000/stateful_chat` with a question.
            2.  The response will contain an `answer` and a `session_id`. Copy the `session_id`.
            3.  Create a new `POST` request, but this time, your JSON body should include both the new question and the `session_id` you received. This simulates a follow-up question.
      * **Git:** Commit the new UI and the updated backend code.
        ```bash
        git add app/main.py chat_ui.py
        git commit -m "feat: Add conversational RAG agent API and Streamlit UI"
        ```

## 🔍 Troubleshooting & Common Issues

  * **Issue:** `FAISS` or other vector store libraries fail to install.
      * **Solution:** These libraries sometimes have complex dependencies. Ensure you are using the correct version of Python (3.11) and that your `pip` is up to date. If issues persist, check the library's official documentation for OS-specific installation instructions.
  * **Issue:** Streamlit UI shows a "Connection error" when trying to reach the API.
      * **Solution:** This means the Streamlit app can't talk to the FastAPI backend. Make sure your FastAPI server is running in a separate terminal and that you see the "Application startup complete" message. Also, ensure the URL in your `requests.post()` call in `chat_ui.py` is correct (`http://127.0.0.1:8000/stateful_chat`).

## ✅ Key Takeaways

  * You have learned the end-to-end process of building a RAG system, from document ingestion to agentic reasoning.
  * You have mastered using LangGraph to create complex, controllable agent workflows with features like routing and quality gates.
  * You have successfully deployed a stateful conversational agent as an API and built a user-friendly web interface for it.
  * With these skills, you can now build intelligent chatbots and assistants that are experts in any domain for which you can provide documents. On Day 7, we'll explore even more advanced agentic workflows.
