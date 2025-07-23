# AI-Driven Software Engineering Program: Setup Guide

Welcome to the AI-Driven Software Engineering Program! This guide provides detailed instructions for setting up your development environment. Completing these steps is crucial for a smooth and effective learning experience throughout the course.

## 1. Prerequisites

Before you begin, please ensure you have the following:

* **Experience:**
  * 1–3 years of software development experience (Python is preferred).
  * Basic familiarity with Git, REST APIs, and the software development lifecycle.
* **Accounts & Software:**
  * A [GitHub](https://github.com/) account.
  * An [OpenAI API Key](https://platform.openai.com/account/api-keys). The labs are optimized for OpenAI models.
  * **(Optional)** A [Hugging Face API Key](https://huggingface.co/settings/tokens) for experimenting with open-source models.
  * Python (version 3.9 or higher) installed on your system.
  * [Git](https://git-scm.com/downloads) installed and configured.
  * A modern IDE, such as [Visual Studio Code](https://code.visualstudio.com/).
* **Recommended VS Code Extensions:**
  * [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-python.python-extension-pack)
  * [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
* **Hardware & Environment:**
  * A computer with a minimum of 8 GB of RAM.
  * A stable internet connection.

### 2. Environment Setup Instructions

Follow these steps precisely to prepare your local project folder:

1. **Download and Unzip Course Files:**
    * Download the course materials `.zip` file provided and extract it to a dedicated project folder on your computer.

2. **Create a Python Virtual Environment:**
    * Open your terminal and navigate to the root of your project folder.
    * Run the following command to create a virtual environment named `venv`. This isolates the project's dependencies.

        ```shell
        python -m venv venv
        ```

3. **Activate the Virtual Environment:**
    * Before installing packages or running notebooks, you must activate the environment in your terminal session.
    * **On macOS/Linux:**

        ```shell
        source venv/bin/activate
        ```

    * **On Windows:**

        ```shell
        .\venv\Scripts\activate
        ```

    * Your terminal prompt should now be prefixed with `(venv)`.

4. **Create Your `.env` File for API Keys:**
    * This file will securely store your secret API keys, so they are not hardcoded in the notebooks.
    * In the root of your project folder, find the example file named `.env.example`.
    * Duplicate this file and rename the copy to `.env`.
    * Open the new `.env` file in a text editor and paste your API keys:

        ```env
        OPENAI_API_KEY="sk-..."
        # HUGGINGFACE_API_KEY="hf_..." # Optional
        ```

5. **Install Dependencies:**
    * With your virtual environment still active, run the following command to install all the required Python libraries listed in the `requirements.txt` file:

        ```shell
        pip install -r requirements.txt
        ```

You are now fully set up and ready to begin Day 1!
