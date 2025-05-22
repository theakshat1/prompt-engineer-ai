# Prompt Engineer AI

An AI-powered prompt engineering tool using OpenAI API and Pinecone for knowledge retrieval, now with an interactive web interface.

## Introduction

A powerful tool designed to generate and optimize prompts for language models. Leveraging OpenAI's GPT models and a Pinecone vector database for knowledge retrieval, this agent helps construct and refine prompts to enhance their performance and accuracy through an easy-to-use web UI.

## Features

- Interactive web interface (`prompt-engineer-ui.html`) for a user-friendly experience.
- Backend API built with Flask to handle prompt engineering logic.
- Refines and optimizes user-provided initial prompts.
- Retrieves relevant context from a Pinecone vector database.
- Asks clarifying questions to gather more information and improve prompt quality.
- Generates a final, optimized prompt based on the conversation and retrieved knowledge.

## Architecture

The application now runs with a client-server architecture:
- **Frontend:** A static HTML page (`prompt-engineer-ui.html`) with JavaScript that provides the user interface. This UI runs in your browser.
- **Backend:** A Flask server (`optimized-prompt-engineer-agent.py`) that exposes API endpoints for prompt engineering tasks. The frontend communicates with these APIs locally.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/theakshat1/prompt-engineer-ai # Or your fork/repo URL
    cd prompt-engineer-ai
    ```

2.  **Install the required dependencies:**
    Make sure you have Python 3.7+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    -   Copy `.env.example` to a new file named `.env` in the project root:
        ```bash
        cp .env.example .env
        ```
    -   Edit the `.env` file and fill in your actual API keys:
        -   `OPENAI_API_KEY`: Your API key for OpenAI services.
        -   `PINECONE_API_KEY`: Your API key for Pinecone services.
    -   Optionally, you can configure:
        -   `PORT`: The port for the Flask backend server (defaults to 5000 if not set).
        -   `SYSTEM_PROMPT`: A general system prompt for the AI agent. The agent has a default if this is not set. Note that more specific prompts for refining, clarifying, and finalizing are hardcoded within the agent.

## Usage

1.  **Set up the Pinecone index (if not already done):**
    The application uses a Pinecone index for knowledge retrieval to enhance prompt generation. If you have a `pinecone-index-setup.py` script and relevant data, run it to populate your index.
    ```bash
    python pinecone-index-setup.py
    ```
    (Ensure your Pinecone environment and index details are correctly configured for this script, possibly using environment variables if the script supports them).

2.  **Run the Backend Server:**
    This command starts the Flask development server which listens for API requests from the frontend.
    ```bash
    python optimized-prompt-engineer-agent.py
    ```
    The server will typically run on `http://127.0.0.1:5000` (or the port specified by your `PORT` environment variable in the `.env` file). You should see output in your terminal indicating the server is running and on which address.

3.  **Access the Web Interface:**
    Open the `prompt-engineer-ui.html` file directly in your web browser (e.g., by double-clicking it, or using "File > Open" in your browser and navigating to the file).
    The web UI will then communicate with the local Flask server you started in the previous step. Follow the instructions on the page to start refining your prompts.

## Credits

[@theakshat1](https://github.com/theakshat1) - Creator and Developer.

[@akshat122402](https://github.com/akshat122402) - Tester and constant support.

## 🚧 Under Development 🚧

This repository is currently under active development. Features and documentation may change frequently. Stay tuned for updates!
