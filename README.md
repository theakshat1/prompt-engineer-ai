# Prompt Engineer AI

An AI-powered prompt engineering tool using OpenAI API and Pinecone for knowledge retrieval, now with an interactive web interface featuring an improved UI/UX and Redux-based state management.

## Introduction

A powerful tool designed to generate and optimize prompts for language models. Leveraging OpenAI's GPT models and a Pinecone vector database for knowledge retrieval, this agent helps construct and refine prompts to enhance their performance and accuracy through an easy-to-use web UI.

## Features

- Interactive web interface (`src/index.html`) for a user-friendly experience, with recent UI/UX enhancements.
- Frontend state management powered by Redux (via CDN).
- Backend API built with Flask to handle prompt engineering logic.
- Refines and optimizes user-provided initial prompts.
- Retrieves relevant context from a Pinecone vector database.
- Asks clarifying questions to gather more information and improve prompt quality.
- Generates a final, optimized prompt based on the conversation and retrieved knowledge.

## Architecture

The application runs with a client-server architecture:
- **Frontend:** Static HTML, CSS, and JavaScript files located in the `src` directory.
    - `src/index.html`: The main HTML file.
    - `src/style.css`: Contains all styles for the application.
    - `src/app.js`: Handles client-side logic, DOM manipulation, and API calls.
    - `src/store.js`: Defines the Redux store, actions, and reducers for state management. Redux library is loaded via CDN.
    - The UI runs in your browser and communicates with the local backend.
- **Backend:** A Flask server (`optimized-prompt-engineer-agent.py`) that exposes API endpoints for prompt engineering tasks.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/prompt-engineer-ai # Or your fork/repo URL
    cd prompt-engineer-ai
    ```

2.  **Install the required Python dependencies (for the backend):**
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
        -   `SYSTEM_PROMPT`: A general system prompt for the AI agent.

## Usage

1.  **Set up the Pinecone index (if not already done):**
    The application uses a Pinecone index for knowledge retrieval. If you have a `pinecone-index-setup.py` script and relevant data, run it to populate your index.
    ```bash
    python pinecone-index-setup.py
    ```

2.  **Run the Backend Server:**
    This command starts the Flask development server.
    ```bash
    python optimized-prompt-engineer-agent.py
    ```
    The server will typically run on `http://127.0.0.1:5000`.

3.  **Access the Web Interface:**
    Open the `src/index.html` file directly in your web browser (e.g., by double-clicking it, or using "File > Open").
    The web UI will communicate with the local Flask server.

## Credits

[@theakshat1](https://github.com/theakshat1) - Creator and Developer.

[@akshat122402](https://github.com/akshat122402) - Tester and constant support.

## 🚧 Under Development 🚧

This repository is currently under active development. Features and documentation may change frequently. Stay tuned for updates!
