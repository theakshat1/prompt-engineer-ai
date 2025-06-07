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

This section guides you through running the Prompt Engineer AI system.

**Prerequisites:**

*   **Python:** Ensure you have Python 3.7 or newer installed. You can check your Python version by opening a terminal or command prompt and typing:
    ```bash
    python --version
    ```
    If Python is not installed or not in your system's PATH, please download it from [python.org](https://www.python.org/) and install it. Make sure to check the option "Add Python to PATH" during installation on Windows.
*   **pip:** Python's package installer, `pip`, should be included with your Python installation.
*   **Dependencies:** You should have already installed the necessary Python packages by running `pip install -r requirements.txt` as mentioned in the "Installation" section. If you encounter errors in later steps, please ensure this step was completed successfully.
*   **API Keys:** Ensure your `.env` file is correctly set up with your `OPENAI_API_KEY` and `PINECONE_API_KEY` as described in "Installation."

**Steps to Run the System:**

**1. Set Up the Pinecone Index (Knowledge Base - Optional for basic operation but recommended for full functionality):**

The application can use a Pinecone vector database to retrieve relevant knowledge and enhance prompt generation. If your project includes a `pinecone-index-setup.py` script and you have data to populate your index:

*   **Purpose:** This step creates and populates a specialized database that the AI uses to find information relevant to your prompts, making the refined prompts more knowledgeable and context-aware.
*   **When to run:** You typically only need to run this setup script once, or whenever you need to update the knowledge base with new data. If the index is already set up and populated, you can skip this step.
*   **Command:**
    ```bash
    python pinecone-index-setup.py
    ```
*   **Note:** Ensure your Pinecone API key and environment details are correctly configured for this script (often via the `.env` file or directly in the script if it's designed that way). If you skip this, the AI will still refine prompts but without custom knowledge retrieval.

**2. Run the Backend Server:**

The backend server is a Flask application that handles the core prompt engineering logic.

*   **Command:** Open your terminal or command prompt, navigate to the project's root directory (where `optimized-prompt-engineer-agent.py` is located), and run:
    ```bash
    python optimized-prompt-engineer-agent.py
    ```
*   **Expected Output:** You should see messages indicating the Flask server is starting. Once running, you'll typically see something like:
    ```
     * Serving Flask app 'optimized-prompt-engineer-agent'
     * Debug mode: off  # (or on, depending on server config)
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
    Keep this terminal window open. The server needs to be running for the web interface to work. The address (e.g., `http://127.0.0.1:5000/`) is where the backend API is accessible.

**3. Access the Web Interface:**

The web interface is a static HTML page that you open in your browser.

*   **Action:** Navigate to the `src` directory within your project folder.
*   Open the `index.html` file (i.e., `src/index.html`) directly in your web browser. You can usually do this by:
    *   Double-clicking the `index.html` file.
    *   Right-clicking the file and choosing "Open with" and then selecting your preferred browser.
    *   Typing the file path directly into your browser's address bar (e.g., `file:///path/to/your/project/src/index.html`).
*   **Interaction:** The web page will load, and it will automatically try to communicate with the backend server you started in Step 2. You can now use the interface to enter your initial prompts and begin the refinement process.

**Troubleshooting Common Issues:**

*   **Server Not Starting / `ModuleNotFoundError`:**
    *   Ensure you have installed all dependencies: `pip install -r requirements.txt`.
    *   Make sure you are in the correct directory in your terminal when running `python optimized-prompt-engineer-agent.py`.
*   **Web Interface Not Working / Not Connecting to Backend:**
    *   Confirm the backend server is running (see terminal output from Step 2).
    *   Ensure the `API_BASE_URL` in `src/app.js` (currently `http://127.0.0.1:5000`) matches the address where your Flask server is running.
    *   Check your browser's developer console (usually F12) for any error messages (e.g., network errors, JavaScript errors).
    *   Ensure JavaScript is enabled in your browser. Try clearing your browser cache or using a different browser.
*   **Pinecone Errors:**
    *   Verify your `PINECONE_API_KEY` and environment settings in the `.env` file are correct.
    *   Ensure your Pinecone index exists and is configured as expected by `pinecone-index-setup.py` or the main agent.

## Credits

[@theakshat1](https://github.com/theakshat1) - Creator and Developer.

[@akshat122402](https://github.com/akshat122402) - Tester and constant support.

## 🚧 Under Development 🚧

This repository is currently under active development. Features and documentation may change frequently. Stay tuned for updates!
