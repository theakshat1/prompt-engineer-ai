# Prompt Engineer AI

An AI-powered prompt engineering tool using OpenAI API and Pinecone for knowledge retrieval.

## Introduction

A powerful tool designed to generate and optimize prompts for language models. Leveraging OpenAI's GPT-4, this agent constructs prompts based on provided task descriptions and examples, and refines them to enhance performance and accuracy.

## Features

- Refine and optimize user prompts
- Retrieve relevant context from a Pinecone index
- Generate clarifying questions to improve prompt quality
- Produce final, optimized prompts based on user input and retrieved knowledge

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/theakshat1/prompt-engineer-ai
   cd prompt-engineer-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your OpenAI API key, Pinecone API key.

## Usage

1. Set up the Pinecone index with prompt engineering knowledge
   ```
   python pinecone-index-setup.py
   ```

2. Run the prompt engineering agent:
   ```
   python optimized-prompt-engineer-agent.py
   ```

## Credits

[@theakshat1](https://github.com/theakshat1) - Creator and Developer.
[@akshat122402](https://github.com/akshat122402) - Tester and constant support.

## 🚧 Under Development 🚧

This repository is currently under active development. Features and documentation may change frequently. Stay tuned for updates!
