from openai import OpenAI
import os
from pinecone import Pinecone
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Constants
NO_QUESTION = "NO_QUESTION"
DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_INDEX_NAME = "prompt-engineering-knowledge"

class PromptEngineerAgent:
    def __init__(self, model: str = DEFAULT_MODEL, index_name: str = DEFAULT_INDEX_NAME):
        self.model = model
        self.index_name = index_name
        self.setup_logging()
        self.setup_apis()
        self.load_system_prompt()
        self.refine_system_prompt = """
You are an AI assistant specializing in prompt engineering. Your task is to refine a user's raw prompt to make it more effective.
Consider the following principles:
- Persona: Who should the AI act as?
- Task: What specific task should the AI perform?
- Format: What is the desired output format (e.g., list, JSON, paragraph)?
- Tone: What tone should the AI adopt (e.g., formal, friendly)?
- Exemplars: Are there good examples to guide the AI?
- Constraints: Any limitations or rules to follow?
- Clarity and Specificity: Is the language unambiguous?
Based on these principles and any retrieved context, improve the given user prompt.
"""
        self.clarify_question_system_prompt = """
You are an AI assistant specializing in prompt engineering. Your role is to generate clarifying questions to help improve a user's prompt based on the current prompt and conversation history.
Focus on:
- Target audience: Who is the prompt for?
- Desired output: What specific format or structure is expected?
- Level of detail: How much information is needed?
- Missing information: What critical details are absent from the current prompt and context?
Review the provided conversation history. If the existing information is sufficient and no further clarification is needed to create a high-quality final prompt, respond with the exact string "NO_QUESTION". Otherwise, ask one specific question to fill the most critical gap.
"""
        self.final_prompt_system_prompt = """
You are an AI assistant specializing in prompt engineering. Your task is to synthesize an initial prompt, subsequent refinements, user's answers to clarifying questions, and any retrieved context (all provided in the conversation history) into a single, final, highly effective prompt.
Ensure the final prompt is clear, specific, and incorporates all relevant information gathered throughout the conversation.
"""

    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def setup_apis(self) -> None:
        try:
            self.index = pc.Index(self.index_name)
            self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        except Exception as e:
            logging.error(f"Error setting up APIs: {e}")
            raise

    def load_system_prompt(self) -> None:
        self.system_prompt = self.get_env_variable("SYSTEM_PROMPT", """
        You are an expert prompt engineer. Your task is to refine and optimize user prompts
        to make them more effective and understandable for AI chatbots like ChatGPT and Claude.
        Analyze the given prompt, identify areas for improvement, and suggest refinements.
        Consider the following key prompt engineering principles:
        - Persona: Who should the AI act as? (e.g., a historian, a scientist, a pirate)
        - Task: What specific task should the AI perform? (e.g., summarize text, write code, generate ideas)
        - Format: What is the desired output format? (e.g., list, JSON, paragraph, markdown table)
        - Tone: What tone should the AI adopt? (e.g., formal, friendly, humorous, assertive)
        - Exemplars: Can you provide good examples of the desired output? (Showing is often better than telling)
        - Constraints: Are there any limitations or rules to follow? (e.g., word count, style guides, topics to avoid)
        - Clarity and Specificity: Is the language unambiguous and precise? (Avoid vague terms)
        If necessary, ask clarifying questions to better understand the user's intent.
        Use the provided context from the knowledge base to inform your decisions and suggestions.
        """)

    @staticmethod
    def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"Environment variable {var_name} is not set")
        return value

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        try:
            query_embedding = self.generate_embedding(query)
            
            if not all(isinstance(x, (int, float)) for x in query_embedding):
                raise ValueError("Invalid embedding generated")
            
            query_embedding_np = np.array(query_embedding)
            query_embedding_normalized = query_embedding_np / np.linalg.norm(query_embedding_np)

            results = self.index.query(vector=query_embedding_normalized.tolist(), top_k=top_k, include_metadata=True)
            
            logging.info(f"Context retrieval results: {results}")

            context = ""
            for match in results['matches']:
                context += match['metadata'].get('text', '') + "\n\n"

            return context.strip()
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            logging.info(f"Generating response with messages: {messages}")
            response = client.chat.completions.create(model=self.model, messages=messages)
            logging.info(f"OpenAI API response: {response}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {e}")
            # Re-raise the exception to be caught by the route's error handler
            raise Exception(f"OpenAI API Error: {e}")

    def refine_prompt(self, user_prompt: str) -> Tuple[str, List[Dict[str, str]]]:
        context = self.retrieve_context(user_prompt)
        logging.info(f"Retrieved context for refining prompt: {context}")
        
        # Initial conversation history for refinement
        conversation_history = [
            {"role": "system", "content": self.refine_system_prompt},
            {"role": "user", "content": f"Please refine and optimize the following prompt, using the provided context (if any): '{user_prompt}'\n\nContext: {context}"}
        ]

        refined_prompt = self.generate_response(conversation_history)
        logging.info(f"Refined prompt: {refined_prompt}")
        
        # Add assistant's refined prompt to history
        conversation_history.append({"role": "assistant", "content": refined_prompt})
        return refined_prompt, conversation_history

    def ask_clarifying_question(self, current_prompt: str, conversation_history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        # current_prompt is the latest version of the prompt (e.g., last refined_prompt)
        # conversation_history is the full chat history up to this point
        
        context_query = f"Current prompt: {current_prompt}\nConversation history: {conversation_history[-3:]}" # Use last few messages for context query
        retrieved_context = self.retrieve_context(context_query)
        logging.info(f"Retrieved context for clarifying question: {retrieved_context}")

        # Prepare messages for the LLM. Start with a copy of the existing history.
        messages_for_question_generation = list(conversation_history)

        # Ensure the correct system prompt is at the beginning for this specific task
        # If a system prompt already exists, replace it. Otherwise, insert it.
        if messages_for_question_generation and messages_for_question_generation[0]["role"] == "system":
            messages_for_question_generation[0] = {"role": "system", "content": self.clarify_question_system_prompt}
        else:
            messages_for_question_generation.insert(0, {"role": "system", "content": self.clarify_question_system_prompt})
        
        # Add an instruction to the user role, asking the LLM to generate a question or NO_QUESTION
        # This instruction should guide the LLM based on the history and context.
        user_instruction_content = (
            f"Based on the conversation history so far and the retrieved external context, "
            f"please ask one specific clarifying question to help improve the prompt. "
            f"The current prompt being worked on is implicitly the last assistant or user message that defines it. "
            f"If no further questions are needed and you have enough information to generate a final high-quality prompt, respond with the exact string '{NO_QUESTION}'.\n\n"
            f"Retrieved External Context:\n{retrieved_context}"
        )
        messages_for_question_generation.append({"role": "user", "content": user_instruction_content})
        
        question = self.generate_response(messages_for_question_generation)
        logging.info(f"Generated clarifying question: {question}")

        # Update conversation history with the assistant's question (or NO_QUESTION response)
        updated_history = list(conversation_history) # Make a copy before appending
        # We append the user instruction that *led* to the question, then the question itself.
        # The user_instruction_content was part of messages_for_question_generation.
        # The actual "question" is the assistant's response.
        updated_history.append({"role": "assistant", "content": question}) # The AI's response IS the question or NO_QUESTION
                                                                       # The messages_for_question_generation that was sent to LLM already has the prompt that generated this question.
                                                                       # So we only add the assistant's response to the original history.
        return question, updated_history


    def generate_final_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        latest_prompt_content = "N/A"
        # Try to find the last substantive prompt information from the history for context retrieval
        for i in range(len(conversation_history) -1, -1, -1):
            msg = conversation_history[i]
            if msg['role'] == 'assistant' and msg['content'] != NO_QUESTION:
                latest_prompt_content = msg['content']
                break
            if msg['role'] == 'user': # User's prompt or answer
                latest_prompt_content = msg['content']
                break
        
        context_for_retrieval = f"Full conversation history leading to final prompt generation. Last significant message: {latest_prompt_content}"
        retrieved_context = self.retrieve_context(context_for_retrieval)
        
        logging.info(f"Context for final prompt generation:\nHistory: {conversation_history}\nRetrieved Context: {retrieved_context}")

        final_messages = [{"role": "system", "content": self.final_prompt_system_prompt}]
        final_messages.extend(list(conversation_history)) 

        user_instruction_for_final_prompt = (
            f"Synthesize all the information from the preceding conversation history (initial prompt, refinements, questions, answers) "
            f"into a single, final, optimized prompt. Use the retrieved external information below to enhance it further.\n\n"
            f"Retrieved External Information:\n{retrieved_context}"
        )
        final_messages.append({"role": "user", "content": user_instruction_for_final_prompt})
        
        final_prompt = self.generate_response(final_messages)
        # The final prompt itself is not part of the history in this design, it's the end product.
        return final_prompt

# Flask routes

@app.route('/refine_initial_prompt', methods=['POST'])
def refine_initial_prompt_route():
    data = request.get_json()
    if not data or 'initial_prompt' not in data:
        return jsonify({"error": "initial_prompt is required"}), 400
    
    initial_prompt = data['initial_prompt']
    
    try:
        agent = PromptEngineerAgent()
        refined_prompt, conversation_history = agent.refine_prompt(initial_prompt)
        return jsonify({
            "refined_prompt": refined_prompt,
            "conversation_history": conversation_history
        })
    except Exception as e:
        logging.error(f"Error in /refine_initial_prompt: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_clarifying_question', methods=['POST'])
def get_clarifying_question_route():
    data = request.get_json()
    if not data or 'current_prompt' not in data or 'conversation_history' not in data:
        return jsonify({"error": "current_prompt and conversation_history are required"}), 400

    current_prompt = data['current_prompt']
    conversation_history = data['conversation_history']

    try:
        agent = PromptEngineerAgent()
        # The conversation_history from client should already include the current_prompt as the last assistant message
        # or user's initial prompt.
        question, updated_history = agent.ask_clarifying_question(current_prompt, conversation_history)
        
        return jsonify({
            "question": question,
            "updated_conversation_history": updated_history
        })
    except Exception as e:
        logging.error(f"Error in /get_clarifying_question: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit_answer_and_get_next', methods=['POST'])
def submit_answer_route():
    data = request.get_json()
    if not data or 'answer' not in data or 'current_prompt' not in data or 'conversation_history' not in data:
        return jsonify({"error": "answer, current_prompt, and conversation_history are required"}), 400

    answer = data['answer']
    current_prompt = data['current_prompt'] # This is the prompt *before* the user's current answer
    conversation_history = data['conversation_history'] # History up to and including the last question asked

    try:
        agent = PromptEngineerAgent()

        # Append user's answer to the conversation history
        # The history received from client should have the assistant's question as the last message.
        # So we append the user's answer to that.
        updated_history_with_answer = list(conversation_history)
        updated_history_with_answer.append({"role": "user", "content": answer})
        
        # Now, ask for the next clarifying question using this updated history
        # The 'current_prompt' here is still relevant for context retrieval or if the agent needs to refer to the base prompt being refined.
        next_question, history_after_next_q = agent.ask_clarifying_question(current_prompt, updated_history_with_answer)

        if next_question.strip().upper() == NO_QUESTION:
            # No more questions, generate final prompt
            # The history_after_next_q already includes the "NO_QUESTION" response from the assistant.
            final_prompt = agent.generate_final_prompt(history_after_next_q)
            return jsonify({
                "type": "final_prompt",
                "final_prompt": final_prompt,
                "updated_conversation_history": history_after_next_q # This history includes the user's answer and NO_QUESTION
            })
        else:
            # Another question was asked
            return jsonify({
                "type": "question",
                "question": next_question,
                "updated_conversation_history": history_after_next_q # This history includes user's answer and the new question
            })
            
    except Exception as e:
        logging.error(f"Error in /submit_answer_and_get_next: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Make sure to set HOST, PORT, and DEBUG settings as appropriate
    # For development, 0.0.0.0 makes it accessible from network, debug=True is helpful
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
