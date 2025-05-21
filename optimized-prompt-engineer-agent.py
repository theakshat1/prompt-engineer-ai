from openai import OpenAI
import os
from pinecone import Pinecone
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

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
You are an AI assistant specializing in prompt engineering. Your role is to generate clarifying questions to help improve a user's prompt.
Focus on:
- Target audience: Who is the prompt for?
- Desired output: What specific format or structure is expected?
- Level of detail: How much information is needed?
- Missing information: What critical details are absent from the current prompt and context?
Ask questions that will help fill these gaps. If no further questions are needed based on the provided information, respond with the exact string "NO_QUESTION".
"""
        self.final_prompt_system_prompt = """
You are an AI assistant specializing in prompt engineering. Your task is to synthesize an original prompt, user's answers to clarifying questions, and any retrieved context into a single, final, highly effective prompt.
Ensure the final prompt is clear, specific, and incorporates all relevant information gathered.
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
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

    def refine_prompt(self, user_prompt: str) -> (str, List[Dict[str, str]]):
        context = self.retrieve_context(user_prompt)
        logging.info(f"Retrieved context for refining prompt: {context}")
        
        messages = [
            {"role": "system", "content": self.refine_system_prompt},
            {"role": "user", "content": f"Please refine and optimize the following prompt, using the provided context: '{user_prompt}'\n\nContext: {context}"}
        ]

        refined_prompt = self.generate_response(messages)
        logging.info(f"Refined prompt: {refined_prompt}")
        return refined_prompt, messages

    def get_feedback_on_refined_prompt(self, refined_prompt: str, refinement_messages: List[Dict[str, str]]) -> (str, List[Dict[str, str]]):
        print(f"Refined prompt: {refined_prompt}")
        user_feedback = input("Are you satisfied with this refined prompt? If not, please provide your feedback. Press Enter to accept: ").strip()

        if user_feedback:
            logging.info(f"User provided feedback on refined prompt: {user_feedback}")
            refinement_messages.append({"role": "assistant", "content": refined_prompt})
            refinement_messages.append({"role": "user", "content": f"I have some feedback on that prompt: {user_feedback}"})
            
            refined_prompt = self.generate_response(refinement_messages)
            logging.info(f"Re-refined prompt after feedback: {refined_prompt}")
        else:
            logging.info("User accepted the refined prompt.")
        
        return refined_prompt, refinement_messages

    def ask_clarifying_question(self, current_interaction_summary: str, previous_messages: List[Dict[str, str]]):
        # current_interaction_summary is still useful for context retrieval and logging
        retrieved_context = self.retrieve_context(current_interaction_summary) 
        logging.info(f"Retrieved context for clarifying question: {retrieved_context}")

        # Start with the history that led to the refined prompt
        messages = list(previous_messages) # Make a copy

        # Append the system prompt for asking clarifying questions
        # It's better to place the system prompt at the beginning if it's not already there,
        # or ensure it's the first message if previous_messages doesn't include one.
        # For simplicity, let's assume previous_messages might not have the *correct* system prompt for *this specific task*.
        # So, we'll prepend the specific system prompt for question generation.
        
        messages_for_question_generation = [{"role": "system", "content": self.clarify_question_system_prompt}]
        
        # Add the existing conversation that led to the current refined prompt.
        # This context should ideally end with the assistant's last refined prompt.
        messages_for_question_generation.extend(messages) 

        # Now add the user message that explicitly asks for clarifying questions
        user_instruction = (
            f"You are an expert prompt engineer. Review the conversation history provided above. "
            f"Based on this history, the interaction summary below, and the retrieved external information, "
            f"identify and list specific clarifying questions to ask the user. These questions should aim to gather "
            f"information that would further optimize the user's prompt, focusing on aspects like target audience, "
            f"desired output format, constraints, examples, or any ambiguities. "
            f"If no further questions are needed, respond with the exact string '{NO_QUESTION}'.\n\n"
            f"Current Interaction Summary (for additional context and keyword extraction for retrieval):\n"
            f"{current_interaction_summary}\n\n"
            f"Retrieved External Information:\n"
            f"{retrieved_context}"
        )
        messages_for_question_generation.append({"role": "user", "content": user_instruction})

        question = self.generate_response(messages_for_question_generation)
        logging.info(f"Generated clarifying question: {question}")
        return question

    def generate_final_prompt(self, full_conversation_history: List[Dict[str, str]], clarifying_questions_and_responses_summary: str) -> str:
        # Determine the latest refined prompt from the history for context retrieval
        latest_refined_prompt = "N/A"
        if full_conversation_history:
            # Iterate backwards to find the last assistant message, which should be the latest refinement
            for i in range(len(full_conversation_history) - 1, -1, -1):
                if full_conversation_history[i]['role'] == 'assistant':
                    latest_refined_prompt = full_conversation_history[i]['content']
                    break
            if latest_refined_prompt == "N/A" and full_conversation_history[0]['role'] == 'user': # Fallback to initial user prompt if no assistant message
                latest_refined_prompt = full_conversation_history[0]['content']


        context_for_retrieval = f"Latest refined prompt: {latest_refined_prompt}\nClarifying questions and responses: {clarifying_questions_and_responses_summary}"
        retrieved_context = self.retrieve_context(context_for_retrieval)
        
        logging.info(f"Context for final prompt generation:\nHistory: {full_conversation_history}\nQ&A Summary: {clarifying_questions_and_responses_summary}\nRetrieved Context: {retrieved_context}")

        final_messages = [{"role": "system", "content": self.final_prompt_system_prompt}]
        final_messages.extend(list(full_conversation_history)) # Use a copy of the history

        user_instruction_for_final_prompt = (
            f"Based on our entire conversation so far (including the initial prompt, my feedback, and your refinements, all detailed in the message history above), "
            f"and additionally considering the following summary of clarifying questions and my answers, please generate the final, "
            f"optimized prompt. Use the retrieved external information to enhance it further.\n\n"
            f"Summary of Clarifying Questions and Answers:\n{clarifying_questions_and_responses_summary}\n\n"
            f"Retrieved External Information:\n{retrieved_context}"
        )
        final_messages.append({"role": "user", "content": user_instruction_for_final_prompt})
        
        return self.generate_response(final_messages)

    @staticmethod
    def get_user_input(prompt: str) -> str:
        while True:
            user_input = input(prompt).strip()
            if user_input:
                return user_input
            print("Input cannot be empty. Please try again.")

    def get_clarifying_responses(self, user_prompt: str, refined_prompt: str, refinement_history_messages: List[Dict[str, str]]) -> (List[str], str):
        user_responses = []
        q_and_a_summary_parts = []
        max_questions = 3
        
        # Append the latest refined prompt by assistant to the history before asking questions
        # This ensures the LLM sees the prompt it's supposed to be asking questions about.
        current_history = list(refinement_history_messages)
        if not current_history or current_history[-1]['content'] != refined_prompt or current_history[-1]['role'] != 'assistant':
             current_history.append({"role": "assistant", "content": refined_prompt})


        for i in range(max_questions):
            current_interaction_summary = f"Original prompt: {user_prompt}\nLatest refined prompt: {refined_prompt}\nPrevious user responses to clarifying questions: {'; '.join(user_responses)}"
            logging.info(f"Context for clarifying question {i + 1}: {current_interaction_summary}")

            # Pass the potentially updated current_history (which includes the latest refined_prompt)
            question = self.ask_clarifying_question(current_interaction_summary, current_history)
            logging.info(f"Clarifying question {i + 1}: {question}")

            if question.strip().upper() == NO_QUESTION:
                logging.info("No further clarifying questions needed.")
                break

            response = self.get_user_input(f"Clarifying question: {question}\nYour response (or press Enter to skip): ")
            logging.info(f"User response to clarifying question {i + 1}: {response}")
            
            q_and_a_summary_parts.append(f"Q: {question}\nA: {response if response else 'Skipped'}")
            current_history.append({"role": "assistant", "content": question}) # Add AI question to history
            if response:
                user_responses.append(response)
                current_history.append({"role": "user", "content": response}) # Add user response to history
            else:
                # If user skips, we might still want to note that in history or just break.
                # For now, we break, and the summary already notes "Skipped".
                break
        
        return user_responses, "\n".join(q_and_a_summary_parts)

    def run(self) -> None:
        try:
            user_prompt = self.get_user_input("Enter your initial prompt: ")
            refined_prompt, refinement_messages = self.refine_prompt(user_prompt)
            
            # The refined_prompt and refinement_messages are now the baseline conversation history
            refined_prompt, refinement_messages = self.get_feedback_on_refined_prompt(refined_prompt, refinement_messages)

            # refinement_messages now contains history up to the point of user accepting the refined prompt (or last re-refinement)
            # user_responses is a simple list of strings, q_and_a_summary is the formatted string
            user_responses, q_and_a_summary = self.get_clarifying_responses(user_prompt, refined_prompt, refinement_messages)

            # Pass the full conversation history (refinement_messages) and the Q&A summary
            final_prompt = self.generate_final_prompt(refinement_messages, q_and_a_summary)
            print(f"Final optimized prompt: {final_prompt}")
        except Exception as e:
            logging.error(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    agent = PromptEngineerAgent()
    agent.run()
