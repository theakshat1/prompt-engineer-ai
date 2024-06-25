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

    def refine_prompt(self, user_prompt: str) -> str:
        context = self.retrieve_context(user_prompt)
        logging.info(f"Retrieved context for refining prompt: {context}")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please refine and optimize the following prompt, using the provided context: '{user_prompt}'\n\nContext: {context}"}
        ]

        refined_prompt = self.generate_response(messages)
        logging.info(f"Refined prompt: {refined_prompt}")
        return refined_prompt

    def ask_clarifying_question(self, context: str):
        retrieved_context = self.retrieve_context(context)
        logging.info(f"Retrieved context for clarifying question: {retrieved_context}")

        messages = [
            {"role": "system", "content": """
            You are an expert prompt engineer. Your task is to refine and optimize user prompts 
            to make them more effective and understandable for AI chatbots like ChatGPT and Claude. 
            Analyze the given prompt, identify areas for improvement, and suggest refinements. 
            If necessary, explicitly ask clarifying questions to better understand the user's intent.
            Use the provided context from the knowledge base to inform your decisions and suggestions.
            """},
            {"role": "user", "content": f"Based on this detailed context and the retrieved information, identify and list specific clarifying questions to ask the user to obtain information that would optimize their initial prompt. If no further questions are needed, respond with '{NO_QUESTION}'.\n\nContext: {context}\n\nRetrieved Information: {retrieved_context}"}
        ]

        question = self.generate_response(messages)
        logging.info(f"Generated clarifying question: {question}")
        return question

    def generate_final_prompt(self, original_prompt: str, user_responses: List[str]) -> str:
        context = f"Original prompt: {original_prompt}\nUser responses: {'; '.join(user_responses)}"
        retrieved_context = self.retrieve_context(context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Generate a final, optimized prompt based on this information and the retrieved context:\n\nContext: {context}\n\nRetrieved Information: {retrieved_context}"}
        ]

        return self.generate_response(messages)

    @staticmethod
    def get_user_input(prompt: str) -> str:
        while True:
            user_input = input(prompt).strip()
            if user_input:
                return user_input
            print("Input cannot be empty. Please try again.")

    def get_clarifying_responses(self, user_prompt: str, refined_prompt: str) -> List[str]:
        user_responses = []
        max_questions = 3
        for i in range(max_questions):
            context = f"Original prompt: {user_prompt}\nRefined prompt: {refined_prompt}\nUser responses: {'; '.join(user_responses)}"
            logging.info(f"Context before clarifying question {i + 1}: {context}")

            question = self.ask_clarifying_question(context)
            logging.info(f"Clarifying question {i + 1}: {question}")

            if question.strip().upper() == NO_QUESTION:
                logging.info("No further clarifying questions needed.")
                break

            response = self.get_user_input(f"Clarifying question: {question}\nYour response (or press Enter to skip): ")
            logging.info(f"User response to clarifying question {i + 1}: {response}")

            if response:
                user_responses.append(response)
            else:
                break

        return user_responses

    def run(self) -> None:
        try:
            user_prompt = self.get_user_input("Enter your initial prompt: ")
            refined_prompt = self.refine_prompt(user_prompt)
            print(f"Refined prompt: {refined_prompt}")

            user_responses = self.get_clarifying_responses(user_prompt, refined_prompt)

            final_prompt = self.generate_final_prompt(user_prompt, user_responses)
            print(f"Final optimized prompt: {final_prompt}")
        except Exception as e:
            logging.error(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    agent = PromptEngineerAgent()
    agent.run()
