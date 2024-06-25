import os
import json
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
# Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "prompt-engineering-knowledge"

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

from pinecone import Pinecone, ServerlessSpec

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 384):
    """Create Pinecone index if it doesn't exist, otherwise return existing index."""
    # List all indexes
    index_list = pc.list_indexes()
    index_created = False
    # Check if our index exists
    if any(index.name == index_name for index in index_list):
        print(f"Pinecone index '{index_name}' already exists. Using existing index.")
    else:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Created new Pinecone index: {index_name}")
        index_created=True

    # Return the index object
    return pc.Index(index_name), index_created
    
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for given text."""
    return model.encode(text).tolist()

def upsert_to_index(index, data: List[Dict]):
    """Upsert data to Pinecone index."""
    batch_size = 100
    vectors = []

    for item in data:
        text = f"{item['category']} - {item['subcategory']} - {item['title']}: {item['content']}"
        vector = (
            item['id'],
            generate_embedding(text),
            {
                'category': item['category'],
                'subcategory': item['subcategory'],
                'title': item['title'],
                'content': item['content']
            }
        )
        vectors.append(vector)

        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)

def process_entries(category: str, subcategory: str, entries: List[Dict], id_prefix: str) -> List[Dict]:
    """Process entries and return a list of formatted dictionaries."""
    return [
        {
            'id': f"{id_prefix}_{i}",
            'category': category,
            'subcategory': subcategory,
            'title': entry['title'],
            'content': entry['content']
        }
        for i, entry in enumerate(entries)
    ]

def process_knowledge_base(knowledge_base: Dict) -> List[Dict]:
    """Process the entire knowledge base and return a flat list of entries."""
    all_entries = []
    for i, category in enumerate(knowledge_base['categories']):
        if 'entries' in category:
            all_entries.extend(process_entries(category['name'], '', category['entries'], f"cat_{i}"))
        if 'subcategories' in category:
            for j, subcategory in enumerate(category['subcategories']):
                all_entries.extend(process_entries(category['name'], subcategory['name'], subcategory['entries'], f"cat_{i}_sub_{j}"))
    return all_entries

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: Pinecone API key not set.")
        return
    pc = Pinecone(api_key=api_key)
    index_name = "prompt-engineering-knowledge"
    try:
    # Load knowledge base from JSON
        knowledge_base = json.loads('''
        {
          "index_name": "prompt-engineering-knowledge",
          "categories": [
            {
              "name": "Overview",
              "entries": [
                {
                  "title": "Definition and Importance",
                  "content": "Prompt engineering is the practice of designing and optimizing input prompts for AI language models to generate desired outputs. It's crucial for effectively leveraging AI capabilities, improving model performance, and achieving specific task outcomes. Example: 'Design a prompt that generates a concise summary of a research paper, highlighting key findings and methodologies.'"
                },
                {
                  "title": "Historical Context",
                  "content": "Prompt engineering emerged with the advent of large language models. It gained prominence with GPT-3's release in 2020, evolving from simple queries to sophisticated techniques for guiding AI behavior and output. Example: Early prompt: 'What is the capital of France?' vs. Advanced prompt: 'Analyze the economic impact of Paris being the capital of France, considering historical, cultural, and political factors.'"
                },
                {
                  "title": "Key Goals",
                  "content": "The main objectives of prompt engineering include enhancing output accuracy, consistency, and relevance; reducing hallucinations; improving task completion; and adapting language models to specific domains or use cases. Example goal: 'Create a prompt that generates a factual, bias-free news article about a given event, including multiple perspectives and verified sources.'"
                }
              ]
            },
            {
              "name": "Techniques",
              "subcategories": [
                {
                  "name": "Prompt Design Strategies",
                  "entries": [
                    {
                      "title": "Direct Prompts vs. Indirect Prompts",
                      "content": "Direct prompts explicitly state the desired outcome, while indirect prompts guide the model through a series of steps or thought processes. Each approach has its merits depending on the task and desired outcome. Example direct prompt: 'List five benefits of exercise.' Example indirect prompt: 'Imagine you're a fitness expert writing an article. What would you tell someone who's never exercised before about the positive changes they might experience if they start working out regularly?'"
                    },
                    {
                      "title": "Instruction-based Prompts",
                      "content": "These prompts provide clear, step-by-step instructions to the model. They're effective for complex tasks or when specific procedures need to be followed, helping to structure the model's response. Example: 'Follow these steps to analyze a poem: 1) Identify the main theme. 2) Describe the rhyme scheme and meter. 3) Highlight any literary devices used. 4) Explain how these elements contribute to the poem's overall meaning.'"
                    },
                    {
                      "title": "Few-shot and Zero-shot Prompting",
                      "content": "Few-shot prompting provides examples within the prompt to guide the model's response. Zero-shot prompting requires the model to perform tasks without specific examples, relying on its pre-trained knowledge. Few-shot example: 'Translate the following sentences from English to French: 1) Hello - Bonjour, 2) Goodbye - Au revoir. Now translate: Good morning.' Zero-shot example: 'Translate the following sentence from English to Swahili: The cat is sleeping on the couch.'"
                    }
                  ]
                },
                {
                  "name": "Prompt Optimization",
                  "entries": [
                    {
                      "title": "Iterative Testing and Refining",
                      "content": "This involves systematically testing different prompt variations, analyzing outputs, and making incremental improvements to achieve optimal results. It's a crucial process for fine-tuning prompts for specific use cases. Example process: Initial prompt: 'Write a product description.' Refined prompt: 'Write a compelling 100-word product description for a high-end smartphone, highlighting its unique features and benefits for tech-savvy professionals.'"
                    },
                    {
                      "title": "Parameter Tuning",
                      "content": "Adjusting model parameters such as temperature, top_p, and max_tokens can significantly impact output quality and consistency. Understanding and optimizing these parameters is key to effective prompt engineering. Example: 'For a creative writing task, increase the temperature to 0.8 to encourage more diverse and imaginative outputs. For a fact-based Q&A task, lower the temperature to 0.2 for more focused and conservative responses.'"
                    },
                    {
                      "title": "Feedback Loop Implementation",
                      "content": "Incorporating user feedback or automated evaluation metrics into the prompt engineering process allows for continuous improvement and adaptation to changing requirements or edge cases. Example: 'Implement a system where users rate the relevance of AI-generated responses on a scale of 1-5. Use this feedback to automatically fine-tune prompts that consistently receive low ratings.'"
                    }
                  ]
                }
              ]
            },
            {
              "name": "Best Practice",
              "entries": [
                {
                  "title": "Clarity and Specificity",
                  "content": "Creating clear, concise, and specific prompts is essential for obtaining desired outputs. Ambiguous or vague prompts often lead to inconsistent or irrelevant responses from the model. Example of a vague prompt: 'Tell me about cars.' Improved, specific prompt: 'Provide a detailed comparison of the fuel efficiency, safety features, and price ranges of the top three best-selling electric vehicles in the US market for 2023.'"
                },
                {
                  "title": "Context Provision",
                  "content": "Providing relevant context helps the model understand the task better and produce more accurate results. Example: Instead of 'Explain quantum computing', use 'Explain quantum computing to a high school student who has a strong foundation in classical physics but no prior knowledge of quantum mechanics.'"
                },
                {
                  "title": "Role Assignment",
                  "content": "Assigning a specific role to the AI can improve the relevance and tone of its responses. Example: 'As an experienced climate scientist, explain the potential long-term effects of rising sea levels on coastal urban areas.'"
                },
                {
                  "title": "Output Format Specification",
                  "content": "Clearly defining the desired output format can significantly improve the structure and usability of the AI's response. Example: 'Create a weekly meal plan for a vegetarian diet. Present it in a table format with columns for each day of the week and rows for breakfast, lunch, and dinner.'"
                },
                {
                  "title": "Ethical Considerations",
                  "content": "Incorporate ethical guidelines into prompts to ensure responsible AI use. Example: 'Generate a balanced news article about the upcoming election, ensuring equal representation of all major political parties and avoiding any form of bias or misleading information.'"
                }
              ]
            },
            {
              "name": "Case Studies",
              "entries": [
                {
                  "title": "Legal Document Summarization",
                  "content": "Prompt: 'As an experienced paralegal, provide a concise summary of the following legal document, focusing on key clauses, parties involved, and potential implications. Include any technical legal terms with brief explanations.' Analysis: This prompt succeeds by 1) Establishing a specific role, 2) Clearly defining the task, 3) Specifying focus areas, and 4) Requesting additional context for technical terms."
                },
                {
                  "title": "E-commerce Product Description Generator",
                  "content": "Prompt: 'You are a skilled copywriter for an e-commerce platform. Create a compelling product description for a new smartphone. Include: 1) A catchy headline (max 10 words), 2) Three key features with brief explanations, 3) A paragraph on why this product stands out from competitors, and 4) A call-to-action sentence. Use a friendly, enthusiastic tone aimed at tech-savvy millennials.' Analysis: This prompt works well because it 1) Assigns a specific role, 2) Provides a clear structure for the output, 3) Specifies the tone and target audience, and 4) Balances creativity with specific requirements."
                },
                {
                  "title": "Data Analysis Report Generation",
                  "content": "Prompt: 'As a data scientist, analyze the provided dataset on customer churn for a telecom company. Generate a report that includes: 1) Summary statistics of key variables, 2) Identification of top 3 factors contributing to churn, 3) Visualizations describing these factors (describe them in text), and 4) Recommendations for reducing churn rate. Use clear, non-technical language suitable for a business audience.' Analysis: This prompt is effective because it 1) Sets a professional context, 2) Clearly outlines the required components of the analysis, 3) Specifies the need for visualizations while acknowledging the text-based nature of the output, and 4) Defines the target audience to guide language use."
                },
                {
                  "title": "Educational Content Creation",
                  "content": "Prompt: 'You are developing an interactive online course on basic programming concepts for high school students. Create a lesson plan for teaching variables and data types in Python. Include: 1) Learning objectives, 2) A brief introduction to the concept, 3) Three examples with explanations, 4) A simple coding exercise for students to practice, and 5) A multiple-choice quiz question to test understanding. Use an engaging, encouraging tone suitable for teenagers.' Analysis: This prompt succeeds by 1) Providing a specific educational context, 2) Outlining a clear structure for the lesson plan, 3) Requesting varied content types (explanation, examples, exercise, quiz), and 4) Specifying the appropriate tone for the target audience."
                }
              ]
            },
            {
              "name": "Advanced Topics",
              "entries": [
                {
                  "title": "Cross-Lingual Prompt Engineering",
                  "content": "Techniques for designing prompts that work effectively across multiple languages, addressing challenges such as cultural nuances, idioms, and varying linguistic structures. Example: 'Design a prompt for generating marketing slogans that can be easily localized: Create a catchy 5-word slogan for a global coffee brand that evokes feelings of warmth, community, and energy. The slogan should use simple, universally understood concepts that can be easily translated into multiple languages without losing its core message.'"
                },
                {
                  "title": "Domain-Specific Prompt Engineering",
                  "content": "Strategies for tailoring prompts to specialized fields like law, medicine, or engineering. This involves incorporating domain-specific terminology, concepts, and reasoning patterns into prompt design. Example for medical domain: 'You are an AI assistant for a radiologist. Analyze the following chest X-ray report and provide a differential diagnosis, considering the patient's age, gender, and reported symptoms. List potential follow-up tests or imaging studies that might be necessary to confirm the diagnosis.'"
                },
                {
                  "title": "Integrating Prompts with Other AI Systems",
                  "content": "Approaches for combining prompt engineering with other AI technologies such as computer vision, speech recognition, or robotic systems to create more comprehensive and capable AI solutions. Example: 'Design a prompt for a multimodal AI system that combines image recognition and natural language processing: Analyze the uploaded image of a busy urban intersection. Describe the scene in detail, paying attention to traffic patterns, pedestrian behavior, and potential safety hazards. Then, suggest three urban planning improvements that could enhance safety and traffic flow based on your analysis.'"
                }
              ]
            }
          ]
        }
        ''')
    
        # Process knowledge base
        processed_data = process_knowledge_base(knowledge_base)

        # Create or get Pinecone index
        index, index_created = create_pinecone_index(pc, index_name)
        print(f"Pinecone index '{index_name}' is ready.")

        # Upsert data to index
        if index_created:
            upsert_to_index(index, processed_data)
            print(f"Uploaded {len(processed_data)} entries to the index.")

        # Verify data insertion
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON data.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()
