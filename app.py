import os
import sys
import gradio as gr
from dotenv import load_dotenv
from code_assistant.utils.utils import css
from code_assistant.utils.db_opt import get_mongo_client
from code_assistant.utils.model_opt import get_embedding_model, get_llm_model, generate_answer

try:
    load_dotenv()
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")

    DB_NAME = os.getenv("DB_NAME", "yolo_code_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_chunks")
    MONGO_URI = os.getenv("MONGO_URI")
    
    # MongoDB
    client = get_mongo_client(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Embedding Model
    embedding_model = get_embedding_model(EMBEDDING_MODEL)
    model, tokenizer = get_llm_model(LLM_MODEL_NAME)
    
except Exception as e:
    # Exit silently on error
    print(f"Initialization error: {str(e)}")
    sys.exit(1)


def find_relevant_chunks(query, k=3):
    try:
        query_embedding = embedding_model.encode(query).tolist()

        search_pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",  # The name of your index in Atlas
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": k
                }
            },
            {
                "$project": {
                    "content_for_llm": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(search_pipeline))

        if not results:
            return []

        return [result["content_for_llm"] for result in results]

    except Exception as e:
        return []


def chat_func(message, history):
    context_chunks = find_relevant_chunks(message, k=3)
    answer = generate_answer(model, tokenizer, message, context_chunks)
    return answer


example_questions = [
    "How does the 'predict' method in the YOLO class work?",
    "Which function is used to load dataset labels? (in ultralytics/data/dataset.py)",
    "How is the 'train' method defined in the BaseTrainer class (engine/trainer.py)?",
    "Show me the 'init' method of the DetectionValidator class.",
]


iface = gr.ChatInterface(
    fn=chat_func,
    title="Ultralytics Code Assistant (Qwen Coder)",
    description="Ask questions about the Ultralytics codebase (`models`, `engine`, `data`).",
    examples=example_questions,
    theme="soft",
    css=css,
)

if __name__ == "__main__":
    iface.launch()
