import os
import sys
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("Error: MONGO_URI not found in environment variables.")
    sys.exit(1)

# Configure models
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Global variables (load models and DB connection once)
try:
    # MongoDB
    client = MongoClient(MONGO_URI)
    db = client["yolo_code_db"]
    collection = db["code_chunks"]
    client.admin.command('ping')

    # Embedding Model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=None, trust_remote_code=True)
    embedding_model.max_seq_length = 8192

    # Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    # Tokenizer for Qwen (to format prompts)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)

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


# --- 3. Answer Generation Function ---

def generate_answer(query, context_chunks):
    if not context_chunks:
        return "Sorry, I could not find any relevant context in the codebase for this question."

    context_str = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are an expert assistant specializing in the Ultralytics YOLO codebase. "
        "Answer the user's question based *only* on the 'CODE CONTEXT' provided below. "
        "Be detailed and accurate, grounding your answer in the provided code. "
        "If the answer is not in the context, say 'Sorry, I could not find that information in the provided context.'"
    )

    user_prompt = f"""
    CODE CONTEXT:
    ---
    {context_str}
    ---
    
    USER QUESTION:
    {query}
    
    ANSWER:
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return answer.strip()

    except Exception as e:
        return f"Sorry, an error occurred during generation: {str(e)}"

def chat_func(message, history):
    context_chunks = find_relevant_chunks(message, k=3)
    answer = generate_answer(message, context_chunks)
    return answer


example_questions = [
    "How does the 'predict' method in the YOLO class work?",
    "Which function is used to load dataset labels? (in ultralytics/data/dataset.py)",
    "How is the 'train' method defined in the BaseTrainer class (engine/trainer.py)?",
    "Show me the 'init' method of the DetectionValidator class.",
]

css = """
footer {display: none !important;}
.gradio-container {
    background-color: #f8f9fa;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
.gradio-interface {
    flex: 1;
    display: flex;
    flex-direction: column;
}
.gr-chatbot {
    flex: 1;
    overflow-y: auto;
    scroll-behavior: smooth;
}
"""


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