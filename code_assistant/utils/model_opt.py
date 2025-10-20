"""Utility functions for loading optimized models."""
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_embedding_model(embedding_model, device=None, model_max_length=8192):
    """ Loads the embedding model with specified configurations.
    Args:
        embedding_model: Name or path of the embedding model
        device: Device to load the model on (e.g., 'cpu', 'cuda')
        model_max_length: Maximum sequence length for the model
    return:
        Loaded SentenceTransformer model
    """
    print(f"Loading embedding model: {embedding_model}...")
    model = SentenceTransformer(embedding_model, device=device, trust_remote_code=True)
    model.max_seq_length = model_max_length
    print("Embedding model loaded.")
    return model


def get_llm_model(llm_model_name, device=None):
    """ Loads the LLM model with specified configurations.
    Args:
        llm_model_name: Name or path of the LLM model
        device: Device to load the model on (e.g., 'cpu', 'cuda')
    return:
        Loaded LLM model and tokenizer
    """

    print(f"Loading LLM model: {llm_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        trust_remote_code=True,
        device_map="auto" if device is None else { "": device },
        torch_dtype="auto"
    )
    print("LLM model loaded.")
    return model, tokenizer

def modify_prompt(query, context_chunks):
    """ Modifies the prompt to include system instructions and user query with context.
    Args:
        query: User's question
        context_chunks: List of relevant code context chunks
    return:
        Formatted messages for the LLM
    """

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
    return messages


def generate_answer(model, tokenizer, query, context_chunks):
    """ Generates an answer to the user's query based on the provided context chunks.
    Args:
        model: Loaded LLM model
        tokenizer: Corresponding tokenizer for the LLM model
        query: User's question
        context_chunks: List of relevant code context chunks
    return:
        Generated answer as a string
    """
    if not context_chunks:
        return "Sorry, I could not find any relevant context in the codebase for this question."

    messages = modify_prompt(query, context_chunks)

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

