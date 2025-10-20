"""Script to index code files from specified directories into a MongoDB database with embeddings."""
import os

from dotenv import load_dotenv
from tqdm import tqdm
from code_assistant.utils.db_opt import get_mongo_client
from code_assistant.utils.model_opt import get_embedding_model
from code_assistant.utils.utils import get_files
from code_assistant.data.chunk_data import CodeChunker

def main():
    load_dotenv()
    SOURCE_DIRECTORIES = os.getenv("SOURCE_DIRECTORIES").split(",")
    DB_NAME = os.getenv("DB_NAME", "yolo_code_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_chunks")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code")
    MONGO_URI = os.getenv("MONGO_URI")

    client = get_mongo_client(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    update_index = os.getenv("UPDATE_INDEX", "False").lower() == "true"
    if update_index:
        print("\nUPDATE_INDEX environment variable is set. Forcing full re-indexing and clearing the collection.")
        collection.delete_many({})
        document_count = 0 
    else:
        document_count = collection.count_documents({})

    if document_count > 0:
        print(f"\nDatabase already indexed. Found {document_count} documents in '{COLLECTION_NAME}'.")
        print("If you need to re-index, please set UPDATE_INDEX=True environment variable or clear the collection manually.")
        client.close()
        return

    print("Database is empty or cleared. Starting fresh indexing process...")
    
    model = get_embedding_model(EMBEDDING_MODEL)

    all_files = get_files(SOURCE_DIRECTORIES)

    all_chunks = []
    for file_path in tqdm(all_files, desc="Chunking files"):
        all_chunks.extend(CodeChunker.chunk_file(file_path))
    
    print(f"Total chunks created: {len(all_chunks)}")

    chunks_to_insert = []

    for chunk in tqdm(all_chunks, desc="Creating embeddings"):
        content_to_embed = f"Path: {chunk['file_path']}\nName: {chunk['name']}\nType: {chunk['type']}\nCode:\n{chunk['code_snippet']}"

        embedding = model.encode(content_to_embed).tolist()

        chunk["embedding"] = embedding
        chunk["content_for_llm"] = content_to_embed
        chunks_to_insert.append(chunk)

    if chunks_to_insert:
        collection.insert_many(chunks_to_insert)
    client.close()


if __name__ == "__main__":
    main()
