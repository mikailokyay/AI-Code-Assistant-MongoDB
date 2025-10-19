import ast
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SOURCE_DIRECTORIES = ["yolo-code-repo/ultralytics/ultralytics/models",
                      "yolo-code-repo/ultralytics/ultralytics/engine",
                      "yolo-code-repo/ultralytics/ultralytics/data"]

DB_NAME = "yolo_code_db"
COLLECTION_NAME = "code_chunks"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"


class CodeChunker(ast.NodeVisitor):
    def __init__(self, file_path, file_content):
        self.file_path = file_path
        self.file_content_lines = file_content.splitlines()
        self.chunks = []

    def get_node_code(self, node):
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line)
        return "\n".join(self.file_content_lines[start_line:end_line])

    def visit_FunctionDef(self, node):
        self.chunks.append({
            "file_path": str(self.file_path),
            "type": "function",
            "name": node.name,
            "lineno": node.lineno,
            "code_snippet": self.get_node_code(node)
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.chunks.append({
            "file_path": str(self.file_path),
            "type": "class",
            "name": node.name,
            "lineno": node.lineno,
            "code_snippet": self.get_node_code(node)
        })
        self.generic_visit(node)

    @staticmethod
    def chunk_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            chunker = CodeChunker(file_path, content)
            chunker.visit(tree)
            return chunker.chunks
        except Exception as e:
            return []


def get_mongo_client():
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        sys.exit(1)
    
    client = MongoClient(mongo_uri)
    try:
        client.admin.command('ping')
    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
        sys.exit(1)
    return client


def get_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL, device=None, trust_remote_code=True)
    model.max_seq_length = 8192
    return model

def main():
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    collection.delete_many({})

    model = get_embedding_model()

    all_files = []
    for dir_path in SOURCE_DIRECTORIES:
        all_files.extend(list(Path(dir_path).rglob("*.py")))
    
    print(f"Found {len(all_files)} Python files to process.")

    all_chunks = []
    for file_path in tqdm(all_files, desc="Chunking files"):
        all_chunks.extend(CodeChunker.chunk_file(file_path))

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
