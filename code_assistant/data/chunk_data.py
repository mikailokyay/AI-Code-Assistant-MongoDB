"""Module to chunk Python code files into functions and classes."""
import ast

class CodeChunker(ast.NodeVisitor):
    """Chunks Python code files into functions and classes."""
    def __init__(self, file_path, file_content):
        """
        Initializes the CodeChunker with file path and content.
        Args:
            file_path: Path to the Python file
            file_content: Content of the Python file as a string
        """
        self.file_path = file_path
        self.file_content_lines = file_content.splitlines()
        self.chunks = []

    def get_node_code(self, node):
        """
        Extracts the source code snippet for a given AST node.
        Args:
            node: AST node (FunctionDef or ClassDef)
        return: 
            Source code snippet as a string
        """
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line)
        return "\n".join(self.file_content_lines[start_line:end_line])

    def visit_FunctionDef(self, node):
        """
        Visits function definitions and creates chunks.
         Args:
             node: AST FunctionDef node
         """
        self.chunks.append({
            "file_path": str(self.file_path),
            "type": "function",
            "name": node.name,
            "lineno": node.lineno,
            "code_snippet": self.get_node_code(node)
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """
        Visits class definitions and creates chunks.
        Args:
            node: AST ClassDef node
        """
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
        """
        Chunks a Python file into functions and classes.
        Args:
            file_path: Path to the Python file
        return:
            List of chunks, each chunk is a dict with keys: file_path, type, name, lineno, code_snippet
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
            chunker = CodeChunker(file_path, content)
            chunker.visit(tree)
            return chunker.chunks
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []