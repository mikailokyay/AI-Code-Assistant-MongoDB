"""Utility functions for file handling and other common tasks."""
from pathlib import Path


def get_files(source_directories):
    """
    Retrieves all Python files from the specified source directories.
    Args:
        source_directories: List of directories to search for Python files
    return:
        List of Path objects for all Python files found
    """
    all_files = []
    for dir_path in source_directories:
        if not Path(dir_path).is_dir():
             print(f"Warning: Source directory not found: {dir_path}. Skipping.")
             continue
        all_files.extend(list(Path(dir_path).rglob("*.py")))
    
    print(f"Found {len(all_files)} Python files to process.")
    return all_files


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