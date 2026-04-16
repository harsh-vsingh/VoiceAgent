import logging
import shutil
from pathlib import Path
from src.config import settings
from src.utils.errors import SandboxViolationError

logger = logging.getLogger(__name__)

# Resolve sandbox root once for strict comparisons
_SANDBOX_ROOT = settings.OUTPUT_DIR.resolve()

def _resolve_and_validate_path(path_str: str) -> Path:
    if not path_str or not path_str.strip():
        raise ValueError("Path cannot be empty.")
    target_path = (_SANDBOX_ROOT / path_str).resolve()
    if not target_path.is_relative_to(_SANDBOX_ROOT):
        logger.warning(f"Sandbox violation attempted: {path_str}")
        raise SandboxViolationError(str(target_path))
    return target_path


def create_file(filename: str, content: str = "") -> str:
    """Creates a new file (and parent directories) inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return f"Successfully created {filename}."
    except Exception as e:
        return f"Error: {e}"

def read_file(filename: str) -> str:
    """Reads the content of a file inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        if not target_path.exists() or not target_path.is_file():
            return f"Error: File {filename} does not exist."
        return target_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: {e}"

def edit_file(filename: str, content: str) -> str:
    """Appends content to an existing file inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        if not target_path.exists() or not target_path.is_file():
            return f"Error: File {filename} does not exist."
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}")
        return f"Successfully appended to {filename}."
    except Exception as e:
        return f"Error: {e}"

def delete_file(filename: str) -> str:
    """Deletes a file inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        if not target_path.exists() or not target_path.is_file():
            return f"Error: File {filename} not found."
        target_path.unlink()
        return f"Successfully deleted {filename}."
    except Exception as e:
        return f"Error: {e}"

def create_directory(dirname: str) -> str:
    """Creates a new directory inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(dirname)
        target_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory {dirname}."
    except Exception as e:
        return f"Error: {e}"

def delete_directory(dirname: str) -> str:
    """Deletes a directory and its contents inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(dirname)
        if target_path == settings.OUTPUT_DIR:
             return "Error: Cannot delete the root output directory."
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory {dirname} not found."
        shutil.rmtree(target_path)
        return f"Successfully deleted directory {dirname}."
    except Exception as e:
        return f"Error: {e}"

def list_directory(dirname: str = ".") -> str:
    """Lists the contents of a directory inside the sandbox. Uses '.' for root sandbox."""
    try:
        target_path = _resolve_and_validate_path(dirname if dirname else ".")
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory {dirname} not found."
        items = [f.name + ("/" if f.is_dir() else "") for f in target_path.iterdir()]
        return f"Contents: {', '.join(items) if items else 'Empty directory'}"
    except Exception as e:
        return f"Error: {e}"