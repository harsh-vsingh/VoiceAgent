import logging
from pathlib import Path
import shutil

from src.config import settings
from src.utils.errors import SandboxViolationError, ToolExecutionError

logger = logging.getLogger(__name__)

def _resolve_and_validate_path(filename: str) -> Path:
    """
    Resolves a filename and ensures it is strictly within the OUTPUT_DIR.
    Raises SandboxViolationError if it escapes the sandbox.
    """
    # Create an absolute path inside the sandbox
    target_path = (settings.OUTPUT_DIR / filename).resolve()
    
    # Check if the resolved path is a subpath of OUTPUT_DIR
    if not target_path.is_relative_to(settings.OUTPUT_DIR):
        logger.warning(f"Sandbox violation attempted: {filename}")
        raise SandboxViolationError(str(target_path))
        
    return target_path

def create_file(filename: str, content: str = "") -> dict:
    """
    Creates a new file (and any necessary parent directories) inside the sandbox.
    """
    try:
        target_path = _resolve_and_validate_path(filename)
        
        # Ensure parent directories exist (e.g., if filename is "scripts/test.py")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the content
        target_path.write_text(content, encoding="utf-8")
        msg = f"Successfully created/updated {filename}."
        logger.info(msg)
        
        return {"status": "success", "message": msg, "path": str(target_path)}
        
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("create_file", str(e))

def read_file(filename: str) -> dict:
    """Reads the content of a file inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        
        if not target_path.exists() or not target_path.is_file():
            return {"status": "error", "message": f"File {filename} does not exist."}
            
        content = target_path.read_text(encoding="utf-8")
        return {"status": "success", "content": content}
        
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("read_file", str(e))


def edit_file(filename: str, content: str) -> dict:
    """Appends content to an existing file inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        if not target_path.exists() or not target_path.is_file():
            return {"status": "error", "message": f"File {filename} does not exist. Cannot append."}
            
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}")
            
        return {"status": "success", "message": f"Appended to {filename}."}
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("edit_file", str(e))

def delete_file(filename: str) -> dict:
    """Deletes a file inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(filename)
        if not target_path.exists() or not target_path.is_file():
            return {"status": "error", "message": f"File {filename} not found."}
            
        target_path.unlink()
        return {"status": "success", "message": f"Deleted file {filename}."}
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("delete_file", str(e))


def list_directory(dirname: str = ".") -> dict:
    """Lists the contents of a directory inside the sandbox. Uses '.' for root sandbox."""
    try:
        target_path = _resolve_and_validate_path(dirname if dirname else ".")
        if not target_path.exists() or not target_path.is_dir():
            return {"status": "error", "message": f"Directory {dirname} not found."}
            
        items = [f.name + ("/" if f.is_dir() else "") for f in target_path.iterdir()]
        return {"status": "success", "message": f"Contents: {', '.join(items) if items else 'Empty directory'}"}
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("list_directory", str(e))

def create_directory(dirname: str) -> dict:
    """Creates a new directory inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(dirname)
        target_path.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "message": f"Created directory {dirname}."}
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("create_directory", str(e))

def delete_directory(dirname: str) -> dict:
    """Deletes a directory and its contents inside the sandbox."""
    try:
        target_path = _resolve_and_validate_path(dirname)
        
        # Security: Prevent deletion of the root output directory itself
        if target_path == settings.OUTPUT_DIR:
             return {"status": "error", "message": "Cannot delete the root output directory."}
             
        if not target_path.exists() or not target_path.is_dir():
            return {"status": "error", "message": f"Directory {dirname} not found."}
            
        shutil.rmtree(target_path)
        return {"status": "success", "message": f"Deleted directory {dirname}."}
    except SandboxViolationError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        raise ToolExecutionError("delete_directory", str(e))