from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.tools.file_ops import read_file


_llm = ChatOllama(
    model=settings.GENERATION_LLM,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,
    num_predict=512,
)

def summarize_text(text: str) -> str:
    """Summarize the provided text concisely and clearly."""
    try:
        if not text or not text.strip():
            return "Error: Empty text provided."
        resp = _llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a precise summarizer. "
                        "Return only the final summary text."
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        return (resp.content or "").strip()
    except Exception as e:
        return f"Error executing summarize_text: {e}"
    
def read_and_summarize_file(filename: str) -> str:
    """Read a sandboxed file and return its summary."""
    try:
        content = read_file(filename=filename)
        if isinstance(content, str) and content.startswith("Error:"):
            return content
        return summarize_text(content)
    except Exception as e:
        return f"Error reading/summarizing file: {e}"