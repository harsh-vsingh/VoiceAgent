from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.utils.errors import ToolExecutionError

def summarize_text(text: str) -> dict:
    """Summarizes provided text using the local LLM."""
    try:
        llm = ChatOllama(model=settings.DEFAULT_LLM, base_url=settings.OLLAMA_BASE_URL)
        
        messages = [
            SystemMessage(content="Summarize the following text concisely. Provide only the summary."),
            HumanMessage(content=text)
        ]
        
        response = llm.invoke(messages)
        return {"status": "success", "summary": response.content.strip()}
        
    except Exception as e:
        raise ToolExecutionError("summarize_text", str(e))