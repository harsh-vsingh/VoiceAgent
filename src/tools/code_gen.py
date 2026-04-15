import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.tools.file_ops import create_file
from src.utils.errors import ToolExecutionError

def write_code(filename: str, prompt: str) -> dict:
    """Generates code using local LLM and saves it to a file."""
    try:
        llm = ChatOllama(model=settings.DEFAULT_LLM, base_url=settings.OLLAMA_BASE_URL)
        
        messages = [
            SystemMessage(content="You are an expert coder. Output ONLY valid, runnable code for the requested prompt. Do not include markdown formatting or explanations."),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        code = response.content.strip()
        
        # Strip markdown syntax if the LLM hallucinates it despite instructions
        code = re.sub(r"^```[\w]*\n", "", code)
        code = re.sub(r"\n```$", "", code)
        
        # Use our sandboxed file creator
        return create_file(filename, code)
        
    except Exception as e:
        raise ToolExecutionError("write_code", str(e))