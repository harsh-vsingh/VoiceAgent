import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import settings
from src.tools.file_ops import create_file

def write_code(filename: str, prompt: str) -> str:
    """Generates code using an LLM based on a prompt and saves it to a file."""
    try:
        # Add temperature=0.0 for strict determinism
        llm = ChatOllama(
        model=settings.CODE_LLM,
        base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0
        )
        
        # Make the system prompt explicitly forbid JSON and tool-call mimicry
        messages = [
            SystemMessage(content=(
                "You are a strict code generator. "
                "Output ONLY valid, runnable raw code for the requested prompt. "
                "Do NOT include markdown formatting. "
                "Do NOT output JSON. Do NOT output tool calls. "
                "Do NOT explain the code. Just output the code."
            )),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        code = response.content.strip()
        
        # Strip markdown syntax just in case it leaks through
        code = re.sub(r"^```[\w]*\n", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\n```$", "", code)
        code = re.sub(r"```", "", code)
        
        # Save using our existing tool logic
        return create_file(filename=filename, content=code)        
    except Exception as e:
        return f"Error executing write_code: {e}"