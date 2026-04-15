import json
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from src.config import settings
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# System prompt to enforce structured JSON array output for compound commands
INTENT_SYSTEM_PROMPT = """You are an AI assistant orchestrating a local agent.
Your job is to analyze the user's input and determine the sequence of actions to take.
You MUST output a valid JSON array of action objects. Do not write any markdown code blocks, explanations, or text outside the JSON array.

Available intents:
- "create_file": Requires "filename" and "content".
- "read_file": Requires "filename".
- "edit_file": Requires "filename" and "content" (to append).
- "delete_file": Requires "filename".
- "create_directory": Requires "dirname".
- "delete_directory": Requires "dirname".
- "list_directory": Requires "dirname" (use "." for root).
- "write_code": Requires "filename" and "prompt".
- "summarize_text": Requires "text" (the text to summarize).
- "general_chat": Requires "response" (a conversational reply to the user).

Example output for "Summarize AI and save to ai.txt":
[
  {"intent": "summarize_text", "text": "AI technology..."},
  {"intent": "create_file", "filename": "ai.txt", "content": "AI technology..."}
]
"""

def parse_intents(state: AgentState) -> dict:
    """Node: Parses the user's audio transcription into structured intents."""
    transcription = state.get("transcription", "")
    if not transcription:
        return {"parsed_intents": [], "messages": []}

    llm = ChatOllama(
        model=settings.DEFAULT_LLM, 
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,  # Low temperature for more deterministic JSON structure
        format="json"     # Request JSON format if native to the model
    )

    # Implement Sliding Window: Get only the last 6 messages (3 conversation turns) to avoid context bloat
    recent_history = state.get("messages", [])[-6:]
    
    messages = [SystemMessage(content=INTENT_SYSTEM_PROMPT)]
    messages.extend(recent_history) # Inject the short-term context
    messages.append(HumanMessage(content=transcription)) # Add current input

    try:
        response = llm.invoke(messages)
        raw_content = response.content.strip()
        
        # Clean up potential markdown formatting hallucinates by local models
        clean_json = re.sub(r"^```json\s*", "", raw_content, flags=re.IGNORECASE)
        clean_json = re.sub(r"\s*```$", "", clean_json).strip()
        
        parsed = json.loads(clean_json)
        
        # Ensure it's always an array to support compound commands seamlessly
        if isinstance(parsed, dict):
            parsed = [parsed]
            
        return {
            "parsed_intents": parsed,
            "messages": [HumanMessage(content=transcription)] # Append to history
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON intents: {e}")
        # Fallback to general chat if structure breaks
        return {
            "parsed_intents": [{"intent": "general_chat", "response": "Sorry, I couldn't format the command correctly."}],
            "messages": [HumanMessage(content=transcription)]
        }




from src.tools.file_ops import (
    create_file, read_file, edit_file, delete_file, 
    create_directory, delete_directory, list_directory
)
from src.tools.code_gen import write_code
from src.tools.summarizer import summarize_text
from langchain_core.messages import AIMessage

def execute_tools(state: AgentState) -> dict:
    """Node: Executes the actions approved in 'parsed_intents'."""
    parsed_intents = state.get("parsed_intents", [])
    results = []
    messages_to_add = []

    for action in parsed_intents:
        intent = action.get("intent")
        result = {"intent": intent, "status": "error", "message": "Unknown intent."}

        try:
            if intent == "create_file":
                result = create_file(action.get("filename"), action.get("content", ""))
            elif intent == "read_file":
                result = read_file(action.get("filename"))
            elif intent == "edit_file":
                result = edit_file(action.get("filename"), action.get("content", ""))
            elif intent == "delete_file":
                result = delete_file(action.get("filename"))
            elif intent == "create_directory":
                result = create_directory(action.get("dirname"))
            elif intent == "delete_directory":
                result = delete_directory(action.get("dirname"))
            elif intent == "list_directory":
                result = list_directory(action.get("dirname", "."))
            elif intent == "write_code":
                result = write_code(action.get("filename"), action.get("prompt", ""))
            elif intent == "summarize_text":
                result = summarize_text(action.get("text", ""))
            elif intent == "general_chat":
                response = action.get("response", "I have nothing to add.")
                result = {"status": "success", "message": response}
                messages_to_add.append(AIMessage(content=response))
            else:
                 result["message"] = f"Tool '{intent}' is not implemented."

        except Exception as e:
            logger.error(f"Tool execution failed for {intent}: {e}")
            result["message"] = str(e)

        # Attach the original intent to the result for UI tracking
        result["intent"] = intent
        results.append(result)

    # Return the aggregated results and add AI messages if it was a general chat
    return {
        "tool_results": results,
        "messages": messages_to_add,
        "parsed_intents": [] # Clear pending intents after execution
    }


