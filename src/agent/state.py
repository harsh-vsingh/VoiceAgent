from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent workflow.
    State is passed between nodes and check-pointed for session memory.
    """
    # Conversation history (automatically appended)
    messages: Annotated[list, add_messages]
    
    # The latest transcribed text from the user's audio input
    transcription: str
    
    # Structured intents parsed from the LLM (queued for HITL approval)
    parsed_intents: List[Dict[str, Any]]
    
    # Execution results of the approved tools
    tool_results: List[Dict[str, Any]]