from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.state import AgentState
from src.agent.nodes import parse_intents, execute_tools
from src.config import settings

def build_graph():
    """Builds and compiles the LangGraph agent workflow with HITL and SQLite capabilities."""
    
    builder = StateGraph(AgentState)
    builder.add_node("think", parse_intents)
    builder.add_node("act", execute_tools)
    
    builder.add_edge(START, "think")
    builder.add_edge("think", "act")
    builder.add_edge("act", END)
    
    # Initialize persistent SQLite checkpointing for long-term session memory
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    memory = SqliteSaver(conn)
    
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["act"]
    )
    
    return graph

agent_graph = build_graph()
