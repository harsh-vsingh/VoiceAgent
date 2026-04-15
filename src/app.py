import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.audio.local_stt import HuggingFaceSTT
from src.audio.recorder import save_audio_to_temp
from src.agent.graph import agent_graph
from src.config import settings

# --- Configuration & Caching ---
st.set_page_config(page_title="Local Voice Agent", page_icon="🎙️", layout="wide")

@st.cache_resource
def load_stt_engine():
    """Loads the STT model once to prevent reloading on every UI interaction."""
    return HuggingFaceSTT()

stt_engine = load_stt_engine()

# --- Session Management ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Configuration passed to LangGraph to identify the session memory
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    new_thread = st.text_input("Session ID", value=st.session_state.thread_id)
    if new_thread != st.session_state.thread_id:
        st.session_state.thread_id = new_thread
        st.rerun()
        
    if st.button("Start New Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    
    st.header("📁 Sandbox (output/)")
    if settings.OUTPUT_DIR.exists():
        for f in settings.OUTPUT_DIR.iterdir():
            icon = "📁" if f.is_dir() else "📄"
            st.text(f"{icon} {f.name}")
    else:
        st.caption("Sandbox empty.")

# --- Helpers ---
def render_chat_history():
    """Renders the conversational history from LangGraph's memory."""
    state = agent_graph.get_state(config)
    messages = state.values.get("messages", [])
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
             with st.chat_message("user"):
                 st.write(msg.content)
        elif isinstance(msg, AIMessage):
             with st.chat_message("assistant"):
                 st.write(msg.content)

def process_audio(audio_file):
    """Handles STT and sends the text to the LangGraph agent."""
    with st.spinner("Transcribing..."):
        temp_path = save_audio_to_temp(audio_file)
        transcription = stt_engine.transcribe(temp_path)
        temp_path.unlink(missing_ok=True) # Cleanup
        
    if transcription:
        st.session_state.last_transcription = transcription
        input_state = {"transcription": transcription}
        
        with st.spinner("Thinking..."):
            for event in agent_graph.stream(input_state, config):
                pass # Graph runs until the exact _interrupt_ before the 'act' node
        st.rerun()

# --- Main UI Layout ---
st.title("🎙️ Local Voice Agent")
st.caption(f"Session ID: {st.session_state.thread_id}")

# 1. Display Chat History
render_chat_history()

# 2. Check current graph state for HITL (Human-in-the-loop)
current_state = agent_graph.get_state(config)

# If the graph's next step is 'act', it means we hit our interrupt.
if current_state.next and "act" in current_state.next:
    st.warning("Action Approval Required")
    pending_intents = current_state.values.get("parsed_intents", [])
    
    with st.expander("Show Pending Actions", expanded=True):
        st.json(pending_intents)
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", use_container_width=True):
            with st.spinner("Executing..."):
                for event in agent_graph.stream(None, config):
                    pass # Resumes execution using None
                
            tool_results = agent_graph.get_state(config).values.get("tool_results", [])
            for res in tool_results:
                if res.get("status") == "success":
                    st.success(f"**{res.get('intent')}**: {res.get('message')}")
                else:
                    st.error(f"**{res.get('intent')}**: {res.get('message')}")
            
    with col2:
        if st.button("❌ Reject", use_container_width=True):
            # To cancel, we simply wipe out the pending intents from the state
            agent_graph.update_state(config, {"parsed_intents": []})
            st.error("Actions rejected.")
            st.rerun()

# 3. Audio Input Zone (Only show if not currently waiting for approval)
if not current_state.next:
    st.write("---")
    
    col_mic, col_file = st.columns(2)
    with col_mic:
        audio_mic = st.audio_input("Record a command")
    with col_file:
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
        
    audio_value = audio_mic or audio_file
    if audio_value:
        process_audio(audio_value)