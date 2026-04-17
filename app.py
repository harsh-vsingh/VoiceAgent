import uuid
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.audio.local_stt import HuggingFaceSTT
from src.audio.recorder import save_audio_to_temp
from src.agent.pipeline import parse_user_input, execute_plan
from src.config import settings

st.set_page_config(page_title="VoiceAgent", page_icon="🎙️", layout="wide")


@st.cache_resource
def load_stt_engine() -> HuggingFaceSTT:
    return HuggingFaceSTT()


def _dir_tree_lines(root: Path, prefix: str = "") -> list[str]:
    if not root.exists():
        return ["(missing)"]
    lines: list[str] = []
    items = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, p in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{p.name}/" if p.is_dir() else f"{prefix}{branch}{p.name}")
        if p.is_dir():
            ext = "    " if is_last else "│   "
            lines.extend(_dir_tree_lines(p, prefix + ext))
    return lines

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

NO_HITL_INTENTS = {
    "general_chat",
    "summarize_text",
    "read_and_summarize_file",
    "read_and_summarise_file",  # alias safety
    "read_file",
}

def _requires_hitl(plan: list[dict]) -> bool:
    if not plan:
        return False
    intents = [str(a.get("intent", "")).strip() for a in plan]
    return not all(i in NO_HITL_INTENTS for i in intents)



def _assistant_text(results: list[dict]) -> str:
    parts = [str(r.get("output", "")).strip() for r in results if str(r.get("output", "")).strip()]
    return "\n\n".join(parts) if parts else "Done."

def _render_assistant(turn: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])
        with st.expander("Details", expanded=settings.UI_DETAILS_EXPANDED_DEFAULT):
            st.json(
                {
                    "transcribed_text": turn.get("transcribed_text", ""),
                    "detected_intent": [a.get("intent") for a in turn.get("plan", [])],
                    "plan": turn.get("plan", []),
                    "results": turn.get("results", []),
                }
            )

def _history_messages(turns: list[dict]):
    msgs = []
    for t in turns:
        msgs.append(HumanMessage(content=t["user"]))
        msgs.append(AIMessage(content=t["assistant"]))
    return msgs


# Session
if "threads" not in st.session_state:
    first = str(uuid.uuid4())
    st.session_state.threads = {first: []}  # thread_id -> list[turns]
    st.session_state.thread_id = first

if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None
if "pending_user_text" not in st.session_state:
    st.session_state.pending_user_text = ""
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

stt_engine = load_stt_engine()

# Sidebar
with st.sidebar:
    st.header("Threads")
    ids = list(st.session_state.threads.keys())
    selected = st.selectbox("Select thread", ids, index=ids.index(st.session_state.thread_id))
    if selected != st.session_state.thread_id:
        st.session_state.thread_id = selected
        st.session_state.pending_plan = None
        st.session_state.pending_user_text = ""
        st.rerun()

    if st.button("➕ New Chat", use_container_width=True):
        nid = str(uuid.uuid4())
        st.session_state.threads[nid] = []
        st.session_state.thread_id = nid
        st.session_state.pending_plan = None
        st.session_state.pending_user_text = ""
        st.rerun()

    st.divider()
    if settings.UI_SHOW_SANDBOX_TREE:
        st.header("Sandbox: output/")
        st.code("\n".join(_dir_tree_lines(settings.OUTPUT_DIR)), language="text")

# Main
st.title("🎙️ VoiceAgent")
turns = st.session_state.threads[st.session_state.thread_id]

# Unified chat rendering
for t in turns:
    with st.chat_message("user"):
        st.write(t["user"])
    _render_assistant(t)

# HITL only for non-general intents
if st.session_state.pending_plan is not None:
    st.info("Pending action. Approve once to run full tool plan.")
    with st.expander("Details", expanded=True):
        st.json(
            {
                "transcribed_text": st.session_state.last_transcription,
                "detected_intent": [a.get("intent") for a in st.session_state.pending_plan],
                "plan": st.session_state.pending_plan,
            }
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve", use_container_width=True):
            results = execute_plan(st.session_state.pending_plan)
            turns.append(
                {
                    "user": st.session_state.pending_user_text,
                    "assistant": _assistant_text(results),
                    "transcribed_text": st.session_state.last_transcription,
                    "plan": st.session_state.pending_plan,
                    "results": results,
                }
            )
            st.session_state.pending_plan = None
            st.session_state.pending_user_text = ""
            st.session_state.last_transcription = ""
            st.rerun()

    with c2:
        if st.button("❌ Reject", use_container_width=True):
            turns.append(
                {
                    "user": st.session_state.pending_user_text,
                    "assistant": "Execution rejected.",
                    "transcribed_text": st.session_state.last_transcription,
                    "plan": st.session_state.pending_plan,
                    "results": [],
                }
            )
            st.session_state.pending_plan = None
            st.session_state.pending_user_text = ""
            st.session_state.last_transcription = ""
            st.rerun()

# Input
if st.session_state.pending_plan is None:
    col1, col2 = st.columns(2)
    with col1:
        mic = st.audio_input("Record")
    with col2:
        upl = st.file_uploader("Upload", type=["wav", "mp3", "m4a"])
    typed = st.chat_input("Or type a command...")

    user_text = typed
    audio_blob = mic or upl

    if audio_blob is not None:
        # Check if we already processed this exact audio file
        file_hash = hash(audio_blob.getvalue())
        if file_hash != st.session_state.processed_file_hash:
            with st.spinner("Transcribing..."):
                tmp = save_audio_to_temp(audio_blob)
                user_text = stt_engine.transcribe(tmp)
                tmp.unlink(missing_ok=True)
                
            # Mark as processed
            st.session_state.processed_file_hash = file_hash
        # Note: No 'else: user_text = None' here. We want to preserve 'typed' if audio is stale.

    if user_text and user_text.strip():
        user_text = user_text.strip()
        st.session_state.last_transcription = user_text

        history = _history_messages(turns)
        plan = parse_user_input(user_text, history)

        if _requires_hitl(plan):
            st.session_state.pending_user_text = user_text
            st.session_state.pending_plan = plan
        else:
            results = execute_plan(plan)
            turns.append(
                {
                    "user": user_text,
                    "assistant": _assistant_text(results),
                    "transcribed_text": user_text,
                    "plan": plan,
                    "results": results,
                }
            )

        st.rerun()