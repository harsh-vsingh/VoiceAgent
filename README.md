# VoiceAgent (Interview Project)

Voice-controlled local AI agent built for the assignment:
- Audio/Text input
- Intent detection using Structured Outputs
- Tool execution via Deterministic Pipeline
- Single-step HITL approval (for risky/write actions)
- Clear UI outputs

---

## 1) Assignment Fit (What is implemented)

### Required flow
1. Input (voice/text)
2. Transcription (for audio)
3. Intent detection
4. Tool execution
5. Final response + UI details

### Required tools
- **File operations** (create file/folder) inside sandboxed `output/`
- **Code generation** and save to file
- **Text summarization**
- **Read + summarize file** helper for better reliability

### Required UI fields
- Transcribed text
- Detected intent
- Action taken
- Final output/result

These are shown in the chat output and hidden/expandable details panel.

---

## 2) Current Architecture

The app uses a deterministic pipeline powered by a split-model workflow:

1. `app.py`
   - Streamlit UI
   - Thread/session handling
   - HITL approval flow

2. `src/agent/pipeline.py`
   - Intent parsing strictly relies on **LLM Structured Outputs** (Pydantic).
   - *(Note: A legacy regex rules-based router is kept in the codebase as a fallback, but is flagged off by default).*
   - Plan normalization and robust argument extraction.
   - Deterministic tool execution (no LangGraph loops).

3. `src/tools/`
   - `file_ops.py`: Sandboxed file/folder ops
   - `code_gen.py`: Generate + save code
   - `summarizer.py`: Summarize text, read+summarize file

4. `src/config.py`
   - Configures the **Two-Model Architecture**: `ROUTER_LLM` (for structured JSON intent planning) and `GENERATION_LLM` (for coding/summarization).
   - No `.env` required for base run.

---

## 3) Repository Structure

```text
VoiceAgent/
├── app.py
├── tests.py
├── requirements.txt
├── README.md
├── benchmarks/
│   └── cases.json
├── output/                  # sandbox directory for all file writes
└── src/
    ├── config.py
    ├── agent/
    │   └── pipeline.py
    ├── tools/
    │   ├── file_ops.py
    │   ├── code_gen.py
    │   └── summarizer.py
    └── audio/
        ├── local_stt.py
        └── recorder.py

```

---

## 4) Setup

### Prerequisites
- Linux
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### 1. Create Environment
```bash
cd /home/harsh/projects/VoiceAgent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Required Local Models
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/whisper-base')"
```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## 5) HITL Policy

**No HITL** for low-risk intents:
- `general_chat`
- `summarize_text`
- `read_and_summarize_file`
- `read_file`

**Single HITL approval** for write/risky actions:
- `create_file`
- `create_directory`
- `write_code`

---

## 6) Hardware Workarounds & Optimizations Used

- **Dual Local Models:** Uses `qwen` for highly dependable structured JSON output (routing) and `llama3.1` (or equivalent) for text/code generation, optimizing 6GB-8GB VRAM limits.
- **Deterministic Executor:** Replaced cyclic agent loops with a structured deterministic executor to eliminate hallucinated tool calls.
- **Structured Outputs (Pydantic):** Enforces strict schema adherence for stable intent plans. Regex fallback is explicitly disabled to showcase pure LLM capabilities.
- **Selective Context:** Avoids dumping full file trees into prompts; context is tailored per tool.
- **Selective HITL:** Reduces UI friction by only pausing on write operations.

---

## 7) Security Notes

- **Path Traversal Blocked:** Protects against `../` or absolute escapes. 
- **Sandboxed Operations:** All read/write restricted strictly to `output/`.
- **Root Protection:** Sandbox root deletion is explicitly protected.

## 8) Performance Benchmarks

Speed tests were run locally to ensure a responsive user experience. The metrics below represent cold/warm inference times on local hardware.

**Speech-to-Text (Whisper) Latency:**
- `openai/whisper-tiny`: **~0.2 seconds**
- `openai/whisper-small`: **~0.6 seconds**

**LLM Intent Routing Latency (Structured JSON Generation):**
Measured on a complex prompt: *"Write code for a binary search in python and save it to search.py"*
- `llama3.1:8b`: **~4.8 seconds**
- `qwen2.5:7b-instruct` (Hybrid Setup): **~5.4 seconds**
- `llama3.2:3b`: **~5.8 seconds**

> **Design Choice:** While `llama3.1:8b` parsed the intent slightly faster in benchmarks, **`qwen2.5-instruct`** is used as the default Router LLM due to its better reliability in generating valid, hallucination-free JSON schemas.