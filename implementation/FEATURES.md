# Feature 1: Audio Input & Speech-to-Text (STT) Pipeline

## Overview
This document details the implementation strategy for capturing user audio and converting it to text using exclusively local models, ensuring OS independence and optimal UI performance.

## 1. UI Input Handling (Streamlit)
We will leverage Streamlit's native components to handle audio inputs:
*   **Microphone Input**: `st.audio_input("Record a voice message")`
*   **File Upload**: `st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])`

Both methods return file-like objects in Python. These will be temporarily saved/processed using standard Python libraries to ensure format compatibility before feeding them to the STT model.

## 2. Speech-to-Text (STT) Implementation
As per system requirements, we will strictly use **local models** for transcription.

*   **Engine**: We will use `faster-whisper` (or standard HuggingFace `transformers` pipeline) to run Whisper variants locally. `faster-whisper` is preferred for optimized local execution.
*   **Model Selection**: The UI will feature a sidebar selection (e.g., `tiny`, `base`, `small`, `distil-large-v3`) allowing the user to pick the STT model based on their hardware capabilities.

## 3. Performance & Memory Management (Caching)
Audio models are resource-heavy. Loading a model from disk on every Streamlit UI interaction will cause the application to freeze.
*   **Implementation**: We will use Streamlit's `@st.cache_resource` decorator on the model loading function. 
*   **Benefit**: The STT model is loaded into memory only once. Subsequent audio transcriptions will use the pre-loaded instance, resulting in near-instant processing.

## 4. OS Independence Strategy
To ensure the project runs seamlessly across Linux, Windows, and macOS:
*   **Audio Capture**: By using Streamlit's browser-based input components, we bypass the need for OS-specific system audio libraries (like `PyAudio` or `PortAudio`).
*   **File Paths**: All temporary audio file handling will strictly utilize Python's `pathlib` module to handle forward (`/`) and backward (`\`) slashes natively depending on the host OS.



# Feature 2: Agent Workflow & Intents

## Overview
This defines the core architecture of the AI agent using **LangGraph**. The workflow handles intent classification, supports compound commands, enforces structured outputs, and implements a Human-In-The-Loop (HITL) pause before tool execution.

## 1. Graph State Management (Memory)
State will be maintained throughout the graph execution using a `TypedDict`. This state is passed between nodes and ensures the system remembers the context.
*   **history**: A list of standard conversation messages (session memory).
*   **transcription**: The text transcribed from the current audio input.
*   **parsed_intents**: A list of structured intents extracted by the LLM.
*   **tool_results**: Outcomes of executed tools for UI display.

## 2. Intent Parsing & Compound Commands
To reliably handle local LLMs (like Llama-3 or Phi-3 via Ollama) without relying on proprietary tool-calling APIs, we will use **Structured JSON Output**.
*   The LLM will be prompted to output a strict JSON **array** of action objects.
*   **Supported Intents**: `create_file`, `write_code`, `summarize_text`, `general_chat`.
*   **Compound Commands**: By enforcing an array, a single input (e.g., "Summarize AI and save it to ai.txt") will output multiple intents: `[{"intent": "summarize_text", ...}, {"intent": "create_file", ...}]`. The execution node iterates through this array.

## 3. Human-In-The-Loop (HITL) Mechanism
Before the agent manipulates any files, it must ask for user permission.
*   **LangGraph Breakpoints**: The graph will be compiled with an interrupt immediately after the intent classification node.
*   **UI Integration**: When the graph pauses, the Streamlit UI reads `parsed_intents` and displays the planned actions with [Approve] or [Reject] buttons.
*   **Execution**: If approved, the graph resumes and triggers the Tool Execution node. If rejected, the state clears the pending actions.



# Feature 3: Tool Execution & Sandboxing

## Overview
This defines how the application securely executes the intents mapped by the LangGraph agent. It separates specialized AI tasks (like code generation) from general Python file/OS executions, while strictly enforcing directory sandboxing.

## 1. Directory Sandboxing (Security constraint)
To prevent prompt injection or accidental modification of host system files, a hard sandbox is enforced over the `output/` directory.
*   **Path Resolution**: Every file path requested by a tool is resolved to an absolute path using Python's `pathlib` (`Path(requested_path).resolve()`).
*   **Bounds Checking**: The system verifies if the resolved path is a child of the absolute path of the `output/` directory (`resolved.is_relative_to(OUTPUT_DIR)`).
*   **Violation Handling**: If an operation attempts to break out of `output/` (e.g., via `../` traversal), a custom `SandboxViolationError` is raised, caught gracefully, and displayed in the UI.

## 2. Tool Suites

### A. General File Manipulation (`file_ops`)
A dedicated suite for OS-level file and directory management.
*   **Supported Operations**: 
    1. `create_file`: Writes raw text/content to a new or existing file.
    2. `create_directory`: Creates nested folders.
    3. `read_file`: Reads content from a file to feed back into the state/LLM.
    4. `list_directory`: Returns contents of a requested folder inside the sandbox.

### B. Code Generation (`write_code`)
A specialized AI tool distinct from basic file creation.
*   **Logic**: 
    1. Accepts a coding prompt and a target filename.
    2. Calls an LLM specifically instructed/system-prompted for code generation.
    3. Extracts standard Markdown code blocks from the LLM output.
    4. Leverages the sandboxed file writer to save the extracted code to the target file.

### C. Text Processing (`summarize_text`)
*   **Logic**: Accepts input text (either directly from the user's audio or read from a file via a compound context) and processes it through the LLM with a strict summarization prompt.

### D. General Chat (`general_chat`)
*   **Logic**: Standard conversational fallback. Processes queries that do not require tool invocation using the state's message history.

## 3. Graceful Error Handling
Each tool function wraps its execution in a `try/except` block. Instead of throwing raw Python exceptions to the UI, tools return standardized execution dicts: `{"status": "success", "result": "..."}` or `{"status": "error", "message": "Sandbox violation..."}`. The LangGraph state stores this to display securely in Streamlit.



# Feature 4: User Interface & Benchmarking

## Overview
This section outlines the Streamlit-based frontend layout to display the full agentic pipeline and the dedicated testing suite for evaluating model performance.

## 1. User Interface (Streamlit) Layout
The UI is designed to be reactive, stateful, and transparent about the agent's internal thought process.
*   **Sidebar Config**: Allows users to select the STT model (e.g., `whisper-base`), the target local LLM, and view the current contents of the `output/` sandbox directory.
*   **Main Chat Interface**: Displays the conversation history. Each interaction provides transparency into the pipeline:
    *   **Transcribed Text**: The raw text output from the local STT engine.
    *   **Detected Intents**: An expandable section showing the structured JSON payload generated by the LLM.
    *   **Action Results**: The final output, generated code, or any sandbox validation errors cleanly formatted.
*   **HITL Container**: When LangGraph pauses for Human-In-The-Loop validation, a dedicated component renders the pending intents with explicit `[ Approve ]` and `[ Reject ]` buttons.
*   **Input Zone**: Anchored at the bottom of the screen, containing `st.audio_input` and `st.file_uploader` for seamless ingestion.

## 2. Model Benchmarking Setup
To fulfill the performance and speed comparison requirements robustly, benchmarking will be handled via a dedicated, automated testing suite rather than UI overlays.
*   **Dedicated Suite**: Scripts located in an isolated `benchmarks/` directory that programmatically process standard audio datasets and benchmark prompts.
*   **Metrics Evaluated**:
    *   **STT Models**: Transcription speed (Real Time Factor - RTF) across model sizes (`tiny` vs `base` vs `small`).
    *   **LLMs**: Intent routing latency, JSON schema adherence reliability, and tokens-per-second (TPS).
    *   **Outputs**: The benchmarking suite will generate comparison tables (CSV/Markdown) to be included in the final documentation, allowing data-driven decisions on the best local models to use.



# Feature 5: Session Memory & Context Management

## Overview
To allow the agent to remember prior instructions (e.g., "rename the file you just created") while respecting the strict token limits of local LLMs, we will implement a hybrid memory approach.

## 1. Short-Term Memory (Sliding Window)
Local models (like Llama-3 8B) have hard context limits. Sending the entire chat history for a long session will cause out-of-memory errors or degrade instruction following.
*   **Mechanism**: Before passing the state's message history to the LLM for intent classification, we will apply a sliding window filter.
*   **Implementation**: Only the most recent `N` interaction turns (e.g., the last 5 user prompts and agent responses) will be injected into the LLM's prompt context. 

## 2. Long-Term Memory (Persistent Storage)
While the LLM only "sees" the sliding window, the entire conversational history and tool execution log must be preserved so the user can review past actions or resume old sessions.
*   **Database**: We will use a lightweight local database (e.g., SQLite via standard Python `sqlite3` or a local JSON/TinyDB store) to save threads.
*   **Thread IDs**: Every session is assigned a unique `thread_id`. The database will map `thread_id` -> `full_message_history`.
*   **UI Integration**: The Streamlit user interface will allow users to select from a list of past "Threads" or "Sessions" in the sidebar. Loading a thread will pull the full history from the database to render in the UI, while the LangGraph state will automatically re-initialize with that historical context.

## 3. LangGraph Integration
*   LangGraph's native `MemorySaver` will be utilized to handle the graph's internal checkpointing mapped to the active `thread_id`. 
*   The State dictionary will differentiate between `full_history` (saved to DB) and `active_context` (the sliding window passed to the LLM).