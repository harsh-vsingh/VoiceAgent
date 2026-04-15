# Assignment: Build a Voice-Controlled Local AI Agent

## Objective

Build a local AI agent that:

* Accepts audio input
* Classifies user intent
* Executes local tools
* Displays the full pipeline in a clean UI

---

## System Requirements

### 1. Audio Input

Support:

* Direct microphone input
* Uploading audio files (`.wav`, `.mp3`)

---

### 2. Speech-to-Text (STT)

Convert audio to text.

* Use a HuggingFace model (e.g., Whisper, wav2vec) or another local model
* If local execution is not feasible, use an API (e.g., Groq, OpenAI)
* If using API, document the reason in README

---

### 3. Intent Understanding

Use a Large Language Model (preferably local via Ollama, LM Studio, etc.) to classify intent.

**Minimum supported intents:**

* Create a file
* Write code to a file
* Summarize text
* General chat

---

### 4. Tool Execution

Execute actions based on detected intent.

* **File Operations:** Create files/folders

  * Restrict all actions to an `output/` directory

* **Code Generation:** Generate and save code

* **Text Processing:** Summarize content

---

### 5. User Interface

Build using Streamlit, Gradio, or similar.

UI must display:

* Transcribed text
* Detected intent
* Action taken
* Final output/result

---

## Example Flow

**User Input:**
"Create a Python file with a retry function."

**System Steps:**

1. Transcribe audio
2. Detect intent ("Write code", "Create file")
3. Generate Python code
4. Create file in `output/`
5. Display results in UI

---


## Additional Features

* Compound commands (multiple actions in one input)
* Human-in-the-loop confirmation before execution
* Graceful error handling
* Session memory (chat + actions)
* Model benchmarking (performance/speed comparison)
