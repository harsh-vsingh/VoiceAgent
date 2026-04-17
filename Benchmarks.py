import time
import urllib.request
from pathlib import Path
from langchain_core.messages import HumanMessage
from transformers import pipeline

from src.config import settings
from src.agent.pipeline import parse_user_input

# --- Configuration Sets ---
STT_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small"
]

LLM_CONFIGS = {
    "All Llama 3.2 3B": {
        "router": "llama3.2:3b",
        "gen": "llama3.2:3b",
        "code": "llama3.2:3b"
    },
    "All Llama 3.1 8B": {
        "router": "llama3.1:8b",
        "gen": "llama3.1:8b",
        "code": "llama3.1:8b"
    },
    "Hybrid Setup": {
        "router": "qwen2.5:7b-instruct-q4_K_M",
        "gen": "llama3.1:8b",
        "code": "qwen2.5-coder:7b"
    }
}

import wave
import struct

def download_sample_audio(filename="sample.wav"):
    """Generates a dummy 2-second audio file offline instead of downloading."""
    if not Path(filename).exists():
        print("Generating a sample audio file locally...")
        sample_rate = 16000 # 16kHz
        duration = 2 # 2 seconds
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)       # mono
            wav_file.setsampwidth(2)       # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            
            # Generate 2 seconds of silence (zero amplitude)
            for _ in range(sample_rate * duration):
                wav_file.writeframes(struct.pack('h', 0))
                
    return filename

def benchmark_stt():
    print("\n=== STT Benchmarks (Whisper) ===")
    audio_file = download_sample_audio()
    
    for model_id in STT_MODELS:
        print(f"Loading {model_id} (may download if first time)...")
        # Load model explicitly for timing
        pipe = pipeline("automatic-speech-recognition", model=model_id)
        
        # Warmup
        pipe(audio_file)
        
        # Benchmark
        start = time.time()
        pipe(audio_file)
        duration = time.time() - start
        
        print(f"[{model_id}] Transcription time: {duration:.2f} seconds\n")

def benchmark_llms():
    print("\n=== LLM Pipeline Benchmarks (Routing specific) ===")
    test_prompt = "Write code for a binary search in python and save it to search.py"
    history = [HumanMessage(content="Hello")]
    
    # Needs to dynamically patch the pipeline llm instances
    import src.agent.pipeline as pl
    from langchain_ollama import ChatOllama
    
    for name, config in LLM_CONFIGS.items():
        print(f"Testing Config: {name}")
        
        # Patch models
        pl.raw_llm = ChatOllama(model=config["router"], base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
        pl.structured_llm = pl.raw_llm.with_structured_output(pl.ActionPlan)
        
        try:
            # Warmup
            pl.parse_user_input("hello", history)
            
            # Benchmark 
            start = time.time()
            plan = pl.parse_user_input(test_prompt, history)
            duration = time.time() - start
            
            intent = plan[0].get("intent") if plan else "None"
            print(f"  -> Time: {duration:.2f}s | Parsed intent: {intent}")
        except Exception as e:
            print(f"  -> Failed: Make sure {config['router']} is pulled in Ollama! ({e})")

if __name__ == "__main__":
    benchmark_stt()
    benchmark_llms()