from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseSettings):
    BASE_DIR: Path = BASE_DIR
    OUTPUT_DIR: Path = BASE_DIR / "output"
    DB_PATH: Path = BASE_DIR / "sessions.db"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM: str = "llama3.1:8b"

    DEFAULT_STT_MODEL: str = "base"
    
    MAX_RECURSION_LIMIT: int = 25

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Ensure the output sandbox directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

settings = AppConfig()