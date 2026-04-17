from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseSettings):
    BASE_DIR: Path = BASE_DIR
    OUTPUT_DIR: Path = BASE_DIR / "output"

    DB_PATH: Path = BASE_DIR / "sessions.db"
    MAX_AGENT_STEPS: int = 20
    HIGH_RISK_TOOLS: tuple[str, ...] = ("delete_file", "delete_directory", "edit_file")

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Split models
    ROUTER_LLM: str = "qwen2.5:7b-instruct-q4_K_M"
    GENERATION_LLM: str = "qwen2.5:7b-instruct-q4_K_M"
    CODE_LLM: str = "qwen2.5-coder:7b"              # (NEW: used for code gen)


    # Optional backward compatibility (if some files still use DEFAULT_LLM)
    DEFAULT_LLM: str = "qwen2.5:7b-instruct-q4_K_M"

    DEFAULT_STT_MODEL: str = "base"
    MAX_RECURSION_LIMIT: int = 25

    UI_DETAILS_EXPANDED_DEFAULT: bool = False
    UI_SHOW_SANDBOX_TREE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

settings = AppConfig()