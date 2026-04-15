import logging
from pathlib import Path
from typing import Union
from transformers import pipeline

from src.core.stt_base import BaseSTT
from src.utils.errors import AudioProcessingError
from src.config import settings

logger = logging.getLogger(__name__)

class HuggingFaceSTT(BaseSTT):
    def __init__(self, model_id: str = None):
        """
        Initializes the Hugging Face automatic speech recognition pipeline locally.
        """
        self.model_id = model_id or f"openai/whisper-{settings.DEFAULT_STT_MODEL}"
        
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device_map="auto" 
            )
        except Exception as e:
            raise AudioProcessingError(f"Failed to load Hugging Face model '{self.model_id}': {e}")

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        try:
            result = self.pipe(str(audio_path))
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"STT Error on {audio_path}: {e}")
            raise AudioProcessingError(f"Transcription failed: {str(e)}")