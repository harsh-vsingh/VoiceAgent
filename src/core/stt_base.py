from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

class BaseSTT(ABC):
    """Abstract interface for Speech-to-Text implementations."""
    
    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: The absolute or relative path to the audio file.
            
        Returns:
            The transcribed text.
        """
        pass