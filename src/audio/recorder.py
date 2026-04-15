import tempfile
from pathlib import Path
from typing import BinaryIO

def save_audio_to_temp(audio_file: BinaryIO, suffix: str = ".wav") -> Path:
    """
    Saves an uploaded audio stream (like Streamlit's UploadedFile) to a temporary file on disk.
    
    Args:
        audio_file: The file-like object containing audio bytes.
        suffix: The file extension to use (default .wav).
        
    Returns:
        The physical Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        return Path(tmp.name)