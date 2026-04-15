class VoiceAgentError(Exception):
    """Base exception for all custom VoiceAgent errors."""
    pass

class SandboxViolationError(VoiceAgentError):
    """Raised when a file operation attempts to access paths outside the allowed output directory."""
    def __init__(self, attempted_path: str):
        super().__init__(f"Security Violation: Attempted to access unauthorized path -> {attempted_path}")

class ToolExecutionError(VoiceAgentError):
    """Raised when a specific tool fails to execute its logic."""
    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' failed: {reason}")

class AudioProcessingError(VoiceAgentError):
    """Raised when STT transcription fails or audio file is invalid."""
    pass