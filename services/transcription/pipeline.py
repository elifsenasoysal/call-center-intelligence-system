from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")


def get_transcription(file_path: str) -> dict:
    """
    Runs the full transcription pipeline on an audio file.

    Args:
        file_path: Path to the audio file (.wav or .mp3).

    Returns:
        dict matching the STTOutput schema defined in shared/contracts.py.
    """
    raise NotImplementedError
