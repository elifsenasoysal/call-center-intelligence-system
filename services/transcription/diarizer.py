from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")


def diarize(file_path: str) -> list:
    """
    Performs speaker diarization on an audio file using pyannote.

    Args:
        file_path: Path to the audio file.

    Returns:
        List of speaker segments, e.g.:
        [{"speaker": "agent", "start": 0.0, "end": 4.2}, ...]
    """
    raise NotImplementedError


def merge_transcription_with_diarization(
    whisper_segments: list, diarization_segments: list
) -> list:
    """
    Aligns Whisper word-level timestamps with diarization speaker segments.

    Returns:
        List of utterances matching the Utterance schema in shared/contracts.py.
    """
    raise NotImplementedError
