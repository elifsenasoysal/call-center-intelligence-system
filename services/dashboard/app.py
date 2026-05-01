import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.transcription.pipeline import get_transcription
# from services.analysis.pipeline import get_analysis


def run_pipeline(file_path: str) -> dict:
    """Runs the full pipeline on a single audio file."""
    stt_result = get_transcription(file_path)
    # llm_result = get_analysis(stt_result)
    return stt_result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <audio_file_path>")
        sys.exit(1)

    result = run_pipeline(sys.argv[1])
    print(result)
