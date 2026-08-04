import os                           # for file path operations
import json                         # for json operations
import torch                        # for torch operations (PyTorch)
import whisper                      # for whisper operations
from dotenv import load_dotenv      # for environment variable operations

# local libraries
from services.transcription.preprocessor import preprocess_audio
from services.transcription.diarizer import diarize, merge_transcription_with_diarization

# config
load_dotenv()
model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")


def get_device() -> str:
    """Detects the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_transcription(file_path: str) -> dict:
    """
    Runs the full transcription pipeline on an audio file.

    Args:
        file_path: Path to the audio file.
                   Supported formats: .wav, .mp3, .mp4, .m4a, .ogg, .flac
                   (any format supported by ffmpeg)

    Returns:
        dict matching the STTOutput schema defined in shared/contracts.py.
    """
    device = get_device()
    print(f"[pipeline] Using device: {device}")

    # 1. preprocess audio (normalize to mono 16kHz, denoise)
    print(f"[pipeline] Preprocessing: {file_path}")
    processed_path = preprocess_audio(file_path)

    # 2. run Whisper STT
    # Whisper's word_timestamps uses float64 which MPS doesn't support, so we fall back to CPU
    whisper_device = "cpu" if device == "mps" else device
    print(f"[pipeline] Loading Whisper model: {model_size} (device: {whisper_device})")
    model = whisper.load_model(model_size, device=whisper_device)

    print("[pipeline] Transcribing...")
    # Manually set to Turkish, although it can auto-detect the language.
    result = model.transcribe(
        processed_path,
        language="tr",
        word_timestamps=True,
        fp16=device == "cuda"
    )

    # extract word-level segments from whisper output
    whisper_segments = []
    for segment in result["segments"]:
        whisper_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })
    print(f"[pipeline] Whisper found {len(whisper_segments)} segments")

    # 3. run diarization on preprocessed audio
    print("[pipeline] Running diarization...")
    diarization_segments = diarize(processed_path, device=device)
    print(f"[pipeline] Diarization found {len(diarization_segments)} speaker segments")

    # 4. merge whisper text with speaker labels (volume from raw audio)
    print("[pipeline] Merging transcription with speaker labels...")
    utterances = merge_transcription_with_diarization(
        whisper_segments, diarization_segments, raw_audio_path=file_path
    )

    # 5. build STTOutput
    from pydub import AudioSegment
    raw_audio = AudioSegment.from_file(file_path)
    duration_seconds = round(len(raw_audio) / 1000.0, 2)

    output = {
        "file_name": os.path.basename(file_path),
        "language": result.get("language", "tr"),
        "duration_seconds": duration_seconds,
        "utterances": utterances
    }

    print(f"[pipeline] Done. {len(utterances)} utterances, {duration_seconds}s total")
    return output


# test entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m services.transcription.pipeline <audio_file_path>")
        sys.exit(1)

    result = get_transcription(sys.argv[1])
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
