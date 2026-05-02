import os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

def diarize(file_path: str, agent_label: str = "SPEAKER_01", device=None) -> list:
    """
    Performs speaker diarization and maps speaker labels to roles.

    Args:
        file_path: Path to the audio file.
        agent_label: Pyannote speaker label to treat as "agent".
                     All other speakers are labeled "customer".
                     Defaults to "SPEAKER_01" (first speaker).

    Returns:
        List of dicts: [{"speaker": "agent"|"customer", "start": float, "end": float}, ...]
    """
    # Initialize the diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token
    )

    # move pipeline to device if specified
    if device:
        pipeline.to(torch.device(device))

    # perform diarization
    diarization = pipeline(file_path, min_speakers=2, max_speakers=2)
    
    # pyannote 3.1 returns DiarizeOutput; actual Annotation is in .speaker_diarization
    annotation = diarization.speaker_diarization

    # extract speaker segments
    segments = []

    for segment, _, speaker in annotation.itertracks(yield_label=True):
        role = "agent" if speaker == agent_label else "customer" #mapping
        segments.append({
            "speaker": role,
            "start": segment.start,
            "end": segment.end
        })

    return segments


def merge_transcription_with_diarization(
    whisper_segments: list, diarization_segments: list, raw_audio_path: str = None
) -> list:
    """
    Aligns Whisper word-level timestamps with diarization speaker segments.

    For each whisper segment, finds the diarization segment with the most
    temporal overlap and assigns that segment's speaker label.

    Args:
        whisper_segments: Whisper output with word-level timestamps.
            [{"start": float, "end": float, "text": str}, ...]
        diarization_segments: Output from diarize().
            [{"speaker": str, "start": float, "end": float}, ...]
        raw_audio_path: Path to the original (non-normalized) audio file.
            Used to extract volume_db per utterance. If None, volume_db is not calculated.

    Returns:
        List of utterances matching the Utterance schema in shared/contracts.py.
    """
    # load raw audio for volume analysis (before normalization)
    raw_audio = None
    if raw_audio_path and os.path.isfile(raw_audio_path):
        raw_audio = AudioSegment.from_file(raw_audio_path)

    utterances = []

    for ws in whisper_segments:
        best_speaker = "unknown"
        best_overlap = 0.0

        for ds in diarization_segments:
            # calculate overlap between whisper segment and diarization segment
            overlap_start = max(ws["start"], ds["start"])
            overlap_end = min(ws["end"], ds["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            # comparing the overlap with the best overlap
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds["speaker"]

        # extract volume from raw audio for this segment
        volume_db = None
        if raw_audio:
            start_ms = int(ws["start"] * 1000)
            end_ms = int(ws["end"] * 1000)
            segment_audio = raw_audio[start_ms:end_ms]
            if len(segment_audio) > 0:
                volume_db = round(segment_audio.dBFS, 2)

        utterances.append({
            "speaker": best_speaker,    #pyannote labels mapped to agent/customer
            "start_time": ws["start"],  #whisper's word-level timestamps
            "end_time": ws["end"],      #whisper's word-level timestamps
            "text": ws["text"],         #whisper's word-level text
            "volume_db": volume_db      #loudness from raw audio (before normalization)
        })

    return utterances
