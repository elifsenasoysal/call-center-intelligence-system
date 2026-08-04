import os
import numpy as np
import noisereduce as nr
from pydub import AudioSegment

def preprocess_audio(file_path: str, output_path: str = None) -> str:
    """
    Normalizes audio to mono 16 kHz and applies noise reduction.

    Args:
        file_path: Path to the raw audio file.
               Supported formats: .wav, .mp3, .mp4, .m4a, .ogg, .flac
               (any format supported by ffmpeg)
        output_path: Optional path to save the processed file.
                     If None, saves to data/processed/ with same filename.

    Returns:
        Path to the processed audio file.
    """

    # check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # load audio file
    audio = AudioSegment.from_file(file_path)

    # normalize to mono 16kHz
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    # convert to float array for noisereduce
    samples = np.array(audio.get_array_of_samples()).astype(np.float64) / (
        1 << (8 * audio.sample_width - 1)
    )
    sample_rate = audio.frame_rate

    # apply noise reduction
    reduced = nr.reduce_noise(y=samples,sr= sample_rate)

    # convert back to int16
    reduced_int16 = (reduced * (1 << (8 * audio.sample_width - 1))).astype(np.int16)
    cleaned_audio = audio._spawn(reduced_int16.tobytes())
    # !!! WARNING : '_spawn()' is an internal method of pydub and may be removed in future versions
    
    # define output path if not specified
    if output_path is None:
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        filename= os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(processed_dir, f"{filename}_preprocessed.wav")
    
    # save cleaned audio
    cleaned_audio.export(output_path, format="wav")
    # *** For Whisper, format has been set to "wav". ***

    # log the output path
    print(f"[preprocessor] Cleaned audio saved to: {output_path}")

    return output_path
