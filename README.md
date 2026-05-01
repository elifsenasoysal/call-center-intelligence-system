# Call Center Intelligence System

## Project Structure

```
├── services/
│   ├── transcription/                 # Speech-to-text service
│   │   ├── pipeline.py                # Main entry point: get_transcription()
│   │   ├── preprocessor.py            # Audio normalization (mono, 16 kHz, denoise)
│   │   ├── diarizer.py                # Speaker diarization & timestamp alignment
│   │   └── requirements.txt
│   │
│   ├── analysis/                      # LLM-based analysis service
│   │   ├── pipeline.py                # Main entry point: get_analysis()
│   │   └── requirements.txt
│   │
│   ├── retrieval/                     # RAG service (planned)
│   │   └── __init__.py
│   │
│   └── dashboard/                     # Pipeline orchestration & UI
│       ├── app.py
│       └── requirements.txt
│
├── shared/
│   ├── contracts.py                   # Shared data models (STTOutput, Utterance)
│   └── mock_data/
│       ├── sample_stt_output.json     # Example transcription output
│       └── sample_llm_output.json     # Example analysis output
│
├── data/
│   ├── raw/                           # Raw audio files (git-ignored)
│   └── processed/                     # Processed outputs (git-ignored)
│
├── .env.example                       # Environment variables template
├── .gitignore
└── README.md
```
