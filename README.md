# Call Center Intelligence System

```
call-center-intelligence-system/
│
├── services/                           # Core service modules
│   ├── __init__.py
│   │
│   ├── transcription/                  # Speech-to-Text service ✅ Functional
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Main entry point: get_transcription()
│   │   ├── preprocessor.py             # Audio preprocessing (mono, 16kHz, noise reduction)
│   │   ├── diarizer.py                 # Speaker diarization + timestamp merge
│   │   └── requirements.txt            # Service dependencies
│   │
│   ├── analysis/                       # LLM-based analysis service 🔧 In development
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Main entry point: get_analysis() (NotImplemented)
│   │   ├── requirements.txt            # Service dependencies
│   │   └── training/                   # Model fine-tuning infrastructure
│   │       ├── README.md               # Training execution notes
│   │       ├── data_prep.py            # Dataset translation script (EN→TR, Llama 3)
│   │       ├── train.py                # LoRA fine-tuning script (Llama 3 8B)
│   │       └── train_data.jsonl        # Training dataset (instruction/input/output format)
│   │
│   ├── retrieval/                      # RAG service 📋 Planned
│   │   └── __init__.py
│   │
│   └── dashboard/                      # Pipeline orchestration + UI 📋 Skeleton
│       ├── __init__.py
│       ├── app.py                      # CLI entry point: run_pipeline()
│       └── requirements.txt
│
├── shared/                             # Shared modules between services
│   ├── __init__.py
│   ├── contracts.py                    # Pydantic data models (STTOutput, Utterance)
│   └── mock_data/                      # Sample outputs (for development & testing)
│       ├── sample_stt_output.json      # Example transcription output
│       └── sample_llm_output.json      # Example analysis output
│
├── data/                               # Data directory
│   ├── raw/                            # Raw audio files (git-ignored)
│   ├── processed/                      # Processed audio files (git-ignored)
│   └── turkish_telecom_dataset.csv     # Turkish-translated call center dataset
│
├── .env.example                        # Environment variable template
├── .gitignore
└── README.md
```

---

## File Descriptions

### `services/transcription/` — Speech-to-Text Service

| File | Description |
|------|-------------|
| `pipeline.py` | Main entry point. The `get_transcription(file_path)` function runs the entire STT pipeline sequentially: preprocess → Whisper STT → diarization → merge. Performs device detection (CUDA > MPS > CPU). Since Whisper does not support float64 on MPS, it falls back to CPU on MPS devices. Returns the output as a dictionary matching the STTOutput schema. |
| `preprocessor.py` | Prepares raw audio files for Whisper: converts audio to mono, resamples to 16kHz, and applies spectral gating-based noise reduction using noisereduce. Saves the output as `.wav` under `data/processed/`. Note: Uses pydub's internal `_spawn()` method, which may change in future versions. |
| `diarizer.py` | Handles two tasks: (1) `diarize()` — performs speaker diarization using pyannote 3.1, labels SPEAKER_01 as the agent and the rest as customers. (2) `merge_transcription_with_diarization()` — matches Whisper segments with pyannote segments based on timestamp overlap and adds dBFS-based audio volume levels from the raw audio file to each utterance. |
| `requirements.txt` | pydub, noisereduce, openai-whisper, pyannote.audio, python-dotenv, pydantic, ffmpeg-python |

### `services/analysis/` — LLM-Based Analysis Service

| File | Description |
|------|-------------|
| `pipeline.py` | The `get_analysis(stt_output)` function is defined but not implemented yet (`NotImplementedError`). It will process transcription output and return sentiment analysis, complaint category, summary, and agent performance scores. |
| `requirements.txt` | transformers, peft, bitsandbytes, datasets, accelerate, torch |
| `training/data_prep.py` | Downloads the `telecom-customer-support-synthetic-replicas` dataset from HuggingFace, translates English dialogues into Turkish using Llama 3 8B (4-bit), and saves the result as a CSV file. Requires GPU (CUDA). |
| `training/train.py` | Fine-tunes a Llama 3 8B model on Turkish call center data using LoRA adapters. Uses Unsloth + SFTTrainer. Memory optimized with 4-bit quantization and gradient checkpointing. Requires GPU (CUDA). |
| `training/train_data.jsonl` | Fine-tuning dataset in instruction/input/output JSONL format. |

### `services/retrieval/` — RAG Service (Planned)

Currently contains only `__init__.py`. Planned future features include vector search over historical call records, similar case retrieval, and knowledge base integration using RAG (Retrieval-Augmented Generation).

### `services/dashboard/` — Orchestration & UI

| File | Description |
|------|-------------|
| `app.py` | Currently a simple CLI-based orchestrator. The `run_pipeline(file_path)` function executes the transcription pipeline. Analysis integration is currently commented out. Planned to be converted into a Streamlit/Gradio-based web UI in the future. |
| `requirements.txt` | Currently only includes python-dotenv. Streamlit/Gradio will be added later. |

### `shared/` — Shared Modules

| File | Description |
|------|-------------|
| `contracts.py` | Data contracts shared across services. `Utterance`: a single speech segment (speaker, start/end time, text, volume_db). `STTOutput`: full transcription output (file_name, language, duration, utterance list). Uses Pydantic BaseModel. |
| `mock_data/sample_stt_output.json` | Example output from the transcription service. Contains a Turkish call center dialogue with agent/customer labels and volume_db values. Intended for development and testing purposes. |
| `mock_data/sample_llm_output.json` | Example target output for the analysis service. Includes fields such as overall_sentiment, sentiment_score, complaint_category, keywords, summary, and agent_performance_score. |

### `data/` — Data Directory

| Directory/File | Description |
|----------------|-------------|
| `raw/` | Raw audio files are placed here. Git-ignored — files are not included in the repository. |
| `processed/` | Outputs generated by the preprocessor are stored here (mono 16kHz WAV). Git-ignored. |
| `turkish_telecom_dataset.csv` | Turkish call center dataset generated using `data_prep.py`. Contains original English dialogues and Turkish translations produced with Llama 3, along with category and similarity scores. |

### Root Files

| File | Description |
|------|-------------|
| `.env.example` | Environment variable template. Includes `HUGGINGFACE_TOKEN` (required for the pyannote model) and `WHISPER_MODEL_SIZE` (default: large-v3). |
| `.gitignore` | Ignores .env, \_\_pycache\_\_, venv, data/raw, data/processed, model files (.pt, .bin), and IDE directories. |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_TOKEN` | ✅ | HuggingFace access token required to download the pyannote diarization model |
| `WHISPER_MODEL_SIZE` | ❌ | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). Default: `large-v3` |

---

## Current Status

| Module | Status | Notes |
|--------|--------|-------|
| Transcription | ✅ Functional | Preprocessing + Whisper + Diarization + Volume analysis |
| Analysis | 🔧 In Development | Fine-tuning infrastructure ready, pipeline integration pending |
| Retrieval | 📋 Planned | RAG-based similar call retrieval |
| Dashboard | 📋 Skeleton | CLI orchestrator exists, UI not implemented yet |