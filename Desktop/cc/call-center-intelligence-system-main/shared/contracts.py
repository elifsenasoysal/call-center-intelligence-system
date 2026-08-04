from pydantic import BaseModel
from typing import List, Literal


class Utterance(BaseModel):
    speaker: Literal["agent", "customer", "unknown"]
    start_time: float
    end_time: float
    text: str
    volume_db: float | None = None 


class STTOutput(BaseModel):
    file_name: str
    language: str
    duration_seconds: float
    utterances: List[Utterance]
