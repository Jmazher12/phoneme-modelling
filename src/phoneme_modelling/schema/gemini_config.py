from pathlib import Path

from pydantic import BaseModel



class GeminiConfig(BaseModel):
    project: str  # GCP project for Vertex AI
    location: str  # Vertex AI region, e.g. us-central1
    model: str
    voices: list[str]
    ref_dir: Path  # reference recordings <phoneme>.wav, sent with every prompt
    prompts_file: Path  # csv with columns phoneme, phoneme_ipa, prompt
    takes: int = 1
