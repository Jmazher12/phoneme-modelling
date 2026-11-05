from pathlib import Path

from pydantic import BaseModel



class OpenAIConfig(BaseModel):
    model: str  # a gpt-realtime* model
    voices: list[str]
    ref_dir: Path  # reference recordings <phoneme>.wav, sent with every prompt
    prompts_file: Path  # csv with columns phoneme, phoneme_ipa, prompt
    takes: int = 1
    min_peak: float = 0.05  # takes quieter than this (0-1) are rejected: prep would boost the noise floor into static
    env_file: Path = Path(".env")  # OPENAI_API_KEY=... (gitignored); an exported variable wins
