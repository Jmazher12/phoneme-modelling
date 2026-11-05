"""Prompts shared by the LLM generators (Gemini, OpenAI). Every take sends the phoneme's reference recording, then
REF_INTRO and the phoneme's prompt from the prompts file."""
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from phoneme_modelling.util.audio import load_as_pcm16


OPENING = (
    "You are helping me build a speech therapy teaching tool. I need very short, isolated speech sounds, "
    "not words. Never say the letter name and never add a vowel afterward unless asked. Keep the clip under "
    "2 seconds. Use a clear, natural voice at a relaxed pace, as if recorded close to a good microphone in a quiet "
    "room: no background noise, music, effects or breathing sounds. The clips will be used to train an audio model, "
    "so make the sound clean and easy to hear. Reply with only the sound."
)

# Sent after the reference recording, then the phoneme's prompt. The voice comes from the generator config, so the
# recording is only a reference for which sound to make, not for voice or recording quality.
REF_INTRO = (
    "The audio is a recording of a person making the correct {name} sound, IPA {ipa}. Use it only as a reference "
    "for which sound to make. Use your own voice, and make the sound as clean and natural as possible. Don't copy "
    "the recording's voice, recording quality, length, background noise, or any vowel before or after the sound. "
    "Then do the following."
)


class PhonemePrompt(BaseModel):
    """One row of the prompts file."""
    phoneme: str
    phoneme_ipa: str
    prompt: str


class MissingReferenceError(Exception):
    """A phoneme has no reference recording or no prompt, so it can't be generated."""


def load_prompts(prompts_file: Path) -> dict[str, PhonemePrompt]:
    """The prompts file's rows, keyed by phoneme."""
    rows: list[dict] = pd.read_csv(prompts_file).to_dict(orient="records")
    return {row["phoneme"]: PhonemePrompt.model_validate(row) for row in rows}


def build_reference_message(phoneme: str, prompts: dict[str, PhonemePrompt], ref_dir: Path) -> tuple[bytes, str, Path]:
    """The reference recording ref_dir/<phoneme>.wav as 16 kHz mono PCM, the text sent after it (REF_INTRO + the
    phoneme's prompt), and the recording's path."""
    path = ref_dir / f"{phoneme}.wav"
    if not path.exists():
        raise MissingReferenceError(f"No reference recording for {phoneme} at {path}")
    if phoneme not in prompts:
        raise MissingReferenceError(f"No prompt for {phoneme} in the prompts file")
    row = prompts[phoneme]
    intro = REF_INTRO.format(name=phoneme.replace("_", " ").upper(), ipa=row.phoneme_ipa)
    return load_as_pcm16(path), f"{intro}\n\n{row.prompt}", path
