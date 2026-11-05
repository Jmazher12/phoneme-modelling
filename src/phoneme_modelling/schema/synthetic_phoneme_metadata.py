import math
from typing import Any, Optional

from pydantic import BaseModel, field_validator



class SyntheticPhonemeMetadata(BaseModel):
    file_name: str
    file_path: str
    target_phoneme_text: str
    actual_phoneme_text: str
    actual_phoneme_ipa: str
    distortion_intensity: float
    # How the clip was made. Filled by generators that report it (see SyntheticAudioGenerationVoice.get_generation_info)
    model: Optional[str] = None
    voice: Optional[str] = None
    take: Optional[int] = None
    reference_audio: Optional[str] = None
    prompt: Optional[str] = None
    transcript: Optional[str] = None
    raw_file_path: Optional[str] = None
    raw_seconds: Optional[float] = None
    cropped: Optional[bool] = None

    @field_validator("*", mode="before")
    @classmethod
    def empty_csv_cell_to_none(cls, value: Any) -> Any:
        """pandas reads empty cells as NaN; treat them as missing."""
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
