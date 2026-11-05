from pydantic import BaseModel



class SyntheticPhonemeMetadata(BaseModel):
    file_name: str
    file_path: str
    model: str
    voice: str
    target_phoneme_text: str
    actual_phoneme_text: str
    actual_phoneme_ipa: str
    distortion_intensity: float
    duration: float