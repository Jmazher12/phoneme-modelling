from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel



class AudioDistorter(ABC):

    def __init__(self, config: Optional[BaseModel] = None):
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError()

    @abstractmethod
    def is_restricted_to_phonemes(self) -> Optional[list[str]]:
        """Returns list of phoneme texts which this distortion can be used for, returns null if this is valid for any audio"""
        raise NotImplementedError()

    @abstractmethod
    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        raise NotImplementedError()

    @abstractmethod
    def distort(
        self, 
        original_file_path: Path,
        original_phoneme_text: str,
        original_phoneme_ipa: str,
        distortion_intensity: float,
        distorted_file_path: Path,
    ) -> tuple[bool, Optional[str]]:
        """
        Applies distortion to original audio with desired intensity and saves distorted audio to new file, returning 
        true if successful or false if the audio was rejected and optionally the ipa representation for the distorted
        audio.
        """
        raise NotImplementedError()
    