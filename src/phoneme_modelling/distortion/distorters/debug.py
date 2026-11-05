


from pathlib import Path
from typing import Optional

from phoneme_modelling.distortion.distorter import AudioDistorter


class DebugDistorter(AudioDistorter):

    @staticmethod
    def get_name() -> str:
        return "debug"

    def is_restricted_to_phonemes(self) -> Optional[list[str]]:
        """Returns list of phoneme texts which this distortion can be used for, returns null if this is valid for any audio"""
        return ["r", "th_voiceless"]

    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        return False

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
        debug_text = (
            f"Running debug distorter for original_file_path: {original_file_path}, "
            f"original_phoneme_text: {original_phoneme_text}, original_phoneme_ipa: {original_phoneme_ipa}, "
            f"distortion_intensity: {distortion_intensity}, distorted_file_path: {distorted_file_path}"
        )
        print(debug_text)
        return True, None