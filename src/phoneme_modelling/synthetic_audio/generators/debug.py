from typing import Optional
from pathlib import Path

from phoneme_modelling.synthetic_audio.generation import SyntheticAudioGenerationVoice, SyntheticAudioGenerator



class DebugGenerationVoice(SyntheticAudioGenerationVoice):

    def get_voice_nickname(self) -> str:
        return "debug"

    def _generate_audio(
            self,
            target_file_path: Path,
            text: str,
            ipa: str,
            distortion_intensity: Optional[float] = None,
            distort_to_text: Optional[str] = None,
            distort_to_ipa: Optional[str] = None,
        ) -> bool:
            if distortion_intensity is None:
                print(f"Generating phoneme {text}({ipa}) at {target_file_path}")
                return True
            print(f"Generating distortion from {text}({ipa}) to {distort_to_text}({distort_to_ipa}) with intensity {distortion_intensity:.2f} at {target_file_path}")
            return True


class DebugGenerator(SyntheticAudioGenerator):

    def _init_voices(self) -> list[SyntheticAudioGenerationVoice]:
        return [DebugGenerationVoice()]

    def is_distortion_capable(self) -> bool:
        """Returns whether the generation method is natively capable of generating distortions between phonemes."""
        return True

    @classmethod
    def get_config_name(cls) -> str:
        return "debug"

    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        return False