from pathlib import Path

from phoneme_modelling.schema.phoneme_pipeline_config import PhonemeValidationConfig
from phoneme_modelling.util.audio_inspector import AudioInspector



class AudioValidationError(Exception):
    """Exception raised when audio fails validation and interrupts are enabled."""
    pass


class PhonemeAudioValidator:
    def __init__(self, config: PhonemeValidationConfig):
        self.config = config

    def validate(self, file_path: Path) -> bool:
        """
        Validates the generated audio file.
        Returns True if valid, False if invalid. Can raise AudioValidationError if configured to interrupt generation.
        """
        if not self.__validate_audio_exists(file_path):
            return False
        duration = AudioInspector.get_info(file_path).duration  
        if not self.__validate_audio_is_long_enough(duration):
            return False
        if not self.__validate_audio_is_short_enough(duration):
            return False
        return True

    def __validate_audio_exists(self, file_path: Path) -> bool:
        if not file_path.exists():
            if self.config.interrupt_generation_on_missing_file:
                raise AudioValidationError(f"Missing generated file at {file_path}")
            return False
        try:
            AudioInspector.get_info(file_path)
        except Exception as e:
            if self.config.interrupt_generation_on_missing_file:
                raise AudioValidationError(f"Failed to read .wav file at {file_path}: {e}")
            return False
        return True

    def __validate_audio_is_long_enough(self, duration: float) -> bool:
        if self.config.min_audio_length is None:
            return True
        if duration >= self.config.min_audio_length:
            return True
        if self.config.interrupt_generation_on_short_audio:
            raise AudioValidationError(f"Audio too short: {duration:.2f}s < {self.config.min_audio_length}s")
        return False

    def __validate_audio_is_short_enough(self, duration: float) -> bool:
        if self.config.max_audio_length is None:
            return True
        if duration <= self.config.max_audio_length:
            return True
        if self.config.interrupt_generation_on_long_audio:
            raise AudioValidationError(f"Audio too long: {duration:.2f}s > {self.config.max_audio_length}s")
        return False