import os
from pathlib import Path
from typing import Callable, Generator

from phoneme_modelling.schema.phoneme_pipeline_config import PhonemeValidationConfig
from phoneme_modelling.util.audio_validator import AudioValidationError, PhonemeAudioValidator



class GenerationRetryMechanism:
    def __init__(
        self, 
        config: PhonemeValidationConfig, 
        validator: PhonemeAudioValidator,
        generator_allows_retries: bool
    ):
        self.config = config
        self.validator = validator
        self.generator_allows_retries = generator_allows_retries
        self.max_attempts = self.__determine_max_attempts()

    def attempts(self, file_path: Path) -> Generator[int, None, None]:
        """
        Yields the current attempt number and validates the generated file when control returns to the generator after 
        the loop body executes.
        """
        for attempt in range(self.max_attempts):
            if attempt > 0 and file_path.exists():
                os.remove(file_path)
            yield attempt
            try:
                if self.validator.validate(file_path):
                    break
            except AudioValidationError:
                if attempt == self.max_attempts - 1:
                    raise

    def __determine_max_attempts(self) -> int:
        max_attempts = 1
        if self.generator_allows_retries and self.config.max_number_of_retries:
            max_attempts += self.config.max_number_of_retries
        return max_attempts
