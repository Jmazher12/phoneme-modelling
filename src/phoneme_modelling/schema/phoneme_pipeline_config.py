from typing import Any, Optional

from pydantic import BaseModel

from phoneme_modelling.schema.phoneme_distortion import PhonemeDistortionRequest



class DistortionApplicationConfig(BaseModel):
    distortion_intensitites: list[float]


class PhonemeValidationConfig(BaseModel):
    interrupt_generation_on_missing_file: bool = False
    min_audio_length: Optional[float] = None
    interrupt_generation_on_short_audio: bool = False
    max_audio_length: Optional[float] = None
    interrupt_generation_on_long_audio: bool = False
    max_number_of_retries: Optional[int] = None


class PhonemePipelineConfig(BaseModel):
    number_of_digits: int
    phonemes_to_generate: list[str]
    distortions_to_generate: Optional[list[PhonemeDistortionRequest]] = None
    distortion_application_config: DistortionApplicationConfig
    validation_config: PhonemeValidationConfig
    generator_configs: dict[str, Any]
    distorter_configs: Optional[dict[str, Any]] = None
    evaluator_configs: Optional[dict[str, Any]] = None