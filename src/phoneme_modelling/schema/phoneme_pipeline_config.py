from typing import Any, Optional

from pydantic import BaseModel

from phoneme_modelling.schema.phoneme_distortion import PhonemeDistortionRequest



class DistortionApplicationConfig(BaseModel):
    distortion_intensitites: list[float]

class PhonemePipelineConfig(BaseModel):
    number_of_digits: int
    phonemes_to_generate: list[str]
    distortions_to_generate: Optional[list[PhonemeDistortionRequest]] = None
    distortion_application_config: DistortionApplicationConfig
    generator_configs: dict[str, Any]
    distorter_configs: Optional[dict[str, Any]] = None
    evaluator_configs: Optional[dict[str, Any]] = None