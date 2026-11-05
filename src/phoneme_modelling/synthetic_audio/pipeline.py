from typing import Any

from pandas import DataFrame
from pathlib import Path

from phoneme_modelling.schema.gemini_config import GeminiConfig
from phoneme_modelling.schema.openai_config import OpenAIConfig
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.synthetic_audio.generators.gemini import GeminiAudioGenerator
from phoneme_modelling.synthetic_audio.generators.openai_realtime import OpenAIAudioGenerator
from phoneme_modelling.synthetic_audio.generation import SyntheticAudioGenerator
from phoneme_modelling.synthetic_audio.generator_factory import SyntheticAudioGeneratorFactory


class SyntheticAudioPipeline():

    @staticmethod
    def build_generators(configs: dict[str, Any]) -> list[SyntheticAudioGenerator]:
        factory = SyntheticAudioGeneratorFactory()
        factory.register(GeminiAudioGenerator, GeminiConfig)
        factory.register(OpenAIAudioGenerator, OpenAIConfig)
        return factory.generate(configs)

    @classmethod
    def run(
        cls,
        generation_request: PhonemePipelineConfig,
        target_folder: Path,
        metadata_file_path: Path
    ) -> int:
        current_audio_count = 0
        generated_metadata: list[SyntheticPhonemeMetadata] = []
        generators = cls.build_generators(generation_request.generator_configs)
        for generator in generators:
            current_audio_count, metadata = generator.generate_synthetic_audios(
                generation_request=generation_request,
                current_audio_count=current_audio_count,
                target_folder=target_folder
            )
            generated_metadata.extend(metadata)
        DataFrame([entry.model_dump() for entry in generated_metadata]).to_csv(metadata_file_path, index=False)
        return current_audio_count
