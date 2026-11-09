import os
from typing import Any, Optional

import pandas as pd
from pandas import DataFrame
from pathlib import Path

from phoneme_modelling.distortion.distorters.debug import DebugDistorter
from phoneme_modelling.distortion.distorter import AudioDistorter
from phoneme_modelling.distortion.distorter_factory import AudioDistorterFactory
from phoneme_modelling.distortion.pair_registry import PAIR_DISTORTERS
from phoneme_modelling.schema.phoneme_pipeline_config import DistortionApplicationConfig, PhonemePipelineConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.audio_inspector import AudioInspector
from phoneme_modelling.util.audio_validator import PhonemeAudioValidator
from phoneme_modelling.util.retry_mechanism import GenerationRetryMechanism


class AudioDistortionPipeline():

    @staticmethod
    def build_distorters(configs: dict[str, Any]) -> list[AudioDistorter]:
        factory = AudioDistorterFactory()
        factory.register(DebugDistorter, None)
        for pair_distorter in PAIR_DISTORTERS:
            factory.register(pair_distorter, None)
        return factory.generate(configs)

    @classmethod
    def run(
        cls,
        pipeline_config: PhonemePipelineConfig,
        target_folder: Path,
        audio_metadata_file_path: Path,
        current_audio_count: int,
    ) -> None:
        distortion_metadata: list[SyntheticPhonemeMetadata] = []
        distorter_configs = pipeline_config.distorter_configs
        if distorter_configs is None:
            return
        distorters = cls.build_distorters(distorter_configs)
        number_of_digits = pipeline_config.number_of_digits
        audio_metadata: list[dict] = pd.read_csv(audio_metadata_file_path).to_dict(orient="records")
        for audio_dict in audio_metadata:
            audio_info = SyntheticPhonemeMetadata.model_validate(audio_dict)
            current_audio_count, metadatas = cls.__generate_distortions_for_audio(
                audio_info=audio_info,
                distorters=distorters,
                pipeline_config=pipeline_config,
                target_folder=target_folder,
                number_of_digits=number_of_digits,
                current_audio_count=current_audio_count
            )
            distortion_metadata.extend(metadatas)
        distortion_metadata_dict = [entry.model_dump() for entry in distortion_metadata]
        audio_metadata.extend(distortion_metadata_dict)
        DataFrame(audio_metadata).to_csv(audio_metadata_file_path, index=False)

    @classmethod
    def __generate_distortions_for_audio(
        cls,
        audio_info: SyntheticPhonemeMetadata,
        distorters: list[AudioDistorter],
        pipeline_config: PhonemePipelineConfig,
        target_folder: Path,
        number_of_digits: int,
        current_audio_count: int
    ) -> tuple[int, list[SyntheticPhonemeMetadata]]:
        original_file_path = Path(audio_info.file_path)
        if not original_file_path.is_file():
            return current_audio_count, []
        generated_metadata = []
        original_phoneme_text = audio_info.actual_phoneme_text
        for distorter in distorters:
            valid_phonemes = distorter.is_restricted_to_phonemes()
            if valid_phonemes is not None and original_phoneme_text not in valid_phonemes:
                continue
            retry_mechanism = cls.__build_retry_mechanism(pipeline_config, distorter)
            current_audio_count, metadatas = cls.__generate_different_distortion_intensities(
                retry_mechanism=retry_mechanism,
                audio_info=audio_info,
                distorter=distorter,
                run_config=pipeline_config.distortion_application_config,
                target_folder=target_folder,
                number_of_digits=number_of_digits,
                current_audio_count=current_audio_count
            )
            generated_metadata.extend(metadatas)
        return current_audio_count, generated_metadata

    @staticmethod
    def __build_retry_mechanism(
        pipeline_config: PhonemePipelineConfig,
        distorter: AudioDistorter
    ) -> GenerationRetryMechanism:
        config = pipeline_config.validation_config
        allow_retries = distorter.allows_retries()
        validator = PhonemeAudioValidator(config)
        retry_mechanism = GenerationRetryMechanism(config, validator, allow_retries)
        return retry_mechanism

    @classmethod
    def __generate_different_distortion_intensities(
        cls,
        retry_mechanism: GenerationRetryMechanism,
        audio_info: SyntheticPhonemeMetadata,
        distorter: AudioDistorter,
        run_config: DistortionApplicationConfig,
        target_folder: Path,
        number_of_digits: int,
        current_audio_count: int
    ) -> tuple[int, list[SyntheticPhonemeMetadata]]:
        number_of_intensity_digits = len(str(len(run_config.distortion_intensitites)))
        intensity_count = 1
        generated_metadata = []
        for intensity in run_config.distortion_intensitites:
            sound_number = f"{current_audio_count:0{number_of_digits}d}"
            intensity_number = f"{intensity_count:0{number_of_intensity_digits}d}"
            has_generated, metadata = cls.__apply_distorter(
                distorter=distorter,
                retry_mechanism=retry_mechanism,
                intensity=intensity,
                audio_info=audio_info,
                target_folder=target_folder,
                sound_number=sound_number,
                intensity_number=intensity_number
            )
            if not has_generated or metadata is None:
                continue
            current_audio_count += 1
            intensity_count += 1
            generated_metadata.append(metadata)
        return current_audio_count, generated_metadata

    @classmethod
    def __apply_distorter(
        cls,
        distorter: AudioDistorter,
        retry_mechanism: GenerationRetryMechanism,
        intensity: float,
        audio_info: SyntheticPhonemeMetadata,
        target_folder: Path,
        sound_number: str,
        intensity_number: str
    ) -> tuple[bool, Optional[SyntheticPhonemeMetadata]]:
        original_file_path = Path(audio_info.file_path)
        original_phoneme_text = audio_info.actual_phoneme_text
        original_phoneme_ipa = audio_info.actual_phoneme_ipa
        target_file_name = cls.__build_file_name(distorter, audio_info, sound_number, intensity_number)
        target_file_path = Path(os.path.join(target_folder, target_file_name))
        has_generated = False
        distorted_ipa = None
        for _ in retry_mechanism.attempts(target_file_path):
            has_generated, distorted_ipa = distorter.distort(
                original_file_path=original_file_path,
                original_phoneme_text=original_phoneme_text,
                original_phoneme_ipa=original_phoneme_ipa,
                distortion_intensity=intensity,
                distorted_file_path=target_file_path,
            )
            if not has_generated:
                return False, None
        if not retry_mechanism.is_success(target_file_path):
            return False, None
        if distorted_ipa is None:
            distorted_ipa = original_phoneme_ipa
        inspected_info = AudioInspector.get_info(target_file_path)
        metadata = SyntheticPhonemeMetadata(
            file_name=target_file_name,
            file_path=str(target_file_path),
            model=audio_info.model,
            voice=audio_info.voice,
            target_phoneme_text=audio_info.target_phoneme_text,
            actual_phoneme_text=original_phoneme_text,
            actual_phoneme_ipa=distorted_ipa,
            distortion_intensity=intensity,
            duration=inspected_info.duration
        )
        return has_generated, metadata

    @staticmethod
    def __build_file_name(
        distorter: AudioDistorter,
        audio_info: SyntheticPhonemeMetadata,
        sound_number: str,
        intensity_number: str
    ) -> str:
        distortion_name = distorter.get_name()
        _, _, base_file_name = audio_info.file_name.replace(".wav", "").partition("_")
        return f"{sound_number}_{base_file_name}_{distortion_name}_distortion_{intensity_number}.wav"