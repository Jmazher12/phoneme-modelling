import os
from typing import Any

import pandas as pd
from pandas import DataFrame
from pathlib import Path

from phoneme_modelling.distortion.distorters.debug import DebugDistorter
from phoneme_modelling.distortion.distorter import AudioDistorter
from phoneme_modelling.distortion.distorter_factory import AudioDistorterFactory
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata


class AudioDistortionPipeline():

    @staticmethod
    def build_distorters(configs: dict[str, Any]) -> list[AudioDistorter]:
        factory = AudioDistorterFactory()
        factory.register(DebugDistorter, None)
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
            original_file_path = Path(audio_info.file_path)
            if not original_file_path.is_file():
                continue
            original_phoneme_text = audio_info.actual_phoneme_text
            original_phoneme_ipa = audio_info.actual_phoneme_ipa
            _, _, base_file_name = audio_info.file_name.replace(".wav", "").partition("_")
            for distorter in distorters:
                valid_phonemes = distorter.is_restricted_to_phonemes()
                if valid_phonemes is not None and original_phoneme_text not in valid_phonemes:
                    continue
                distortion_name = distorter.get_name()
                sound_number = f"{current_audio_count:0{number_of_digits}d}"
                target_file_name = f"{sound_number}_{base_file_name}_{distortion_name}_distortion.wav"
                target_file_path = Path(os.path.join(target_folder, target_file_name))
                distortion_intensity = 0.5
                has_generated, distorted_ipa = distorter.distort(
                    original_file_path=original_file_path,
                    original_phoneme_text=original_phoneme_text,
                    original_phoneme_ipa=original_phoneme_ipa,
                    distortion_intensity=distortion_intensity,
                    distorted_file_path=target_file_path,
                )
                if not has_generated:
                    continue
                current_audio_count += 1
                if distorted_ipa is None:
                    distorted_ipa = original_phoneme_ipa
                metadata = SyntheticPhonemeMetadata(
                    file_name=target_file_name,
                    file_path=str(target_file_path),
                    target_phoneme_text=audio_info.target_phoneme_text,
                    actual_phoneme_text=original_phoneme_text,
                    actual_phoneme_ipa=distorted_ipa,
                    distortion_intensity=distortion_intensity
                )
                distortion_metadata.append(metadata)
        distortion_metadata_dict = [entry.model_dump() for entry in distortion_metadata]
        audio_metadata.extend(distortion_metadata_dict)
        DataFrame(audio_metadata).to_csv(audio_metadata_file_path, index=False)
