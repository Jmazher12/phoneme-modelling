import os
import random
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

from pydantic import BaseModel

from phoneme_modelling.phonemes import TEXT_TO_IPA, VALID_IPA_STRINGS, VALID_SOUND_TEXTS
from phoneme_modelling.schema.phoneme_distortion import PhonemeDistortionRequest
from phoneme_modelling.schema.phoneme_pipeline_config import PhonemePipelineConfig, PhonemeValidationConfig
from phoneme_modelling.schema.synthetic_phoneme_metadata import SyntheticPhonemeMetadata
from phoneme_modelling.util.audio_inspector import AudioInspector
from phoneme_modelling.util.audio_validator import PhonemeAudioValidator
from phoneme_modelling.util.retry_mechanism import GenerationRetryMechanism



class SyntheticAudioGenerationVoice(ABC):

    @abstractmethod
    def get_voice_nickname(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _generate_audio(
        self,
        target_file_path: Path,
        text: str,
        ipa: str,
        distortion_intensity: Optional[float] = None,
        distort_to_text: Optional[str] = None,
        distort_to_ipa: Optional[str] = None,
    ) -> bool:
        raise NotImplementedError()

    def run_audio_generation(
        self,
        retry_mechanism: GenerationRetryMechanism,
        target_file_path: Path,
        text: str,
        ipa: str,
        distortion_intensity: Optional[float] = None,
        distort_to_text: Optional[str] = None,
        distort_to_ipa: Optional[str] = None,
    ) -> bool:
        self.__validate_audio_generation_input(
            target_file_path=target_file_path,
            text=text,
            ipa=ipa,
            distortion_intensity=distortion_intensity,
            distort_to_text=distort_to_text,
            distort_to_ipa=distort_to_ipa
        )
        has_attempted_generation = False
        for _ in retry_mechanism.attempts(target_file_path):
            has_attempted_generation = self._generate_audio(
                target_file_path=target_file_path,
                text=text,
                ipa=ipa,
                distortion_intensity=distortion_intensity,
                distort_to_text=distort_to_text,
                distort_to_ipa=distort_to_ipa
            )
            if not has_attempted_generation:
                break
        return has_attempted_generation

    def __validate_audio_generation_input(
        self,
        target_file_path: Path,
        text: str,
        ipa: str,
        distortion_intensity: Optional[float] = None,
        distort_to_text: Optional[str] = None,
        distort_to_ipa: Optional[str] = None,
    ):
        if not target_file_path or os.path.exists(target_file_path):
            raise ValueError(f"Invalid target_file_path for synthetic audio generation sent to {self.__class__}. target_file_path: {target_file_path}")
        if not text:
            raise ValueError(f"Invalid text for synthetic audio generation sent to {self.__class__}. text: {text}")
        if not ipa or ipa not in VALID_IPA_STRINGS:
            raise ValueError(f"Invalid ipa for synthetic audio generation sent to {self.__class__}. ipa: {ipa}")
        distortion_params = (distortion_intensity, distort_to_text, distort_to_ipa)
        if 0 < sum(v is not None for v in distortion_params) < 3:
            raise ValueError(
                f"All of distortion_intensity, distort_to_text and distort_to_ipa must be provided together to use " \
                f"distortions. distortion_intensity: {distortion_intensity}, distort_to_text: {distort_to_text}, " \
                f"distort_to_ipa: {distort_to_ipa}"
            )
        if distortion_intensity is not None and not (0.0 <= distortion_intensity <= 1.0):
            raise ValueError(f"distortion_intensity must be between 0.0 and 1.0. distortion_intensity: {distortion_intensity}")
        if distort_to_ipa is not None and distort_to_ipa not in VALID_IPA_STRINGS:
            raise ValueError(f"Invalid distort_to_ipa for synthetic audio generation sent to {self.__class__}. distort_to_ipa: {distort_to_ipa}")

class SyntheticAudioGenerator(ABC):

    def __init__(self, config: Optional[BaseModel] = None) -> None:
        super().__init__()
        self.__voices = self._init_voices()
        self.__number_of_digits = 3

    @abstractmethod
    def _init_voices(self) -> list[SyntheticAudioGenerationVoice]:
        raise NotImplementedError()

    @abstractmethod
    def is_distortion_capable(self) -> bool:
        """Returns whether the generation method is natively capable of generating distortions between phonemes."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_config_name(cls) -> str:
        """Returns the name used for the configs for the class."""
        raise NotImplementedError()

    @abstractmethod
    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        raise NotImplementedError()

    @staticmethod
    def __text_to_ipa(phoneme_text: str):
        if phoneme_text not in VALID_SOUND_TEXTS:
            raise ValueError(f"Requested phoneme {phoneme_text} is not part of valid sounds to generate")
        return TEXT_TO_IPA[phoneme_text]

    def __format_sound_number(self, sound_number: int) -> str:
        return f"{sound_number:0{self.__number_of_digits}d}"

    def __generate_phoneme_from_text(
        self, 
        phoneme_text: str,
        current_audio_count: int,
        target_folder: Path
    ) -> tuple[int, list[SyntheticPhonemeMetadata]]:
        generated_metadata = []
        phoneme_ipa = self.__text_to_ipa(phoneme_text)
        for voice in self.__voices:
            sound_number = self.__format_sound_number(current_audio_count)
            nickname = voice.get_voice_nickname()
            sound_file_name = f"{sound_number}_{phoneme_text}_{nickname}.wav"
            sound_file_path = Path(os.path.join(target_folder, sound_file_name))
            has_generated_file = voice.run_audio_generation(
                retry_mechanism=self.__retry_mechanism,
                target_file_path=sound_file_path,
                text=phoneme_text,
                ipa=phoneme_ipa
            )
            if has_generated_file:
                current_audio_count += 1
                audio_info = AudioInspector.get_info(sound_file_path)
                generated_audio_info = SyntheticPhonemeMetadata(
                    file_name=sound_file_name,
                    file_path=str(sound_file_path),
                    target_phoneme_text=phoneme_text,
                    actual_phoneme_text=phoneme_text,
                    actual_phoneme_ipa=phoneme_ipa,
                    distortion_intensity=0.0,
                    duration=audio_info.duration
                )
                generated_metadata.append(generated_audio_info)
        return current_audio_count, generated_metadata

    def __sample_distortion_intensities(self, phoneme_distortion: PhonemeDistortionRequest) -> list[float]:
        intensities = phoneme_distortion.distortion_intensities
        min_intensity = phoneme_distortion.min_distortion_intensity
        max_intensity = phoneme_distortion.max_distortion_intensity
        number_of_samples = phoneme_distortion.number_of_samples
        if intensities is not None and (min_intensity or max_intensity or number_of_samples):
            raise ValueError(f"Cannot receive both fixed distortion intensity and sample from range. Received: {phoneme_distortion}")
        if intensities is not None:
            return intensities
        if min_intensity is None or max_intensity is None or number_of_samples is None:
            raise ValueError(f"All of min_intensity, max_intensity and number_of_samples are needed for distortion sampling. Received: {phoneme_distortion}")
        if not (0.0 <= min_intensity <= max_intensity <= 1.0):
            raise ValueError(f"Received invalid distortion intensity range. min_intensity: {min_intensity:.2f} max_intensity: {max_intensity:.2f}")
        if number_of_samples < 1:
            raise ValueError(f"Received invalid distortion number of samples. number_of_samples: {number_of_samples}")
        intensities = []
        for _ in range(number_of_samples):
            while True:
                intensity = random.uniform(min_intensity, max_intensity)
                if intensity not in intensities:
                    break
            intensities.append(intensity)
        intensities.sort()
        return intensities

    def __generate_phoneme_distortion(
        self, 
        phoneme_distortion: PhonemeDistortionRequest,
        current_audio_count: int,
        target_folder: Path
    ) -> tuple[int, list[SyntheticPhonemeMetadata]]:
        generated_metadata = []
        distortion_intensities = self.__sample_distortion_intensities(phoneme_distortion)
        from_phoneme = phoneme_distortion.from_phoneme
        to_phoneme = phoneme_distortion.to_phoneme
        from_phoneme_ipa = self.__text_to_ipa(from_phoneme)
        to_phoneme_ipa = self.__text_to_ipa(to_phoneme)
        for voice in self.__voices:
            for distortion_intensity in distortion_intensities:
                sound_number = self.__format_sound_number(current_audio_count)
                nickname = voice.get_voice_nickname()
                from_percentage = int(round((1-distortion_intensity)*100))
                to_percentage = int(round(distortion_intensity*100))
                from_percentage_text = f"{from_percentage:03d}"
                to_percentage_text = f"{to_percentage:03d}"
                sound_file_name = f"{sound_number}_{from_phoneme}_{from_percentage_text}_{to_phoneme}_{to_percentage_text}_{nickname}.wav"
                sound_file_path = Path(os.path.join(target_folder, sound_file_name))
                has_generated_file = voice.run_audio_generation(
                    retry_mechanism=self.__retry_mechanism,
                    target_file_path=sound_file_path,
                    text=from_phoneme,
                    ipa=from_phoneme_ipa,
                    distort_to_text=to_phoneme,
                    distort_to_ipa=to_phoneme_ipa,
                    distortion_intensity=distortion_intensity
                )
                if has_generated_file:
                    current_audio_count += 1
                    dominant_phoneme_text = from_phoneme
                    if distortion_intensity > 0.5:
                        dominant_phoneme_text = to_phoneme
                    distortion_ipa = f"distortion[{from_phoneme_ipa}({from_percentage}%)|{to_phoneme_ipa}({to_percentage}%)]"
                    audio_info = AudioInspector.get_info(sound_file_path)
                    generated_audio_info = SyntheticPhonemeMetadata(
                        file_name=sound_file_name,
                        file_path=str(sound_file_path),
                        target_phoneme_text=from_phoneme,
                        actual_phoneme_text=dominant_phoneme_text,
                        actual_phoneme_ipa=distortion_ipa,
                        distortion_intensity=distortion_intensity,
                        duration=audio_info.duration
                    )
                    generated_metadata.append(generated_audio_info)
        return current_audio_count, generated_metadata

    def __build_retry_mechanism(self, generation_request: PhonemePipelineConfig) -> GenerationRetryMechanism:
        config = generation_request.validation_config
        allow_retries = self.allows_retries()
        validator = PhonemeAudioValidator(config)
        retry_mechanism = GenerationRetryMechanism(config, validator, allow_retries)
        return retry_mechanism

    def generate_synthetic_audios(
        self,
        generation_request: PhonemePipelineConfig,
        current_audio_count: int,
        target_folder: Path
    ) -> tuple[int, list[SyntheticPhonemeMetadata]]:
        generated_metadata = []
        phonemes_to_generate = generation_request.phonemes_to_generate
        self.__number_of_digits = generation_request.number_of_digits
        self.__retry_mechanism = self.__build_retry_mechanism(generation_request)
        for phoneme_text in phonemes_to_generate:
            current_audio_count, phoneme_metadata = self.__generate_phoneme_from_text(
                phoneme_text=phoneme_text,
                current_audio_count=current_audio_count,
                target_folder=target_folder
            )
            generated_metadata.extend(phoneme_metadata)
        distortions_to_generate = generation_request.distortions_to_generate
        if not distortions_to_generate or not self.is_distortion_capable():
            return current_audio_count, generated_metadata
        for phoneme_distortion in distortions_to_generate:
            current_audio_count, phoneme_metadata = self.__generate_phoneme_distortion(
                phoneme_distortion=phoneme_distortion,
                current_audio_count=current_audio_count,
                target_folder=target_folder
            )
            generated_metadata.extend(phoneme_metadata)
        return current_audio_count, generated_metadata
