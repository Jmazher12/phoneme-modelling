import json
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from phoneme_modelling.distortion.distorter import AudioDistorter
from phoneme_modelling.phonemes import TEXT_TO_IPA
from phoneme_modelling.schema.pair_parameters import CueCheck, NuisanceSample, PairParameters
from phoneme_modelling.util.acoustics import dsp
from phoneme_modelling.util.acoustics.clip import Clip
from phoneme_modelling.util.acoustics.delivery import OUTPUT_PEAK, SAMPLE_RATE, as_clip, deliver, save_wav, stable_rng
from phoneme_modelling.util.acoustics.measures import RATIO_MEASURES, SOUNDS, measure_as, measure_sound, segment_sound
from phoneme_modelling.util.acoustics.retime import MAX_SEMITONES, guard, retime, shift_pitch
from phoneme_modelling.util.loaders import load_as_numpy_array

PARAMETERS_FOLDER = Path(__file__).resolve().parents[3] / "config" / "distortion"
DEFAULT_F0 = 150.0
INTENSITY_POINTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MATCHED_INTENSITIES = [0.0, 0.6, 1.0]
DURATION_JITTER = 0.1
LEVEL_JITTER_DB = 2.0
MIN_GAIN_DB = -30.0
MAX_PEAK_DBFS = -1.0
MAX_GUARD_RETRIES = 3


class PairDistorter(AudioDistorter):
    """Turns a clean recording of FROM_PHONEME into a graded error toward TO_PHONEME.

    Subclasses are one specific pair: they set the two sounds, a name, and the edit in ``_apply``. With MATCH_NUISANCE,
    the clip also takes the length, pitch and level of the other label, so those can't give the label away.
    """

    FROM_PHONEME: str
    TO_PHONEME: str
    MATCH_NUISANCE = False

    def __init__(self, config: Optional[BaseModel] = None):
        self.__pair_name = f"{self.FROM_PHONEME}_to_{self.TO_PHONEME}"
        parameters_text = (PARAMETERS_FOLDER / f"{self.__pair_name}.json").read_text(encoding="utf-8")
        self.parameters = PairParameters.model_validate_json(parameters_text)
        voice_f0 = json.loads((PARAMETERS_FOLDER / "voice_f0.json").read_text(encoding="utf-8"))
        self.__voice_f0 = {voice.lower(): f0 for voice, f0 in voice_f0.items()}

    @abstractmethod
    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Returns the samples of ``clip`` with this pair's edit applied at strength ``alpha``."""
        raise NotImplementedError()

    def is_restricted_to_phonemes(self) -> Optional[list[str]]:
        """Returns list of phoneme texts which this distortion can be used for, returns null if this is valid for any audio"""
        return [self.FROM_PHONEME]

    def allows_retries(self) -> bool:
        """Returns whether the generation method is non-deterministic and can yield a different result if retried."""
        return False

    def alpha_for(self, distortion_intensity: float) -> float:
        """Calibrated edit strength: intensity n/5 reaches n/5 of the natural difference between the two sounds."""
        return float(np.interp(distortion_intensity, INTENSITY_POINTS, [0.0, *self.parameters.severity_alpha]))

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
        if self.MATCH_NUISANCE and not np.isclose(distortion_intensity, MATCHED_INTENSITIES).any():
            return False, None
        samples = load_as_numpy_array(original_file_path, SAMPLE_RATE).astype(np.float64)
        if dsp.rms(samples) == 0.0:
            return False, None
        # The leading number is the pipeline's running counter; leaving it out keeps audio independent of run order.
        seed_name = original_file_path.stem.partition("_")[2]
        is_error = distortion_intensity > 0.0
        measured_as = self.TO_PHONEME if is_error else self.FROM_PHONEME
        try:
            delivered = self.__distort_samples(samples, seed_name, distortion_intensity)
            if self.MATCH_NUISANCE:
                delivered = self.__match_nuisance(delivered, measured_as, is_error, seed_name, distortion_intensity)
        except (ValueError, IndexError, RuntimeError) as error:  # one bad clip must not stop the batch
            print(f"Rejected {original_file_path} for {self.get_name()}: {error}")
            return False, None
        if delivered is None:
            return False, None
        save_wav(distorted_file_path, delivered)
        return True, self.__distorted_ipa(distortion_intensity)

    def __distort_samples(self, samples: np.ndarray, seed_name: str, distortion_intensity: float) -> np.ndarray:
        name = f"{seed_name}_{self.__pair_name}_{distortion_intensity:g}"
        edited = samples
        if distortion_intensity > 0.0:
            clip = segment_sound(samples, SAMPLE_RATE, self.FROM_PHONEME)
            measures = measure_sound(clip, self.FROM_PHONEME)
            alpha = self.alpha_for(distortion_intensity)
            edited = self._apply(clip, measures, alpha, self.__find_voice_f0(seed_name), stable_rng(name))
        gain = OUTPUT_PEAK / (float(np.max(np.abs(samples))) + 1e-12)
        return deliver(edited, as_clip(samples), stable_rng(name)) * gain

    def __find_voice_f0(self, seed_name: str) -> float:
        """The voice is the last part of a clip's name (``{phoneme}_{voice}``); pitch can't come from voiceless sounds."""
        return self.__voice_f0.get(seed_name.rpartition("_")[2].lower(), DEFAULT_F0)

    def __distorted_ipa(self, distortion_intensity: float) -> Optional[str]:
        if distortion_intensity == 0.0:
            return None
        from_percentage = int(round((1 - distortion_intensity) * 100))
        to_percentage = int(round(distortion_intensity * 100))
        from_ipa, to_ipa = TEXT_TO_IPA[self.FROM_PHONEME], TEXT_TO_IPA[self.TO_PHONEME]
        return f"distortion[{from_ipa}({from_percentage}%)|{to_ipa}({to_percentage}%)]"

    def __match_nuisance(
        self, natural: np.ndarray, measured_as: str, is_error: bool, seed_name: str, distortion_intensity: float
    ) -> Optional[np.ndarray]:
        """Move the clip's length, pitch and level to those of a clip of the other label, shrinking the edit while it
        would change the sound's defining cues, and leaving length and pitch alone if it still does."""
        name = f"{seed_name}_{self.__pair_name}_{distortion_intensity:g}"
        before = measure_as(natural, SAMPLE_RATE, measured_as)
        own_duration = float(before["duration_ms"])
        others = self.parameters.correct_clips if is_error else self.parameters.error_clips
        if not others or not np.isfinite(own_duration):
            return None
        plan_rng = stable_rng(name, "plan")
        target = others[int(plan_rng.integers(len(others)))]
        factor = target.duration_ms * float(np.exp(plan_rng.normal(0, DURATION_JITTER))) / own_duration
        semitones = self.__plan_pitch_shift(before, measured_as, seed_name, others, plan_rng)
        level_target = others[int(plan_rng.integers(len(others)))].rms_dbfs + float(plan_rng.normal(0, LEVEL_JITTER_DB))

        checks = self.__guard_checks(is_error)
        rng = stable_rng(name, "hard")
        background = as_clip(natural)
        delivered = None
        for _ in range(MAX_GUARD_RETRIES + 1):
            try:
                edited = retime(natural, SAMPLE_RATE, measured_as, factor, rng)
                edited = shift_pitch(edited, SAMPLE_RATE, semitones) if semitones else edited
                candidate = deliver(edited, background, rng)
                if guard(before, measure_as(candidate, SAMPLE_RATE, measured_as), checks, is_error):
                    delivered = candidate
                    break
            except (ValueError, IndexError, RuntimeError):  # treated like a failed guard: retry smaller, then fall back
                pass
            factor, semitones = float(np.sqrt(factor)), semitones / 2
        if delivered is None:
            delivered = deliver(natural, background, rng)

        level = float(measure_as(delivered, SAMPLE_RATE, measured_as)["rms_dbfs"])
        headroom = MAX_PEAK_DBFS - 20 * np.log10(float(np.max(np.abs(delivered))) + 1e-12)
        gain_db = float(np.clip(level_target - level, MIN_GAIN_DB, headroom)) if np.isfinite(level) else 0.0
        return delivered * 10 ** (gain_db / 20)

    @staticmethod
    def __plan_pitch_shift(
        before: dict[str, object],
        measured_as: str,
        seed_name: str,
        others: list[NuisanceSample],
        rng: np.random.Generator,
    ) -> float:
        """Semitones to another clip of the same voice, for voiced sounds only (see ``shift_pitch``)."""
        own_f0 = float(before["f0_hz"])
        voice = seed_name.rpartition("_")[2].lower()
        same_voice = [other.f0_hz for other in others if other.voice == voice and other.f0_hz is not None]
        if not np.isfinite(own_f0) or not SOUNDS[measured_as].voiced or not same_voice:
            return 0.0
        target_f0 = same_voice[int(rng.integers(len(same_voice)))]
        return float(np.clip(12 * np.log2(target_f0 / own_f0), -MAX_SEMITONES, MAX_SEMITONES))

    def __guard_checks(self, is_error: bool) -> list[tuple[str, float, bool]]:
        checks: list[CueCheck] = self.parameters.error_checks if is_error else self.parameters.correct_checks
        return [(check.cue, check.natural_gap, check.cue in RATIO_MEASURES) for check in checks]
