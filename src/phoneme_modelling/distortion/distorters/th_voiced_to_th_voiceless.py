import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import devoice_fricative
from phoneme_modelling.util.acoustics.clip import Clip


class ThVoicedToThVoicelessDistorter(PairDistorter):
    """voiced th said as voiceless th: removes voicing, then reshapes the hiss."""

    FROM_PHONEME = "th_voiced"
    TO_PHONEME = "th_voiceless"

    @staticmethod
    def get_name() -> str:
        return "th_voiced_to_th_voiceless"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return devoice_fricative(clip, alpha, self.parameters, rng)


class ThVoicedToThVoicelessMatchedDistorter(ThVoicedToThVoicelessDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "th_voiced_to_th_voiceless_matched"
