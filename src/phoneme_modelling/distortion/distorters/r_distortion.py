import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import shift_formants
from phoneme_modelling.util.acoustics.clip import Clip


class RDistortionDistorter(PairDistorter):
    """distorted r (part way to w): raises F3 only, toward w's."""

    FROM_PHONEME = "r"
    TO_PHONEME = "r"
    PAIR_NAME = "r_distortion"
    TO_IPA = "ɹ̞"

    @staticmethod
    def get_name() -> str:
        return "r_distortion"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return shift_formants(clip, measures, alpha, self.parameters, formants=(3,))


class RDistortionMatchedDistorter(RDistortionDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "r_distortion_matched"
