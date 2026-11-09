import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import shift_formants
from phoneme_modelling.util.acoustics.clip import Clip


class RToYDistorter(PairDistorter):
    """r said as y: lowers F1, raises F2 and raises F3."""

    FROM_PHONEME = "r"
    TO_PHONEME = "y"

    @staticmethod
    def get_name() -> str:
        return "r_to_y"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return shift_formants(clip, measures, alpha, self.parameters, formants=(1, 2, 3))


class RToYMatchedDistorter(RToYDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "r_to_y_matched"
