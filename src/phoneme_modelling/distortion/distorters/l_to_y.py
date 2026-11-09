import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import shift_formants
from phoneme_modelling.util.acoustics.clip import Clip


class LToYDistorter(PairDistorter):
    """l said as y: lowers F1, raises F2 and raises F3. Measured on 2 voices only."""

    FROM_PHONEME = "l"
    TO_PHONEME = "y"

    @staticmethod
    def get_name() -> str:
        return "l_to_y"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return shift_formants(clip, measures, alpha, self.parameters, formants=(1, 2, 3))


class LToYMatchedDistorter(LToYDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "l_to_y_matched"
