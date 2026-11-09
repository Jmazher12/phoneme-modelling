import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import shift_formants
from phoneme_modelling.util.acoustics.clip import Clip


class WToRDistorter(PairDistorter):
    """w said as r: raises F2 and lowers F3."""

    FROM_PHONEME = "w"
    TO_PHONEME = "r"

    @staticmethod
    def get_name() -> str:
        return "w_to_r"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return shift_formants(clip, measures, alpha, self.parameters, formants=(2, 3))


class WToRMatchedDistorter(WToRDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "w_to_r_matched"
