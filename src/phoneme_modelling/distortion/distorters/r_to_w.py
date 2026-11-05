import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import shift_formants
from phoneme_modelling.util.acoustics.clip import Clip


class RToWDistorter(PairDistorter):
    """r said as w: raises F3 and lowers F2. F1 stays, it barely differs between the two sounds."""

    FROM_PHONEME = "r"
    TO_PHONEME = "w"

    @staticmethod
    def get_name() -> str:
        return "r_to_w"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return shift_formants(clip, measures, alpha, self.parameters, formants=(2, 3))


class RToWMatchedDistorter(RToWDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "r_to_w_matched"
