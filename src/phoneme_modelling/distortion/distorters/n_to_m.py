import numpy as np

from phoneme_modelling.distortion.pair_distorter import PairDistorter
from phoneme_modelling.distortion.recipes import change_nasal_place
from phoneme_modelling.util.acoustics.clip import Clip


class NToMDistorter(PairDistorter):
    """n said as m: reshapes the murmur. Measured on 1 voice only."""

    FROM_PHONEME = "n"
    TO_PHONEME = "m"

    @staticmethod
    def get_name() -> str:
        return "n_to_m"

    def _apply(
        self, clip: Clip, measures: dict[str, object], alpha: float, voice_f0: float, rng: np.random.Generator
    ) -> np.ndarray:
        return change_nasal_place(clip, alpha, self.parameters)


class NToMMatchedDistorter(NToMDistorter):
    """Same distortion, with length, pitch and level taken from the other label so they can't give the label away."""

    MATCH_NUISANCE = True

    @staticmethod
    def get_name() -> str:
        return "n_to_m_matched"
